"""Scheduler-process hooks: lease-bounded matching, renewal, billing. Installed on every rank.

Ranks stay identical without extra broadcasts: the only clock is the HTTP arrival stamp
carried by each request, and SGLang already delivers the same request list to every rank.
"""
from array import array
from dataclasses import replace
from functools import wraps
import heapq
import logging
import os
import time

from qwen_ttl.core import DEFAULT_TTL, LeaseTable, bill, decode_carrier, page_keys, readable_pages

LOG = logging.getLogger(__name__)
COUNTERS = ('requests', 'requests_1h', 'physical_tokens', 'read_tokens', 'expired_reuse_tokens',
            'unbacked_read_tokens', 'write_5m_tokens', 'write_1h_tokens', 'input_tokens', 'swept_rows')


class State:
    def __init__(self):
        self.table = LeaseTable(int(os.environ.get('QWEN_TTL_KEEP_SECONDS', '7200')))
        self.now = 0.0          # watermark of request arrival stamps; never moves backwards
        # Billing and lease granularity (Kimi caches 256-token blocks); a multiple of the engine page.
        self.page_size = int(os.environ.get('QWEN_TTL_BLOCK_TOKENS', '256'))
        self.counters = dict.fromkeys(COUNTERS, 0)
        self.core = None
        self.cache = None
        # L3 (file) tier: only valid 1h leases are written, at the L2 eviction tail, within a token budget.
        self.l3_rate = float(os.environ.get('QWEN_TTL_L3_TOKENS_PER_SECOND', '0'))
        if os.environ.get('QWEN_TTL_L3_WRITES') == '0':   # diagnosis: storage attached, nothing written
            self.l3_rate = 0.0
        self.l3_burst = float(os.environ.get('QWEN_TTL_L3_BURST_TOKENS', '0'))
        self.l3_budget = self.l3_burst
        self.l3_refilled = 0.0
        self.l3_writing = False
        self.l3_keys = set()
        self.last_1h = float('-inf')
        self.purged = time.monotonic()


STATE = State()


class Metrics:
    """Rank 0 exports; every rank computes the same numbers."""

    def __init__(self):
        from prometheus_client import Counter, Gauge, Histogram
        self.tokens = Counter('qwen_ttl_tokens', 'Prompt tokens by TTL accounting class', ['kind'])
        self.requests = Counter('qwen_ttl_requests', 'Admitted requests by requested tier', ['tier'])
        self.rows = Gauge('qwen_ttl_lease_rows', 'Lease table rows', multiprocess_mode='livesum')
        self.idle = Histogram('qwen_ttl_idle_seconds', 'Idle time of a reused prefix when it is used again', ['kind'],
                              buckets=(10, 30, 60, 120, 300, 600, 1200, 1800, 3600, 5400, 7200))

    def record(self, req, usage, physical, idle):
        self.requests.labels('1h' if req._ttl_seconds == 3600 else '5m').inc()
        for kind, value in (('read', usage['read']), ('write_5m', usage['write_5m']), ('write_1h', usage['write_1h']),
                            ('input', usage['input']), ('physical', physical),
                            ('expired_reuse', max(physical - usage['read'], 0)),
                            ('unbacked_read', max(usage['read'] - physical, 0))):
            if value:
                self.tokens.labels(kind).inc(value)
        if idle is not None and physical:
            self.idle.labels('hit' if usage['read'] else 'expired_reuse').observe(idle)
        self.rows.set(len(STATE.table.rows))


METRICS = [None]


class TTLStrategy:
    """Evict beyond-retention data first, then expired leases, then the native order."""

    def __init__(self, base):
        self.base = base

    def get_priority(self, node):
        tag, now = getattr(node, '_ttl', None), STATE.now
        if tag is None:
            # First seen by an eviction pass shortly after creation: start a default lease clock.
            tag = node._ttl = (now, now + DEFAULT_TTL, DEFAULT_TTL)
        rank = 0 if now >= tag[0] + STATE.table.keep else 2 if now < tag[1] else 1
        return rank, self.base.get_priority(node)


def tag_path(node_id, now, expires, tier):
    """Stamp the tree path a request uses; requests hold node ids, not nodes."""
    core = STATE.core
    if core is None or node_id is None:
        return
    try:
        node = node_id if hasattr(node_id, 'parent') else core.node_by_id(node_id)
    except (KeyError, IndexError, AttributeError):
        return
    while node is not None:
        tag = getattr(node, '_ttl', None)
        node._ttl = (now, expires, tier) if tag is None or expires >= tag[1] else (now, tag[1], tag[2])
        node = node.parent


def prompt_keys(req):
    keys = getattr(req, '_ttl_keys', None)
    if keys is None:
        keys = req._ttl_keys = page_keys(req.origin_input_ids, STATE.page_size, req.cache_salt, req.extra_key)
    return keys


def own_prefix(req):
    # An active request keeps its own computed state across chunks and retraction.
    return max(getattr(req, 'already_computed', 0), getattr(getattr(req, 'kv', None), 'cache_inserted_len', 0))


def reuse_limit(req):
    """Tokens of this request's prompt that retention still allows any tier to serve."""
    _, reusable = STATE.table.lengths(prompt_keys(req), STATE.now)
    limit = max(reusable * STATE.page_size, own_prefix(req))
    req._ttl_limit = limit
    return limit


def l3_take(tokens):
    """Token bucket on the request clock, so every rank spends the same budget."""
    now = STATE.now
    if STATE.l3_refilled:
        STATE.l3_budget = min(STATE.l3_burst, STATE.l3_budget + (now - STATE.l3_refilled) * STATE.l3_rate)
    STATE.l3_refilled = now
    if tokens > STATE.l3_budget:
        return False
    STATE.l3_budget -= tokens
    return True


def backup_tail(core, num_tokens):
    """Before L2 evicts, persist the valid 1h leaves at its tail. The write pins them past this pass."""
    now, covered = STATE.now, 0
    for node in heapq.nsmallest(64, core.evictable_host_leaves, key=core.eviction_strategy.get_priority):
        if covered >= 2 * num_tokens:
            break
        tokens = len(node.key)
        covered += tokens
        tag = node._ttl
        if tag[2] != 3600 or tag[1] - now <= 300 or getattr(node, '_ttl_l3', False):
            continue
        if not l3_take(tokens):
            count('l3_dropped', tokens)
            continue
        STATE.l3_writing = True
        try:
            STATE.cache.write_backup_storage(node.id)
        finally:
            STATE.l3_writing = False
        node._ttl_l3 = True
        remember_storage(node)
        count('l3_written', tokens)


def count(kind, tokens):
    if METRICS[0] is not None:
        METRICS[0].tokens.labels(kind).inc(tokens)


def purge_storage():
    """Unlink L3 files idle beyond retention. They are already unreachable through matching."""
    if time.monotonic() - STATE.purged < 300:
        return
    STATE.purged = time.monotonic()
    evictor = getattr(getattr(getattr(STATE.cache, 'cache_controller', None), 'storage_backend', None), '_evictor', None)
    if evictor is None or not getattr(evictor, '_eviction_enabled', False):
        return
    deadline = time.time() - STATE.table.keep
    with evictor._lock:
        for _ in range(4096):
            stem = next(iter(evictor._lru), None)
            if stem is None:
                break
            try:
                if os.path.getmtime(os.path.join(evictor.file_path, stem + '.bin')) > deadline:
                    break
            except OSError:
                pass
            if evictor._evict_one_lru_locked()[0] != 'evicted':
                break


def remember_storage(node):
    """Record the lease blocks of a path whose tail was just written to L3."""
    parts = []
    walk = node
    while walk is not None and walk.key is not None:
        parts.append(walk.key.raw_token_ids())
        walk = walk.parent
    tokens = array('q')
    for part in reversed(parts):
        tokens.extend(part)
    keys = page_keys(tokens, STATE.page_size, node.key.cache_salt, node.key.extra_key)
    start = (len(tokens) - len(node.key)) // STATE.page_size
    STATE.l3_keys.update(keys[start:])


def storage_candidate(req, matched):
    """Whether the block right after the L1+L2 match was written to L3 under a retained lease."""
    keys = prompt_keys(req)
    index = matched // STATE.page_size
    known = STATE.l3_keys
    for key in keys[index:index + 2]:
        if key in known:
            row = STATE.table.rows.get(key)
            return row is not None and STATE.now < row[0] + STATE.table.keep
    return False


def admit(req):
    """First prefill chunk of a request: renew what it uses, lease what it writes, fix its bill."""
    page, counters = STATE.page_size, STATE.counters
    physical = min(req.cached_tokens, len(req.origin_input_ids))
    keys = prompt_keys(req)
    row = STATE.table.rows.get(keys[0]) if keys else None
    idle = STATE.now - row[0] if row else None
    # Bill by lease alone: where the engine can physically resume (state checkpoints, capacity)
    # must not leak into user-visible fields. The shortfall is tracked as unbacked_read_tokens.
    read_pages, tier, billing = STATE.table.admit(
        prompt_keys(req), readable_pages(len(req.origin_input_ids), page), req._ttl_seconds, STATE.now)
    usage = bill(len(req.origin_input_ids), page, read_pages, tier)
    req._ttl_bill = usage
    req._ttl_physical = physical
    # Everything reported to the user is the billing view.
    req.cached_tokens = usage['read']
    req.cached_tokens_device, req.cached_tokens_host, req.cached_tokens_storage = usage['read'], 0, 0
    counters['requests'] += 1
    counters['requests_1h'] += req._ttl_seconds == 3600
    counters['physical_tokens'] += physical
    counters['read_tokens'] += usage['read']
    counters['expired_reuse_tokens'] += max(physical - usage['read'], 0)
    counters['unbacked_read_tokens'] += max(usage['read'] - physical, 0)
    counters['write_5m_tokens'] += usage['write_5m']
    counters['write_1h_tokens'] += usage['write_1h']
    counters['input_tokens'] += usage['input']
    req._ttl_expires = STATE.now + tier
    for node in (getattr(req, 'last_node', None), getattr(req, 'last_host_node', None)):
        tag_path(node, STATE.now, req._ttl_expires, tier)
    req._ttl_tier = tier
    if tier == 3600:
        STATE.last_1h = STATE.now
    if METRICS[0] is not None:
        METRICS[0].record(req, usage, physical, idle)


def install_classes(cache_class, req_class, batch_class, scheduler_class, streamer_class, core_class, base_component):
    if getattr(cache_class, '_qwen_ttl_hook', False):
        return
    original_init = req_class.__init__
    original_cache_init = cache_class.__init__
    original_match = cache_class.match_prefix
    original_finished = cache_class.cache_finished_req
    original_unfinished = cache_class.cache_unfinished_req
    original_max_prefix = req_class._compute_max_prefix_len
    original_extend = batch_class.prepare_for_extend
    original_prefetch = scheduler_class._prefetch_kvcache
    original_details = streamer_class.get_cached_tokens_details

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        carried = decode_carrier(self.cache_salt)
        if carried is None:
            self._ttl_seconds = DEFAULT_TTL
        else:
            self._ttl_seconds, stamp, self.cache_salt = carried
            if stamp > STATE.now:
                STATE.now = stamp

    @wraps(original_cache_init)
    def cache_init(self, *args, **kwargs):
        original_cache_init(self, *args, **kwargs)
        if STATE.page_size % self.page_size:
            raise ValueError('QWEN_TTL_BLOCK_TOKENS must be a multiple of the engine page size')
        if not self.disable:
            self.tree_core.eviction_strategy = TTLStrategy(self.tree_core.eviction_strategy)
            STATE.core = self.tree_core
            STATE.cache = self
        import torch.distributed as dist
        if self.tp_world_size == 1 or dist.get_rank(self.tp_group) == 0:
            try:
                METRICS[0] = Metrics()
            except Exception:
                LOG.exception('qwen_ttl metrics unavailable')

    @wraps(original_match)
    def match(self, params):
        req = params.req
        if req is None or not hasattr(req, '_ttl_seconds') or self.disable:
            return original_match(self, params)
        key = params.key
        limit = min(key._raw_len(), reuse_limit(req))
        # Never modify the caller's RadixKey; native prefetch and hashing may retain it.
        bounded = type(key)(key.token_ids, key.extra_key, is_bigram=key.is_bigram, limit=limit,
                            cache_salt=key.cache_salt)
        return original_match(self, replace(params, key=bounded))

    @wraps(original_max_prefix)
    def max_prefix(self, input_len):
        result = original_max_prefix(self, input_len)
        if getattr(self, '_ttl_prefetching', False):
            result = min(result, max(self._ttl_limit, own_prefix(self)))
        return result

    @wraps(original_prefetch)
    def prefetch(self, req, storage_hit_end=None):
        if not hasattr(req, '_ttl_seconds'):
            return original_prefetch(self, req, storage_hit_end)
        if not self.enable_hicache_storage:
            return None
        # Look on disk only for what L1+L2 no longer hold and a 1h lease still covers. Nearly
        # every request is fully served from memory; an unconditional lookup stalls admission.
        req.init_next_round_input(self.tree_cache, cow_mamba=False)
        matched = len(req.prefix_indices) + req.host_hit_length
        if not storage_candidate(req, matched):
            req.storage_prefetch_last_match_len = matched
            return None
        count('l3_lookup', 1)
        req._ttl_prefetching = True
        try:
            return original_prefetch(self, req, storage_hit_end)
        finally:
            req._ttl_prefetching = False

    @wraps(original_extend)
    def extend(self):
        first = [req for req in self.reqs if hasattr(req, '_ttl_seconds') and not hasattr(req, '_ttl_bill')
                 and not req._cache_breakdown_computed and not req.retracted_stain]
        result = original_extend(self)
        for req in first:
            admit(req)
        swept = STATE.table.sweep(STATE.now)
        if swept and STATE.l3_keys:
            STATE.l3_keys.intersection_update(STATE.table.rows)
        if STATE.l3_rate:
            purge_storage()
        return result

    @wraps(original_finished)
    def finished(self, req, *args, **kwargs):
        result = original_finished(self, req, *args, **kwargs)
        is_insert = kwargs.get('is_insert', args[0] if args else True)
        if hasattr(req, '_ttl_seconds') and not self.disable and is_insert:
            # Generated tokens join the tree; keep them reusable without ever billing them as a hit.
            page, keys = STATE.page_size, prompt_keys(req)
            total = (len(req.origin_input_ids) + len(req.output_ids)) // page
            if total > len(keys):
                extra = page_keys(req.origin_input_ids + req.output_ids, page, req.cache_salt, req.extra_key,
                                  previous=keys[-1] if keys else None, start=len(keys))
                STATE.table.retain(extra[:total - len(keys)], STATE.now)
        return result

    @wraps(original_unfinished)
    def unfinished(self, req, *args, **kwargs):
        result = original_unfinished(self, req, *args, **kwargs)
        expires = getattr(req, '_ttl_expires', None)
        if expires is not None and not self.disable:
            tag_path(getattr(req, 'last_node', None), STATE.now, expires, req._ttl_tier)
        return result

    original_storage_write = cache_class.write_backup_storage

    @wraps(original_storage_write)
    def storage_write(self, node_id):
        # The native policy writes every host backup through to storage; only the TTL tail writer may.
        if STATE.l3_writing:
            return original_storage_write(self, node_id)

    original_drive = core_class.drive_host_eviction

    @wraps(original_drive)
    def drive(self, component_type, num_tokens):
        if (STATE.l3_rate and self.enable_storage and component_type == base_component
                and STATE.now - STATE.last_1h < STATE.table.keep):
            backup_tail(self, num_tokens)
        return original_drive(self, component_type, num_tokens)

    @wraps(original_details)
    def details(self, req):
        usage = getattr(req, '_ttl_bill', None)
        if usage is None:
            return original_details(self, req)
        return {'device': usage['read'], 'host': 0,
                'ttl': [usage['read'], usage['write_5m'], usage['write_1h'], len(req.origin_input_ids)]}

    req_class.__init__ = init
    req_class._compute_max_prefix_len = max_prefix
    cache_class.__init__ = cache_init
    cache_class.match_prefix = match
    cache_class.cache_finished_req = finished
    cache_class.cache_unfinished_req = unfinished
    cache_class.write_backup_storage = storage_write
    core_class.drive_host_eviction = drive
    batch_class.prepare_for_extend = extend
    scheduler_class._prefetch_kvcache = prefetch
    streamer_class.get_cached_tokens_details = details
    cache_class._qwen_ttl_hook = True


def install():
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_components import output_streamer
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    streamer = next(value for name, value in vars(output_streamer).items()
                    if isinstance(value, type) and 'get_cached_tokens_details' in vars(value))
    from sglang.srt.mem_cache.unified_cache.component_type import BASE_COMPONENT_TYPE
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    install_classes(UnifiedRadixCache, Req, ScheduleBatch, Scheduler, streamer, UnifiedTreeCore, BASE_COMPONENT_TYPE)
    LOG.info('qwen_ttl hooks installed (keep=%ss)', STATE.table.keep)
