"""Scheduler-process hooks: lease-bounded matching, renewal, billing. Installed on every rank.

Ranks stay identical without extra broadcasts: the only clock is the HTTP arrival stamp
carried by each request, and SGLang already delivers the same request list to every rank.
"""
from dataclasses import replace
from functools import wraps
import logging
import os

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
            tag = node._ttl = (now, now + DEFAULT_TTL)
        rank = 0 if now >= tag[0] + STATE.table.keep else 2 if now < tag[1] else 1
        return rank, self.base.get_priority(node)


def tag_path(node_id, now, expires):
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
        node._ttl = (now, max(expires, tag[1]) if tag else expires)
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
        tag_path(node, STATE.now, req._ttl_expires)
    if METRICS[0] is not None:
        METRICS[0].record(req, usage, physical, idle)


def install_classes(cache_class, req_class, batch_class, scheduler_class, streamer_class):
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
        reuse_limit(req)
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
        STATE.counters['swept_rows'] += STATE.table.sweep(STATE.now)
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
            tag_path(getattr(req, 'last_node', None), STATE.now, expires)
        return result

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
    install_classes(UnifiedRadixCache, Req, ScheduleBatch, Scheduler, streamer)
    LOG.info('qwen_ttl hooks installed (keep=%ss)', STATE.table.keep)
