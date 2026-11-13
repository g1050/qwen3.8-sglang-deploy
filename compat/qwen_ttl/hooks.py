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
        self.host_state_pool = None
        self.state_component = None
        self.state_pool_name = None
        self.cache = None
        # L3 (file) tier: only valid 1h leases are written, at the L2 eviction tail, within a token budget.
        self.l3_rate = float(os.environ.get('QWEN_TTL_L3_TOKENS_PER_SECOND', '0'))
        if os.environ.get('QWEN_TTL_L3_WRITES') == '0':   # diagnosis: storage attached, nothing written
            self.l3_rate = 0.0
        self.l3_burst = float(os.environ.get('QWEN_TTL_L3_BURST_TOKENS', '0'))
        self.l3_budget = self.l3_burst
        self.l3_refilled = 0.0
        self.l3_writing = False
        self.tail_demand = 0
        self.l3_pages = {}          # engine page hash on L3 -> id of the node whose write covers it
        self.l3_state_pages = set()  # page hashes stored together with a state (restore points)
        self.key_class = self.page_hashes = None
        self.last_1h = float('-inf')
        self.purged = time.monotonic()


STATE = State()


class Metrics:
    """Rank 0 exports; every rank computes the same numbers."""

    def __init__(self):
        from prometheus_client import Counter, Gauge, Histogram
        self.tokens = Counter('qwen_ttl_tokens', 'Prompt tokens by TTL accounting class', ['kind'])
        self.requests = Counter('qwen_ttl_requests', 'Admitted requests by requested tier', ['tier'])
        self.seconds = Counter('qwen_ttl_overhead_seconds', 'Scheduler time spent in TTL bookkeeping (rank 0)', ['part'])
        self.evictions = Counter('qwen_ttl_evictions', 'Evicted KV tokens / state slots by pool, tier and lease state',
                                 ['pool', 'layer', 'lease'])
        self.host_states = Gauge('qwen_ttl_host_state_slots', 'Host (L2) state-pool slots', ['state'],
                                 multiprocess_mode='livesum')
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
        pool = STATE.host_state_pool
        if pool is None and STATE.state_pool_name is not None:
            # The host pools are attached after the cache is constructed.
            group = getattr(STATE.cache, 'host_pool_group', None)
            if group is not None and STATE.state_pool_name in group.entry_map:
                pool = STATE.host_state_pool = group.get_pool(STATE.state_pool_name)
        if pool is not None:
            total = pool.size
            self.host_states.labels('total').set(total)
            self.host_states.labels('used').set(total - pool.available_size())


METRICS = [None]


class TTLStrategy:
    """Evict beyond-retention data first, then expired leases, then the native order."""

    def __init__(self, base):
        self.base = base

    def get_priority(self, node):
        return lease_rank(node), self.base.get_priority(node)


def lease_rank(node):
    """0 beyond retention, 1 expired but retained, 2 valid lease."""
    tag, now = getattr(node, '_ttl', None), STATE.now
    if tag is None:
        # First seen by an eviction pass shortly after creation: start a default lease clock.
        tag = node._ttl = (now, now + DEFAULT_TTL, DEFAULT_TTL)
    return 0 if now >= tag[0] + STATE.table.keep else 2 if now < tag[1] else 1


LEASE_NAMES = ('beyond', 'expired', 'valid')
STATE_ORDER = os.environ.get('QWEN_TTL_STATE_ORDER', '1') != '0'   # diagnosis: 0 restores the native state-pool LRU
STATE_WINDOW = 4096    # state-pool LRU entries examined ahead of one eviction pass (the whole pool in practice)


def reorder_tail(lru, window=STATE_WINDOW):
    """Make the LRU tail of a state pool evict by lease rank first, LRU order within a rank.

    The state pools (Mamba) evict straight off an LRU list instead of asking the eviction
    strategy. Rewriting the order of the tail window gives them the same priority as KV
    without touching the native walk. Returns the window in eviction order.
    """
    if lru.mid is not None or not STATE_ORDER:   # session-partitioned lists keep their native order
        return []
    pt, node, nodes = lru._pt, lru.tail.lru_prev[lru._pt], []
    while node is not lru.head and len(nodes) < window:
        nodes.append(node)
        node = node.lru_prev[pt]
    ranks = [lease_rank(n) for n in nodes]
    if all(a <= b for a, b in zip(ranks, ranks[1:])):
        return nodes
    anchor = node                  # first entry beyond the window, towards the MRU end
    order = [n for _, _, n in sorted(zip(ranks, range(len(nodes)), nodes), key=lambda item: item[:2])]
    for n in nodes:
        lru._remove_node(n)
    for n in reversed(order):      # the first to evict ends up next to the tail
        lru._add_node_after(anchor, n)
        anchor = n
    return order


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


TAIL_WINDOW = 65536     # tokens of upcoming L2 eviction that one tail scan covers


def backup_tail(core, num_tokens):
    """Persist the valid 1h leaves at the L2 eviction tail before they are evicted.

    A scan sorts every host leaf, so it runs once per TAIL_WINDOW / 2 tokens of eviction
    demand (a rank-consistent count), not on every eviction call. Writes pin their node.
    """
    STATE.tail_demand += num_tokens
    if STATE.tail_demand < TAIL_WINDOW // 2:
        return
    STATE.tail_demand = 0
    started = time.perf_counter()
    now, covered = STATE.now, 0
    for node in heapq.nsmallest(256, core.evictable_host_leaves, key=core.eviction_strategy.get_priority):
        if covered >= TAIL_WINDOW:
            break
        covered += write_chain(node, core.root_node)
    spend('tail_scan', started)


def writable(node):
    """A host node worth persisting: valid 1h lease with time left, and a host copy to read from."""
    tag = getattr(node, '_ttl', None)
    return tag is not None and tag[2] == 3600 and tag[1] - STATE.now > 300 and node.backuped


def eviction_chain(leaf, root):
    """The leaf plus the ancestors eviction reaches right after it, shallowest first.

    Host eviction cascades: once a leaf is gone its single-child parent is the next leaf, inside
    the same pass, so a scan of the current leaves alone never sees the prompt nodes above them.
    """
    chain, node = [], leaf
    while node is not None and node is not root:
        chain.append(node)
        parent = node.parent
        if parent is None or len(parent.children) != 1:
            break
        node = parent
    chain.reverse()
    return chain


def write_chain(leaf, root):
    """Persist a leaf's eviction chain as one unit, or not at all.

    L3 is read as a hash chain from the L2 match point and (with a state layer) can only
    resume on a page stored with its state. A partly written chain is unreadable, so decide
    first: the longest writable run from the top that ends on a node with host state, and
    only if the budget covers all of it.
    """
    chain = eviction_chain(leaf, root)
    run = []
    for node in chain:
        if not getattr(node, '_ttl_l3', False) and not writable(node):
            break
        run.append(node)
    if not run and chain:
        # Why nothing of this chain can go to L3 (tokens of the whole chain).
        top, tag = chain[0], getattr(chain[0], '_ttl', None)
        reason = ('l3_skipped_5m' if tag is None or tag[2] != 3600 else
                  'l3_skipped_expiring' if tag[1] - STATE.now <= 300 else 'l3_skipped_no_host_copy')
        count(reason, sum(len(node.key) for node in chain))
    run = run[:restorable(run)]
    pending = [node for node in run if not getattr(node, '_ttl_l3', False)]
    tokens = sum(len(node.key) for node in pending)
    if DEBUG and METRICS[0] is not None:
        LOG.warning('qwen_ttl l3 chain leaf=%s chain=%s run=%s pending=%s tokens=%d budget=%d', leaf.id,
                    [len(n.key) for n in chain], len(run), len(pending), tokens, STATE.l3_budget)
    if pending:
        if l3_take(tokens):
            STATE.l3_writing = True
            try:
                for node in pending:
                    STATE.cache.write_backup_storage(node.id)
                    node._ttl_l3 = True
                    remember_storage(node)
            finally:
                STATE.l3_writing = False
            count('l3_written', tokens)
        else:
            count('l3_dropped', tokens)
    return sum(len(node.key) for node in chain)


def restorable(chain):
    """How many leading chain nodes are worth writing.

    A model with a state layer can only resume from L3 at a page whose state was stored with
    it: the engine discards a fetched run that ends on a page without one. So the written
    chain must end on a node that still has its host state; anything below that is useless.
    """
    ct = STATE.state_component
    if ct is None:
        return len(chain)
    for index in range(len(chain) - 1, -1, -1):
        if chain[index].component_data[ct].host_value is not None:
            return index + 1
    if chain:
        count('l3_skipped_stateless', sum(len(node.key) for node in chain))
    return 0


def spend(part, started):
    if METRICS[0] is not None:
        METRICS[0].seconds.labels(part).inc(time.perf_counter() - started)


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


DEBUG = os.environ.get('QWEN_TTL_DEBUG') == '1'
L3_PAGE_LIMIT = 4_000_000     # remembered L3 pages; the oldest are forgotten first


def remember_storage(node):
    """Record exactly which engine pages a node's L3 write puts on disk, and where a state is."""
    pages = STATE.l3_pages
    for page_hash in node.hash_value:
        pages[page_hash] = node.id
    ct = STATE.state_component
    if ct is None or node.component_data[ct].host_value is not None:
        STATE.l3_state_pages.add(node.hash_value[-1])
    while len(pages) > L3_PAGE_LIMIT:
        STATE.l3_state_pages.discard(next(iter(pages)))
        del pages[next(iter(pages))]


def storage_end(cache, req, matched, limit):
    """Absolute token end of what L3 can restore right after the L1+L2 match, or None.

    The engine fetches a hash chain of pages and, with a state layer, keeps it only if the
    last page was stored with its state. So: the run of consecutive pages known to be on
    disk (and fully written), cut back to the deepest page that has a state.
    """
    pages = STATE.l3_pages
    if not pages:
        return None
    engine_page = cache.page_size
    tokens = req.full_untruncated_fill_ids
    end = min(limit, req._compute_max_prefix_len(len(tokens))) // engine_page * engine_page
    if matched % engine_page or end - matched < engine_page:
        return None
    prior = cache.get_last_hash_value(req.last_host_node)
    key = STATE.key_class(tokens[matched:matched + engine_page], req.extra_key, cache_salt=req.cache_salt)
    first = STATE.page_hashes(key, prior, page_size=engine_page)
    if not first or first[0] not in pages:
        return None                     # the common case: one page hashed, nothing on disk
    key = STATE.key_class(tokens[matched:end], req.extra_key, cache_salt=req.cache_salt)
    writing = {entry[0] for entry in cache.ongoing_backup.values()}
    # The fetched run is staged in L2 first. If L2 cannot hold all of it the engine keeps a
    # shorter prefix, which ends off a state page and is thrown away. Make room up front and
    # never ask for more than L2 can take.
    host_pool = cache.cache_controller.mem_pool_host
    room = host_pool.available_size()
    if room < end - matched:
        cache.evict_host(end - matched - room)
        room = host_pool.available_size()
    best = last = None
    index, why = 0, 'end'
    for index, page_hash in enumerate(STATE.page_hashes(key, prior, page_size=engine_page)):
        node_id = pages.get(page_hash)
        if node_id is None or node_id in writing or (index + 1) * engine_page > room:
            why = 'missing' if node_id is None else 'writing' if node_id in writing else 'room'
            break
        if page_hash in STATE.l3_state_pages:
            best = matched + (index + 1) * engine_page
            last = page_hash
    if DEBUG and METRICS[0] is not None:
        LOG.warning('qwen_ttl l3 lookup rid=%s matched=%d end=%s want=%d run_pages=%d stop=%s room=%d state_page=%s', req.rid,
                    matched, best, end, index, why, room, last[:12] if best else None)
    return best


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


def install_classes(cache_class, req_class, batch_class, scheduler_class, streamer_class, core_class, base_component,
                    state_class=None, state_component=None, state_pool_name=None, full_class=None):
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
            if state_component in self.tree_core.components_by_type:
                STATE.state_component = state_component
                STATE.state_pool_name = state_pool_name
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
        started = time.perf_counter()
        limit = min(key._raw_len(), reuse_limit(req))
        spend('match', started)
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
        started = time.perf_counter()
        req.init_next_round_input(self.tree_cache, cow_mamba=False)
        matched = len(req.prefix_indices) + req.host_hit_length
        hit_end = storage_end(self.tree_cache, req, matched, req._ttl_limit)
        spend('storage_gate', started)
        if hit_end is None:
            req.storage_prefetch_last_match_len = matched
            return None
        count('l3_lookup', 1)
        req._ttl_prefetching = True
        try:
            return original_prefetch(self, req, hit_end)
        finally:
            req._ttl_prefetching = False

    @wraps(original_extend)
    def extend(self):
        first = [req for req in self.reqs if hasattr(req, '_ttl_seconds') and not hasattr(req, '_ttl_bill')
                 and not req._cache_breakdown_computed and not req.retracted_stain]
        result = original_extend(self)
        if first:
            started = time.perf_counter()
            for req in first:
                admit(req)
            spend('admit', started)
        STATE.table.sweep(STATE.now)
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
        storage = STATE.l3_rate and self.enable_storage and STATE.now - STATE.last_1h < STATE.table.keep
        if component_type == base_component:
            if storage:
                backup_tail(self, num_tokens)
        elif component_type == STATE.state_component:
            # State-pool pressure: same lease priority as KV, and the same L3 rescue for 1h leaves.
            started = time.perf_counter()
            nodes = reorder_tail(self.host_lru_lists[component_type])
            if storage:
                reach = 0
                for node in nodes:
                    if reach >= 2 * num_tokens:
                        break
                    reach += 1
                    # A node about to lose its host state is a restore point: persist the path
                    # ending on it, whether or not it is a leaf (prompt checkpoints are internal).
                    write_chain(node, self.root_node)
            spend('state_tail', started)
        return original_drive(self, component_type, num_tokens)

    def counted(component_class, pool, per_token):
        """Count what each pool gives up, by tier and by the lease state of the evicted node."""
        original = component_class.evict_component

        @wraps(original)
        def evict_component(self, node, *args, **kwargs):
            metrics = METRICS[0]
            if metrics is None:
                return original(self, node, *args, **kwargs)
            data = node.component_data[self.component_type]
            had_device, had_host = data.value is not None, data.host_value is not None
            result = original(self, node, *args, **kwargs)
            amount, lease = (len(node.key) if per_token else 1), LEASE_NAMES[lease_rank(node)]
            if had_device and data.value is None:
                metrics.evictions.labels(pool, 'device', lease).inc(amount)
            if had_host and data.host_value is None:
                metrics.evictions.labels(pool, 'host', lease).inc(amount)
            return result

        component_class.evict_component = evict_component

    if full_class is not None:
        counted(full_class, 'kv', True)
    if state_class is not None:
        counted(state_class, 'mamba', False)

    if state_class is not None:
        original_state_start = state_class._evict_device_start

        @wraps(original_state_start)
        def state_start(self, request_cnt):
            started = time.perf_counter()
            reorder_tail(self.tree_core.lru_lists[self.component_type])
            spend('state_order', started)
            return original_state_start(self, request_cnt)

        state_class._evict_device_start = state_start

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
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
    from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.utils import get_storage_hash_str
    STATE.key_class, STATE.page_hashes = RadixKey, get_storage_hash_str
    install_classes(UnifiedRadixCache, Req, ScheduleBatch, Scheduler, streamer, UnifiedTreeCore, BASE_COMPONENT_TYPE,
                    MambaComponent, ComponentType.MAMBA, PoolName.MAMBA, FullComponent)
    LOG.info('qwen_ttl hooks installed (keep=%ss)', STATE.table.keep)
