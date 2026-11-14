"""Pure rendering of the cache profile (TTL / file L3); no filesystem or service changes."""
import hashlib
import json
import os


def host_memory_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemTotal:'):
            return int(line.split()[1]) / 1024 ** 2
    raise ValueError('MemTotal not found')


def l2_size_gb(l2, tp_size, memory_gb):
    """Per-rank host cache size: as large as host memory allows after the fixed reserve."""
    if os.environ.get('QWEN_L2_SIZE_GB'):   # stress tests shrink L2 to reach eviction quickly
        return int(os.environ['QWEN_L2_SIZE_GB'])
    if l2.get('hicache_size_gb'):
        return int(l2['hicache_size_gb'])
    size = int((memory_gb - l2['reserve_gb']) // tp_size)
    if size < 1:
        raise ValueError('Not enough host memory for an L2 cache; disable l2 in config/cache.json')
    return size


def apply_profile(cfg, server_args, server_env, profile, memory_gb=None, disk_bytes=0):
    """Return (args, env, record) with the enabled cache tiers rendered into the server arguments."""
    ttl, l3 = profile['ttl'], profile['l3']
    l2 = profile['l2']
    record = {'ttl_enabled': bool(ttl.get('enabled')), 'l2_enabled': bool(l2.get('enabled')), 'l3_enabled': bool(l3.get('enabled')),
              'gpu_prefix_cache': 'native RadixAttention, unchanged'}
    server_args = dict(server_args)
    if l2.get('enabled'):
        memory_gb = host_memory_gb() if memory_gb is None else memory_gb
        size = l2_size_gb(l2, int(server_args['tp-size']), memory_gb)
        server_args.update(l2['args'])
        server_args['hicache-size'] = size
        record.update(l2_hicache_size_gb=size, l2_host_memory_gb=round(memory_gb, 1),
                      l2_total_gb=size * int(server_args['tp-size']) )
    elif l3.get('enabled'):
        raise ValueError('L3 requires the L2 host cache; enable l2 in config/cache.json')
    server_env = dict(server_env)
    if ttl.get('enabled'):
        server_env.update(PYTHONPATH='/compat', QWEN_TTL_KEEP_SECONDS=str(ttl['keep_seconds']))
        if ttl.get('state_pool_lease_order'):
            server_env['QWEN_TTL_STATE_ORDER'] = '1'
        # Acceptance / benchmark-only switches, taken from the operator's environment.
        for name in ('QWEN_TTL_TEST_CLOCK', 'QWEN_TTL_DEFAULT_SECONDS', 'QWEN_TTL_L3_WRITES', 'QWEN_TTL_STATE_ORDER', 'QWEN_TTL_DEBUG'):
            if os.environ.get(name):
                server_env[name] = os.environ[name]
                record[name.lower()] = os.environ[name]
        record['ttl_keep_seconds'] = ttl['keep_seconds']
    if not l3.get('enabled'):
        return server_args, server_env, record
    if not ttl.get('enabled'):
        raise ValueError('L3 is TTL-gated; enable ttl in config/cache.json')
    # Native raw files omit dtype/layout in their keys, so namespace the storage
    # directory by image, format and the effective arguments.
    identity = {'format': l3['cache_format'], 'image': cfg['image_id'],
                'server_env': dict(server_env), 'args': {**server_args, **l3['args']}}
    for key in ('host', 'port'):
        identity['args'].pop(key, None)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]
    base = '/cache/hicache-l3/' + digest
    # Endurance budget -> write rate and per-rank quota (design 4.8).
    daily = l3['endurance_dwpd'] * disk_bytes * l3['disk_share']
    rate = daily / 86400 / l3['kv_bytes_per_token']
    quota = int(daily * ttl['keep_seconds'] / 86400 / int(server_args['tp-size']))
    storage = {**l3['storage'], 'max_size': f'{max(quota // 2 ** 20, 1)}Mi', 'file_storage_path': base + '/blocks'}
    args = {**server_args, **l3['args'],
            'hicache-storage-backend-extra-config': json.dumps(storage, sort_keys=True, separators=(',', ':'))}
    env = {**server_env, 'SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR': base + '/blocks',
           'QWEN_TTL_L3_TOKENS_PER_SECOND': f'{rate:.3f}',
           'QWEN_TTL_L3_BURST_TOKENS': f'{rate * l3["burst_seconds"]:.0f}'}
    record.update(namespace=digest, container_storage_dir=base + '/blocks',
                  per_rank_max_size=storage['max_size'], l3_tokens_per_second=round(rate, 1),
                  l3_daily_budget_gb=round(daily / 1e9),
                  min_free_space=l3['storage']['min_free_space'])
    return args, env, record
