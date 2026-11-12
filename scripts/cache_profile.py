"""Pure rendering of the cache profile (TTL / file L3); no filesystem or service changes."""
import hashlib
import json


def apply_profile(cfg, server_args, server_env, profile):
    """Return (args, env, record). With TTL and L3 disabled the inputs pass through unchanged."""
    ttl, l3 = profile['ttl'], profile['l3']
    if ttl.get('enabled'):
        raise ValueError('TTL is not available in v0.1.0: the Kimi TTL adapter has not been '
                         'ported to this SGLang build. Keep config/cache.json ttl.enabled=false.')
    record = {'ttl_enabled': False, 'l3_enabled': bool(l3.get('enabled')),
              'gpu_prefix_cache': 'native RadixAttention, unchanged'}
    if not l3.get('enabled'):
        return dict(server_args), dict(server_env), record
    # Native raw files omit dtype/layout in their keys, so namespace the storage
    # directory by image, format and the effective arguments.
    identity = {'format': l3['cache_format'], 'image': cfg['image_id'],
                'server_env': dict(server_env), 'args': {**server_args, **l3['args']}}
    for key in ('host', 'port'):
        identity['args'].pop(key, None)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]
    base = '/cache/hicache-l3/' + digest
    args = {**server_args, **l3['args'],
            'hicache-storage-backend-extra-config': json.dumps(
                {**l3['storage'], 'file_storage_path': base + '/blocks'},
                sort_keys=True, separators=(',', ':'))}
    env = {**server_env, 'SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR': base + '/blocks'}
    record.update(namespace=digest, container_storage_dir=base + '/blocks',
                  per_rank_max_size=l3['storage']['max_size'],
                  min_free_space=l3['storage']['min_free_space'])
    return args, env, record
