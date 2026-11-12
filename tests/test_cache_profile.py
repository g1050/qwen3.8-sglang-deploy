import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cache_profile import apply_profile  # noqa: E402

CFG = json.loads((ROOT / 'config/deployment.json').read_text())
PROFILE = json.loads((ROOT / 'config/cache.json').read_text())


class CacheProfile(unittest.TestCase):
    def test_default_disables_ttl_and_l3(self):
        args, env, record = apply_profile(CFG, CFG['args'], CFG['env'], PROFILE)
        self.assertEqual((args, env), (CFG['args'], CFG['env']))
        self.assertFalse(record['ttl_enabled'] or record['l3_enabled'])
        self.assertFalse(any(k.startswith('hicache') or k == 'enable-hierarchical-cache' for k in args))

    def test_ttl_is_refused(self):
        profile = copy.deepcopy(PROFILE)
        profile['ttl']['enabled'] = True
        with self.assertRaises(ValueError):
            apply_profile(CFG, CFG['args'], CFG['env'], profile)

    def test_l3_renders_namespaced_file_backend(self):
        profile = copy.deepcopy(PROFILE)
        profile['l3']['enabled'] = True
        args, env, record = apply_profile(CFG, CFG['args'], CFG['env'], profile)
        self.assertEqual(args['hicache-storage-backend'], 'file')
        extra = json.loads(args['hicache-storage-backend-extra-config'])
        self.assertEqual(extra['file_storage_path'], record['container_storage_dir'])
        self.assertEqual(env['SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR'], record['container_storage_dir'])
        other, _, changed = apply_profile(CFG, {**CFG['args'], 'tp-size': 4}, CFG['env'], profile)
        self.assertNotEqual(record['namespace'], changed['namespace'])


if __name__ == '__main__':
    unittest.main()
