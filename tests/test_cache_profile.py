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
    def test_defaults_enable_every_tier(self):
        _, _, record = apply_profile(CFG, CFG['args'], CFG['env'], PROFILE, memory_gb=503, disk_bytes=3.5e12)
        self.assertTrue(record['ttl_enabled'] and record['l2_enabled'] and record['l3_enabled'])

    def test_ttl_adds_adapter_environment(self):
        _, env, record = apply_profile(CFG, CFG['args'], CFG['env'], PROFILE, memory_gb=503)
        self.assertEqual((env['PYTHONPATH'], env['QWEN_TTL_KEEP_SECONDS']), ('/compat', '7200'))
        self.assertTrue(record['ttl_enabled'])

    def test_l3_is_sized_from_the_endurance_budget(self):
        args, env, record = apply_profile(CFG, CFG['args'], CFG['env'], PROFILE, memory_gb=503, disk_bytes=3.5e12)
        self.assertEqual(args['hicache-storage-backend'], 'file')
        extra = json.loads(args['hicache-storage-backend-extra-config'])
        self.assertEqual(extra['file_storage_path'], record['container_storage_dir'])
        self.assertEqual(record['l3_daily_budget_gb'], 1750)
        self.assertAlmostEqual(float(env['QWEN_TTL_L3_TOKENS_PER_SECOND']), 1.75e12 / 86400 / 140000, places=2)
        self.assertEqual(extra['max_size'], f'{int(1.75e12 * 7200 / 86400 / 8) // 2 ** 20}Mi')
        _, _, changed = apply_profile(CFG, {**CFG['args'], 'tp-size': 4}, CFG['env'], PROFILE, memory_gb=503, disk_bytes=3.5e12)
        self.assertNotEqual(record['namespace'], changed['namespace'])

    def test_l3_requires_ttl(self):
        profile = copy.deepcopy(PROFILE)
        profile['ttl']['enabled'] = False
        with self.assertRaises(ValueError):
            apply_profile(CFG, CFG['args'], CFG['env'], profile, memory_gb=503, disk_bytes=3.5e12)


if __name__ == '__main__':
    unittest.main()
