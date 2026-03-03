import unittest

from topsim_simulator_v6 import (
    _fit_all_models,
    sozialplan_kosten_pro_person,
    umwelt_strafe,
)


class CoreMathTests(unittest.TestCase):
    def test_fit_all_models_detects_linear_relation(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 7, 9, 11, 13]
        typ, params, r2, predict = _fit_all_models(x, y)
        self.assertEqual(typ, "linear")
        self.assertAlmostEqual(params["a"], 2.0, places=6)
        self.assertAlmostEqual(params["b"], 3.0, places=6)
        self.assertGreaterEqual(r2, 0.999999)
        self.assertAlmostEqual(predict(6), 15.0, places=6)

    def test_fit_all_models_detects_constant_relation(self):
        x = [1, 2, 3]
        y = [10, 10, 10]
        typ, params, r2, predict = _fit_all_models(x, y)
        self.assertEqual(typ, "constant")
        self.assertAlmostEqual(params["c"], 10.0, places=6)
        self.assertEqual(r2, 1.0)
        self.assertAlmostEqual(predict(999), 10.0, places=6)

    def test_sozialplan_thresholds_follow_spec(self):
        self.assertEqual(sozialplan_kosten_pro_person(0, 100), 0)
        self.assertEqual(sozialplan_kosten_pro_person(5, 100), 0)
        self.assertEqual(sozialplan_kosten_pro_person(10, 100), 15)
        self.assertEqual(sozialplan_kosten_pro_person(35, 100), 20)
        self.assertEqual(sozialplan_kosten_pro_person(50, 100), 25)

    def test_umwelt_strafe_is_non_negative(self):
        self.assertGreaterEqual(umwelt_strafe(0), 0.0)
        self.assertGreaterEqual(umwelt_strafe(50), 0.0)
        self.assertGreaterEqual(umwelt_strafe(91.5), 0.0)
        self.assertGreaterEqual(umwelt_strafe(95), 0.0)
        self.assertEqual(umwelt_strafe(100), 0.0)


if __name__ == "__main__":
    unittest.main()
