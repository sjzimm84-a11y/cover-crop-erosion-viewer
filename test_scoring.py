"""
test_scoring.py
---------------
Test suite for the continuous exponential C-factor implementation in scoring.py.

Run:
    python test_scoring.py
    python -m pytest test_scoring.py -v

Coverage:
    - _continuous_c_factor: edge cases, negative NDVI clamping, unknown residue fallback
    - _continuous_c_array:  NaN propagation, shape preservation, negative clamping
    - pixel_risk_index:     NaN propagation, monotonicity, shape, value bounds
    - _analytical_ls_factor: scalar and array, minimum floor, monotonicity
    - score_erosion_concern: return keys, c_factor <= baseline, rusle_score consistency
    - estimate_soil_loss:   missing K, status codes, value sanity
"""

import sys
import math
import unittest
import numpy as np

sys.path.insert(0, ".")

from src.scoring import (
    _continuous_c_factor,
    _continuous_c_array,
    _analytical_ls_factor,
    pixel_risk_index,
    score_erosion_concern,
    estimate_soil_loss,
    CONTINUOUS_C_PARAMS,
    RESIDUE_OPTIONS,
    _CONTINUOUS_C_FALLBACK,
)

# Residue systems that have explicit params (all options including Unknown)
ALL_RESIDUE = RESIDUE_OPTIONS
NO_TILL_CORN = "No-till corn (high residue ~80% cover)"
CONVENTIONAL  = "Tillage — < 30% residue (conventional tillage)"
UNKNOWN        = "Unknown — not recorded (conservative default)"


class TestContinuousCFactor(unittest.TestCase):

    def test_ndvi_zero_returns_intercept(self):
        """C(NDVI=0) must equal the intercept for every residue system."""
        for residue in ALL_RESIDUE:
            p = CONTINUOUS_C_PARAMS.get(residue, CONTINUOUS_C_PARAMS[_CONTINUOUS_C_FALLBACK])
            expected = p["intercept"]
            result   = _continuous_c_factor(0.0, residue)
            self.assertAlmostEqual(result, expected, places=6,
                msg=f"{residue}: C(0) expected {expected}, got {result}")

    def test_high_ndvi_approaches_floor(self):
        """C(NDVI=5) should be within 0.001 of the floor (exp term ~= 0)."""
        for residue in ALL_RESIDUE:
            p     = CONTINUOUS_C_PARAMS.get(residue, CONTINUOUS_C_PARAMS[_CONTINUOUS_C_FALLBACK])
            result = _continuous_c_factor(5.0, residue)
            self.assertAlmostEqual(result, p["floor"], delta=0.001,
                msg=f"{residue}: C(5.0) expected near floor {p['floor']}, got {result}")

    def test_negative_ndvi_clamped_to_zero(self):
        """Negative NDVI must return the same value as NDVI=0."""
        for residue in ALL_RESIDUE:
            c_zero = _continuous_c_factor(0.0, residue)
            c_neg  = _continuous_c_factor(-0.50, residue)
            self.assertAlmostEqual(c_neg, c_zero, places=8,
                msg=f"{residue}: C(-0.5) != C(0)")

    def test_result_bounded_by_floor_and_intercept(self):
        """C must stay within [floor, intercept] for all NDVI >= 0."""
        test_ndvi = [0.0, 0.10, 0.25, 0.40, 0.60, 0.80, 1.00]
        for residue in ALL_RESIDUE:
            p = CONTINUOUS_C_PARAMS.get(residue, CONTINUOUS_C_PARAMS[_CONTINUOUS_C_FALLBACK])
            for ndvi in test_ndvi:
                c = _continuous_c_factor(ndvi, residue)
                self.assertGreaterEqual(c, p["floor"] - 1e-9,
                    msg=f"{residue} NDVI={ndvi}: C={c} below floor={p['floor']}")
                self.assertLessEqual(c, p["intercept"] + 1e-9,
                    msg=f"{residue} NDVI={ndvi}: C={c} above intercept={p['intercept']}")

    def test_monotone_decreasing_with_ndvi(self):
        """C must be non-increasing across the full NDVI range.
        Below UNIVERSAL_NDVI_BASELINE the curve is flat (C = intercept),
        so equality is allowed there; above the threshold C must strictly decrease.
        """
        from src.scoring import UNIVERSAL_NDVI_BASELINE
        ndvi_seq = [0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80]
        for residue in ALL_RESIDUE:
            values = [_continuous_c_factor(n, residue) for n in ndvi_seq]
            for i in range(len(values) - 1):
                # Non-increasing everywhere
                self.assertGreaterEqual(values[i], values[i + 1],
                    msg=f"{residue}: not non-increasing at NDVI {ndvi_seq[i]} -> "
                        f"{ndvi_seq[i+1]} ({values[i]:.4f} -> {values[i+1]:.4f})")
                # Strictly decreasing once both points are above the baseline
                if ndvi_seq[i] > UNIVERSAL_NDVI_BASELINE:
                    self.assertGreater(values[i], values[i + 1],
                        msg=f"{residue}: not strictly decreasing above baseline at "
                            f"NDVI {ndvi_seq[i]} -> {ndvi_seq[i+1]}")

    def test_unknown_residue_falls_back_to_conventional(self):
        """Unknown residue system must produce same result as _CONTINUOUS_C_FALLBACK."""
        for ndvi in [0.0, 0.25, 0.50]:
            c_unknown      = _continuous_c_factor(ndvi, UNKNOWN)
            c_conventional = _continuous_c_factor(ndvi, _CONTINUOUS_C_FALLBACK)
            self.assertAlmostEqual(c_unknown, c_conventional, places=8,
                msg=f"Unknown fallback mismatch at NDVI={ndvi}")

    def test_completely_unknown_key_falls_back(self):
        """Unrecognized residue key must fall back without raising."""
        result = _continuous_c_factor(0.30, "Fictitious system")
        fallback = _continuous_c_factor(0.30, _CONTINUOUS_C_FALLBACK)
        self.assertAlmostEqual(result, fallback, places=8)


class TestContinuousCArray(unittest.TestCase):

    def test_nan_propagates(self):
        """NaN in input must produce NaN in output."""
        arr = np.array([0.10, np.nan, 0.40, np.nan])
        result = _continuous_c_array(arr, NO_TILL_CORN)
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[3]))
        self.assertFalse(np.isnan(result[0]))
        self.assertFalse(np.isnan(result[2]))

    def test_shape_preserved(self):
        """Output shape must match input shape."""
        arr = np.random.rand(5, 7)
        result = _continuous_c_array(arr, CONVENTIONAL)
        self.assertEqual(result.shape, arr.shape)

    def test_negative_clamped_not_nan(self):
        """Negative NDVI must clamp to 0 (not NaN) in the array version."""
        arr    = np.array([-0.20, -0.10, 0.0])
        result = _continuous_c_array(arr, NO_TILL_CORN)
        self.assertFalse(np.any(np.isnan(result)))
        # All negatives should equal C(0)
        c_zero = _continuous_c_factor(0.0, NO_TILL_CORN)
        for val in result:
            self.assertAlmostEqual(val, c_zero, places=8)

    def test_matches_scalar_function(self):
        """Array result must match scalar function element-wise."""
        ndvi_vals = np.array([0.0, 0.15, 0.30, 0.50, 0.70])
        for residue in ALL_RESIDUE:
            arr_result = _continuous_c_array(ndvi_vals, residue)
            for i, ndvi in enumerate(ndvi_vals):
                scalar = _continuous_c_factor(float(ndvi), residue)
                self.assertAlmostEqual(arr_result[i], scalar, places=8,
                    msg=f"{residue} NDVI={ndvi}: array vs scalar mismatch")


class TestAnalyticalLsFactor(unittest.TestCase):

    def test_minimum_floor(self):
        """LS must be >= 0.03 for all non-negative slopes."""
        for slope in [0, 0.5, 1, 2, 5, 9, 15, 30]:
            ls = float(_analytical_ls_factor(float(slope)))
            self.assertGreaterEqual(ls, 0.03,
                msg=f"LS below floor at slope={slope}%")

    def test_monotone_increasing(self):
        """LS must increase with slope."""
        slopes = [1, 3, 5, 9, 15, 25]
        ls_vals = [float(_analytical_ls_factor(float(s))) for s in slopes]
        for i in range(len(ls_vals) - 1):
            self.assertLess(ls_vals[i], ls_vals[i + 1],
                msg=f"LS not increasing at slope {slopes[i]} -> {slopes[i+1]}")

    def test_scalar_and_array_agree(self):
        """Scalar and array inputs must give the same result."""
        for slope in [2.0, 5.0, 9.0, 15.0]:
            scalar = float(_analytical_ls_factor(slope))
            arr    = _analytical_ls_factor(np.array([slope]))
            self.assertAlmostEqual(scalar, float(arr[0]), places=8)

    def test_shape_preserved(self):
        """Array input shape must be preserved."""
        inp = np.array([[2.0, 5.0], [9.0, 15.0]])
        out = _analytical_ls_factor(inp)
        self.assertEqual(out.shape, inp.shape)

    def test_known_value_slope5(self):
        """Spot-check LS at 5% slope against hand-calculated McCool value."""
        # theta = arctan(5/100) ≈ 0.04996
        # S = 10.8*sin(theta)+0.03 (slope < 9%)
        # m = 0.5 (slope >= 5 branch: np.where(slope_pct < 5, 0.4, 0.5))
        # L = (100/22.13)^0.5
        theta = math.atan(5 / 100.0)
        S = 10.8 * math.sin(theta) + 0.03
        m = 0.5
        L = (100.0 / 22.13) ** m
        expected = L * S
        result = float(_analytical_ls_factor(5.0))
        self.assertAlmostEqual(result, expected, places=5)


class TestPixelRiskIndex(unittest.TestCase):

    def _make_arrays(self):
        ndvi  = np.array([[0.10, 0.30, 0.50],
                          [0.20, 0.40, 0.60]])
        slope = np.array([[2.0,  5.0,  9.0],
                          [3.0,  7.0, 12.0]])
        return ndvi, slope

    def test_shape_preserved(self):
        ndvi, slope = self._make_arrays()
        result = pixel_risk_index(ndvi, slope, NO_TILL_CORN)
        self.assertEqual(result.shape, ndvi.shape)

    def test_all_positive_when_no_nan(self):
        ndvi, slope = self._make_arrays()
        result = pixel_risk_index(ndvi, slope, NO_TILL_CORN)
        self.assertTrue(np.all(result > 0))

    def test_nan_in_ndvi_propagates(self):
        ndvi  = np.array([[np.nan, 0.30], [0.20, 0.40]])
        slope = np.array([[5.0,    5.0],  [5.0,  5.0]])
        result = pixel_risk_index(ndvi, slope, NO_TILL_CORN)
        self.assertTrue(np.isnan(result[0, 0]))
        self.assertFalse(np.isnan(result[0, 1]))

    def test_nan_in_slope_propagates(self):
        ndvi  = np.array([[0.30, 0.30], [0.30, 0.30]])
        slope = np.array([[5.0,  np.nan], [5.0, 5.0]])
        result = pixel_risk_index(ndvi, slope, NO_TILL_CORN)
        self.assertTrue(np.isnan(result[0, 1]))
        self.assertFalse(np.isnan(result[0, 0]))

    def test_monotone_decreasing_with_ndvi(self):
        """Risk must be non-increasing with NDVI; strictly decreasing above baseline."""
        from src.scoring import UNIVERSAL_NDVI_BASELINE
        slope    = np.array([[5.0, 5.0, 5.0, 5.0, 5.0]])
        ndvi_pts = [0.10, 0.25, 0.40, 0.55, 0.70]
        ndvi     = np.array([ndvi_pts])
        for residue in ALL_RESIDUE:
            result = pixel_risk_index(ndvi, slope, residue)
            vals = result[0]
            for i in range(len(vals) - 1):
                self.assertGreaterEqual(vals[i], vals[i + 1],
                    msg=f"{residue}: risk not non-increasing from NDVI "
                        f"{ndvi_pts[i]} -> {ndvi_pts[i+1]}")
                if ndvi_pts[i] > UNIVERSAL_NDVI_BASELINE:
                    self.assertGreater(vals[i], vals[i + 1],
                        msg=f"{residue}: risk not strictly decreasing above baseline at "
                            f"NDVI {ndvi_pts[i]} -> {ndvi_pts[i+1]}")

    def test_monotone_increasing_with_slope(self):
        """Risk must increase as slope increases (same NDVI)."""
        ndvi  = np.array([[0.30, 0.30, 0.30, 0.30]])
        slope = np.array([[2.0,  5.0,  9.0, 15.0]])
        for residue in ALL_RESIDUE:
            result = pixel_risk_index(ndvi, slope, residue)
            vals = result[0]
            for i in range(len(vals) - 1):
                self.assertLess(vals[i], vals[i + 1],
                    msg=f"{residue}: risk not increasing at slope "
                        f"{slope[0,i]} -> {slope[0,i+1]}")

    def test_default_residue_is_conservative(self):
        """Default (unknown) residue must produce >= risk than no-till corn."""
        ndvi  = np.array([[0.30]])
        slope = np.array([[5.0]])
        risk_default  = pixel_risk_index(ndvi, slope)  # default = unknown
        risk_no_till  = pixel_risk_index(ndvi, slope, NO_TILL_CORN)
        self.assertGreaterEqual(float(risk_default[0, 0]), float(risk_no_till[0, 0]))

    def test_values_positive_and_finite(self):
        """All non-NaN outputs must be positive and finite."""
        ndvi  = np.array([0.0, 0.20, 0.40, 0.60, 0.80])
        slope = np.array([2.0, 5.0,  9.0,  15.0, 30.0])
        for residue in ALL_RESIDUE:
            result = pixel_risk_index(ndvi, slope, residue)
            self.assertTrue(np.all(np.isfinite(result)),
                msg=f"{residue}: non-finite values in output")
            self.assertTrue(np.all(result > 0),
                msg=f"{residue}: non-positive values in output")


class TestEstimateSoilLoss(unittest.TestCase):

    def test_missing_k_returns_unavailable(self):
        result = estimate_soil_loss(c_factor=0.20, ls_factor=1.21, k_factor=None)
        self.assertEqual(result["status_code"], "unavailable")
        self.assertIsNone(result["soil_loss_tons_ac_yr"])

    def test_non_numeric_k_returns_unavailable(self):
        result = estimate_soil_loss(c_factor=0.20, ls_factor=1.21, k_factor="unknown")
        self.assertEqual(result["status_code"], "unavailable")

    def test_within_t_status(self):
        # R=175, K=0.10, LS=1.0, C=0.20 → A = 3.5 t/ac (within T=5)
        result = estimate_soil_loss(
            c_factor=0.20, ls_factor=1.0, k_factor=0.10,
            t_value=5, r_factor=175.0
        )
        self.assertAlmostEqual(result["soil_loss_tons_ac_yr"], 3.5, places=1)
        self.assertEqual(result["status_code"], "within_t")

    def test_over_t_status(self):
        # R=175, K=0.37, LS=1.21, C=0.30 → A ≈ 23.5 t/ac (>> T=5)
        result = estimate_soil_loss(
            c_factor=0.30, ls_factor=1.21, k_factor=0.37,
            t_value=5, r_factor=175.0
        )
        self.assertGreater(result["soil_loss_tons_ac_yr"], 5.0)
        self.assertIn(result["status_code"], ("near_t", "over_t", "critical_t"))

    def test_ratio_to_t_consistency(self):
        result = estimate_soil_loss(
            c_factor=0.10, ls_factor=1.21, k_factor=0.37,
            t_value=5, r_factor=175.0
        )
        sl  = result["soil_loss_tons_ac_yr"]
        ratio = result["ratio_to_t"]
        self.assertAlmostEqual(ratio, sl / 5.0, places=1)

    def test_iowa_typical_conditions_range(self):
        """No-till corn at moderate NDVI on 5% slope should be within T."""
        c  = _continuous_c_factor(0.35, NO_TILL_CORN)   # ~0.008
        ls = float(_analytical_ls_factor(5.0))
        result = estimate_soil_loss(c_factor=c, ls_factor=ls, k_factor=0.37,
                                    t_value=5, r_factor=175.0)
        # Expect well under T for good no-till corn stand
        self.assertEqual(result["status_code"], "within_t",
            msg=f"Expected within_t for no-till corn NDVI=0.35; "
                f"got {result['soil_loss_tons_ac_yr']:.2f} t/ac")


class TestScoreErosionConcern(unittest.TestCase):

    def _run(self, ndvi_mean=0.35, slope_mean=5.0, residue=NO_TILL_CORN,
             k_factor=0.37):
        return score_erosion_concern(
            ndvi_stats={"mean": ndvi_mean, "std": 0.05, "min": 0.10, "max": 0.60,
                        "p10": 0.20, "p25": 0.28, "p75": 0.45, "p90": 0.55,
                        "count": 500, "valid_frac": 0.92},
            slope_stats={"mean": slope_mean, "std": 1.5, "min": 1.0, "max": 12.0,
                         "p10": 2.0, "p25": 3.5, "p75": 7.5, "p90": 9.0,
                         "count": 500, "valid_frac": 0.95},
            residue_system=residue,
            k_factor=k_factor,
        )

    def test_required_keys_present(self):
        result = self._run()
        for key in ("concern_level", "score", "c_factor", "c_factor_baseline",
                    "c_factor_method", "ls_factor", "rusle_score",
                    "residue_system", "soil_loss"):
            self.assertIn(key, result, msg=f"Missing key: {key}")

    def test_c_factor_method_label(self):
        result = self._run()
        self.assertEqual(result["c_factor_method"], "exponential_v2")

    def test_c_factor_lte_baseline(self):
        """c_factor (adjusted for NDVI) must always be <= c_factor_baseline (NDVI=0)."""
        for residue in ALL_RESIDUE:
            result = self._run(residue=residue)
            self.assertLessEqual(result["c_factor"], result["c_factor_baseline"],
                msg=f"{residue}: c_factor {result['c_factor']} > baseline "
                    f"{result['c_factor_baseline']}")

    def test_rusle_score_matches_c_times_ls(self):
        """rusle_score must equal c_factor × ls_factor (within rounding)."""
        result = self._run()
        expected = round(result["c_factor"] * result["ls_factor"], 3)
        self.assertAlmostEqual(result["rusle_score"], expected, places=2)

    def test_concern_level_matches_score_int(self):
        """score int and concern_level string must be consistent."""
        mapping = {"Low": 1, "Moderate": 2, "High": 3, "Critical": 4}
        for residue in ALL_RESIDUE:
            result = self._run(residue=residue)
            self.assertEqual(
                result["score"], mapping[result["concern_level"]],
                msg=f"{residue}: score {result['score']} != "
                    f"concern_level '{result['concern_level']}'"
            )

    def test_no_till_corn_lower_c_than_conventional(self):
        """No-till corn must always produce lower C-factor than conventional."""
        r_notill = self._run(residue=NO_TILL_CORN)
        r_conv   = self._run(residue=CONVENTIONAL)
        self.assertLess(r_notill["c_factor"], r_conv["c_factor"])

    def test_missing_k_factor_gives_unavailable_soil_loss(self):
        """When k_factor is None, soil_loss dict must report unavailable status."""
        result = self._run(k_factor=None)
        self.assertIsNotNone(result["soil_loss"])
        self.assertEqual(result["soil_loss"]["status_code"], "unavailable")
        self.assertIsNone(result["soil_loss"]["soil_loss_tons_ac_yr"])

    def test_high_ndvi_not_high_concern_for_notill(self):
        """NDVI=0.60 no-till corn on 5% slope should be Low concern."""
        result = self._run(ndvi_mean=0.60, slope_mean=5.0, residue=NO_TILL_CORN)
        self.assertIn(result["concern_level"], ("Low",),
            msg=f"Expected Low for high NDVI no-till corn, got {result['concern_level']}")

    def test_low_ndvi_conventional_steep_is_high_or_critical(self):
        """NDVI=0.05 conventional on 15% slope should be High or Critical."""
        result = self._run(ndvi_mean=0.05, slope_mean=15.0, residue=CONVENTIONAL)
        self.assertIn(result["concern_level"], ("High", "Critical"),
            msg=f"Expected High/Critical for bare conventional steep slope, "
                f"got {result['concern_level']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
