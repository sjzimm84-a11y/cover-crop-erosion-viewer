# Technical debt

- 2026-09-18: Removed the deprecated bin-based C-factor path (`_lookup_c_factor` + `IOWA_C_FACTOR_TABLE`) from `src/scoring.py` and retired its only consumer, `compare_methods.py`; `_continuous_c_factor` is now the sole C-factor model.
- 2026-09-18: Removed the dead `pixel_level_concern()` (last of the hardcoded NDVI bin logic) from `src/scoring.py` and dropped its unused import from `app.py`; `pixel_risk_index()` is the sole per-pixel scoring path.
