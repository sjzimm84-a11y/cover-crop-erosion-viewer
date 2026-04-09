from typing import Dict

DEFAULT_THRESHOLDS = {
    "ndvi_low": 0.35,
    "slope_steep": 6.0,
}


def score_erosion_concern(
    ndvi_mean: float,
    slope_mean: float,
    ndvi_threshold: float = DEFAULT_THRESHOLDS["ndvi_low"],
    slope_threshold: float = DEFAULT_THRESHOLDS["slope_steep"],
) -> Dict[str, object]:
    low_cover = ndvi_mean < ndvi_threshold
    steep = slope_mean > slope_threshold
    if low_cover and steep:
        concern_level = "High"
        score = 3
    elif low_cover or steep:
        concern_level = "Moderate"
        score = 2
    else:
        concern_level = "Low"
        score = 1

    return {
        "concern_level": concern_level,
        "score": score,
        "low_cover": low_cover,
        "steep_slope": steep,
        "ndvi_threshold": ndvi_threshold,
        "slope_threshold": slope_threshold,
    }
