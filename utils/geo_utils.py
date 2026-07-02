import numpy as np

# =====================================================
# GEO ENCODING
# =====================================================

def encode_geo_risk(geo_risk):

    geo_risk = geo_risk.strip().lower()

    if geo_risk == "low":
        return 0.0

    elif geo_risk == "medium":
        return 1.0

    elif geo_risk == "high":
        return 2.0

    else:
        raise ValueError("Invalid Geo Risk")


# =====================================================
# PREPROCESS GEO
# =====================================================

def preprocess_geo(geo_risk):

    value = encode_geo_risk(geo_risk)

    return np.array(
        [[value]],
        dtype=np.float32
    )


# Backward compatibility
prepare_geo_input = preprocess_geo