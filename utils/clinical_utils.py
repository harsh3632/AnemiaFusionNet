import numpy as np

# =====================================================
# GENDER ENCODING
# =====================================================

def encode_gender(gender):

    gender = gender.strip().lower()

    if gender == "male":
        return 1.0

    elif gender == "female":
        return 0.0

    else:
        raise ValueError("Gender must be Male or Female")


# =====================================================
# PREPROCESS CLINICAL
# =====================================================

def preprocess_clinical(
    hemoglobin,
    rdw,
    mcv,
    age,
    gender
):

    gender = encode_gender(gender)

    return np.array([
        [
            float(hemoglobin),
            float(rdw),
            float(mcv),
            float(age),
            gender
        ]
    ], dtype=np.float32)


# Backward compatibility
prepare_clinical_input = preprocess_clinical