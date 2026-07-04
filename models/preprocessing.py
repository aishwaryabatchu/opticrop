import numpy as np

def validate_inputs(n, p, k, temperature, humidity, ph, rainfall):
    """
    Validates that the environmental and soil inputs fall within realistic bounds.
    Returns (is_valid, error_message).
    """
    try:
        n = float(n)
        p = float(p)
        k = float(k)
        temperature = float(temperature)
        humidity = float(humidity)
        ph = float(ph)
        rainfall = float(rainfall)
    except ValueError:
        return False, "All input parameters must be numerical values."

    if not (0 <= n <= 200):
        return False, "Nitrogen (N) must be between 0 and 200 kg/ha."
    if not (0 <= p <= 200):
        return False, "Phosphorus (P) must be between 0 and 200 kg/ha."
    if not (0 <= k <= 300):
        return False, "Potassium (K) must be between 0 and 300 kg/ha."
    if not (-10 <= temperature <= 60):
        return False, "Temperature must be between -10°C and 60°C."
    if not (0 <= humidity <= 100):
        return False, "Humidity must be between 0% and 100%."
    if not (0 <= ph <= 14):
        return False, "pH must be between 0 and 14."
    if not (0 <= rainfall <= 1000):
        return False, "Rainfall must be between 0 and 1000 mm."

    return True, None
