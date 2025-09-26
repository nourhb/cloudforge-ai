from typing import List
import numpy as np
from sklearn.linear_model import LinearRegression

# Simple linear regression-based forecast for CPU series
# Input series: list of numbers (0-100)
# Output: list of predicted values length=steps

def forecast_cpu(series: List[float], steps: int = 12) -> List[float]:
    if not series:
        return [50.0] * steps
    y = np.array(series, dtype=float)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    future_x = np.arange(len(y), len(y) + steps).reshape(-1, 1)
    preds = model.predict(future_x)
    # clamp to 0..100
    preds = np.clip(preds, 0, 100)
    return preds.round(2).tolist()

# TEST: forecast_cpu returns list length steps
