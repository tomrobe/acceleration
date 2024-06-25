# dynamic moving average
import numpy as np
def dynamic_moving_average(data, phase, window_factor=1):
    smoothed_data = np.zeros_like(data)
    for i in range(len(data)):
        if i < len(data) - 1:
            phase_diff = abs(phase[i + 1] - phase[i])
        else:
            phase_diff = abs(phase[i] - phase[i - 1])
        # Define window size based on phase (window_factor is a scaling parameter)
        window_size = int(window_factor / phase_diff)
        if window_size == 0:
            window_size = 1  # Avoid zero window size
        half_window = window_size // 2
        
        # Determine window boundaries
        start = max(i - half_window, 0)
        end = min(i + half_window, len(data))
        
        # Calculate average within window
        smoothed_data[i] = np.mean(data[start:end])
    return smoothed_data

# Savitzky-Golay
from scipy.signal import savgol_filter
def smooth_savgol(data, window_size=51, poly_order=0):
    return savgol_filter(data, window_size, poly_order)

#dynamic Savitzky-Golay
def dynamic_smooth_savgol(data, phase, window_factor=1, poly_order=0):
    for i in range(len(data)):
        if i < len(data) - 1:
            phase_diff = abs(phase[i + 1] - phase[i])
        else:
            phase_diff = abs(phase[i] - phase[i - 1])
        # Define window size based on phase (window_factor is a scaling parameter)
        window_size = int(window_factor / phase_diff)
        if window_size == 0:
            window_size = 1  # Avoid zero window size
    return savgol_filter(data, window_size, poly_order)


# exponential Savitzky-Golay
from scipy.optimize import curve_fit

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

def smooth_exp_savgol(data, window_size=51):
    smoothed_data = np.zeros_like(data)
    half_window = window_size // 2
    
    for i in range(len(data)):
        start = max(i - half_window, 0)
        end = min(i + half_window, len(data))
        
        x = np.arange(start, end)
        y = data[start:end]
        
        if len(x) < 3:  # Need at least 3 points to fit
            smoothed_data[i] = data[i]
        else:
            # Fit an exponential curve to the data in the window
            popt, _ = curve_fit(exp_func, x, y, p0=(1, 0.1, 1))
            smoothed_data[i] = exp_func(i, *popt)
    
    return smoothed_data
