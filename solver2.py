import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import smoothers

# Define the system of ODEs
def coupled_odes(x, y, pi_0, E, a, b, l):
    rho, rho_prime, theta, theta_prime = y
    rho_double_prime = (
        (theta_prime**2 - 2*pi_0*theta_prime + E**2) * rho
        + (a * np.cos((l+1)*theta) + b * np.sin((l+1)*theta)) * rho**l
    )
    theta_double_prime = (
        -(2 * rho_prime * (theta_prime - pi_0)
        - (a * np.sin((l+1)*theta) - b * np.cos((l+1)*theta)) * rho**l) / rho
    )
    return [rho_prime, rho_double_prime, theta_prime, theta_double_prime]

def solver():
    # Define parameters
    pi_0 = 1.0
    E = 1.0
    a = 1.0
    b = 0.5
    l = 5.0
    
    # Initial conditions
    rho_0 = 1.0
    rho_prime_0 = rho_0**3
    theta_0 = 0.0
    theta_prime_0 = 10.0
    y0= [rho_0, rho_prime_0, theta_0, theta_prime_0]
    
    # Solve the system over the interval x_span
    x_span = (0, 10)
    x_eval = np.linspace(*x_span, 1000)
    
    sol= solve_ivp(coupled_odes, x_span, y0, args=(pi_0, E, a, b, l), t_eval=x_eval)
    
    # Extract the updated solutions
    rho_sol= sol.y[0]
    theta_sol= sol.y[2]
    theta_prime_sol= sol.y[3]
    rho_prime_sol= sol.y[1]
    
    # Plot the updated results including theta'
    plt.figure(figsize=(21, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(sol.t, rho_sol, label=r'$\rho(x)$')
    plt.xlabel('x')
    plt.ylabel(r'$\rho$')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(sol.t, theta_sol, label=r'$\theta(x)$')
    plt.xlabel('x')
    plt.ylabel(r'$\theta$')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(sol.t, theta_prime_sol, label=r"$\theta'(x)$")
    plt.plot(sol.t, rho_prime_sol/rho_sol, label=r"$\rho' / \rho (x) $")
    plt.xlabel('x')
    plt.ylabel(r"$\theta'$")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    time = sol.t
    data = rho_sol
    phase = theta_sol

    # Apply smoothing methods
    window_factor = 10
    window_size = 51  # Choose an odd number for the window size
    poly_order = 0

    dynamic_ma = smoothers.dynamic_moving_average(data, phase, window_factor)
    smoothed_sg = smoothers.smooth_savgol(data, window_size, poly_order)
    dynamic_smoothed_sg = smoothers.dynamic_smooth_savgol(data, phase, window_factor, poly_order)
#    smoothed_exp_sg = smoothers.smooth_exp_savgol(rho_sol, window_size)
    
    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(time, data, label=r'Original $\rho(x)$', alpha=0.5)
    plt.plot(time, dynamic_ma, label='Dynamic Moving Average', linewidth=2)
    plt.plot(time, smoothed_sg, label='Savitzky-Golay', linewidth=2)
    plt.plot(time, dynamic_smoothed_sg, label='Dynamic Savitzky-Golay', linewidth=2)
#    plt.plot(time, smoothed_exp_sg, label='Exponential Savitzky-Golay', linewidth=2)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Smoothing Oscillations in Data with Dynamic and Exponential Smoothing')
    plt.show()
solver()
