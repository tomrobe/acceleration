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
    plt.figure(figsize=(10, 4))
    
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
    plt.savefig('original.pdf')
    plt.clf()

    time = sol.t
    data = rho_sol
    phase = theta_sol

    # Apply smoothing methods
    window_factor = 20
    window_size = 51  # Choose an odd number for the window size
    poly_order = 0

    smoothed_theta = smoothers.smooth_savgol(theta_sol, window_size, 1)

    plt.plot(time, theta_sol, label=r'$\theta(x)$')
    plt.plot(time, smoothed_theta, label=r'$\theta_s(x)$')
    plt.legend()
    plt.show()
    
    # Fit smoothed theta for angular frequency omega
    def func_linear(t,alpha,c):
        return alpha*t + c

    from scipy.optimize import curve_fit

    popt, pcov = curve_fit(func_linear, time[120:], smoothed_theta[120:])

    omega = popt[0]

    print(omega)

    plt.plot(time, theta_sol, label=r'$\theta(x)$')
    plt.plot(time, smoothed_theta, label=r'$\theta_s(x)$')
    plt.plot(time[120:], func_linear(time[120:],popt[0],popt[1]), label=r'$\theta_{sf}(x)$')
    plt.legend()
    plt.savefig('linearfittheta.pdf')
    plt.show()
    
    #Define window size WF in terms of omega
    WF = int(2*np.pi/omega/(time[1]-time[0]))

    print(WF)

    #Changed polynomial order to 6 on the smoother_sg
    dynamic_ma = smoothers.dynamic_moving_average(data, smoothed_theta, time, window_factor)
    smoothed_sg = smoothers.smooth_savgol(data, WF, 6)
    dynamic_smoothed_sg = smoothers.dynamic_smooth_savgol(data, phase, window_factor, poly_order)
#    smoothed_exp_sg = smoothers.smooth_exp_savgol(rho_sol, window_size)
    
    #Fit of exponential part
    popt, pcov = curve_fit(func_linear, time[198:], np.log(smoothed_sg[198:]))

    b,a = popt

    print('exponential parameter:')
    print(b)

    # Plot the results
    plt.figure(figsize=(10, 4))
    #plt.plot(time, dynamic_ma, label='Dynamic Moving Average', linewidth=2)
    plt.plot(time, smoothed_sg, label='Savitzky-Golay', linewidth=2)
    plt.plot(time[198:],np.exp(func_linear(time[198:],b,a)), label=r'$\rho_{fit}(x)$')
    #plt.plot(time, dynamic_smoothed_sg, label='Dynamic Savitzky-Golay', linewidth=2)
    plt.plot(time, data, linestyle='dashed', label=r'Original $\rho(x)$', alpha=0.5)
#    plt.plot(time, smoothed_exp_sg, label='Exponential Savitzky-Golay', linewidth=2)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Smoothing Oscillations in Data with Dynamic and Exponential Smoothing')
    plt.savefig('smoothed.pdf')
    plt.show()

    plt.plot(time[195:], np.log(smoothed_sg[195:]))
    plt.show()
solver()
