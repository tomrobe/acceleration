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
    pi_0 = 1
    E = 2 # If E=pi_0, then \mu=0
    a = 2.1
    b = 2.1
    l = 5.0
    
    # Initial conditions
    rho_0 = 9
    rho_prime_0 = rho_0**3
    theta_0 = -0.13 # Picked based on a and b, such that a*sin()+b*cos()~0, where a=b. Next value is ~0.39.
    theta_prime_0 = 1.0
    y0 = [rho_0, rho_prime_0, theta_0, theta_prime_0]

    # The following two quantities need to be small
    ## The "perturbation" value needs to be small in order to start at the perturbative regime
    term_kinetic_0 = (theta_prime_0**2 - 2*pi_0*theta_prime_0 + E**2)*rho_0
    term_potential_0 = (a*np.cos((l+1)*theta_0) - b*np.sin((l+1)*theta_0))*rho_0**l
    perturbation_0 = term_kinetic_0/term_potential_0
    print(perturbation_0)

    ## Similarly, let's calculate the initial Hubble slow roll parameter
    rho_doubleprime_0 = term_kinetic_0 + term_potential_0
    epsilonh_0 = (9/2) - (3/2)*((rho_doubleprime_0*rho_0)/(rho_prime_0**2))
    print(epsilonh_0)
    
    # Solve the system over the interval x_span
    x_span = (0, 0.06) # Needs to be adjusted if the resolution drops
    x_eval = np.linspace(*x_span, 1000) # May need to be adjusted in case of error messages
    
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
    poly_order = 1

    rho_dynamic_ma = smoothers.dynamic_moving_average(data, phase, window_factor)
    rho_smoothed_sg = smoothers.smooth_savgol(data, window_size, poly_order)
    rho_dynamic_smoothed_sg = smoothers.dynamic_smooth_savgol(data, phase, window_factor, poly_order)
#    smoothed_exp_sg = smoothers.smooth_exp_savgol(rho_sol, window_size)
    
    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(time, data, label=r'Original $\rho(x)$', alpha=0.5)
#    plt.plot(time, rho_dynamic_ma, label='Dynamic Moving Average', linewidth=2)
    plt.plot(time, rho_smoothed_sg, label='Savitzky-Golay', linewidth=2)
#    plt.plot(time, rho_dynamic_smoothed_sg, label='Dynamic Savitzky-Golay', linewidth=2)
#    plt.plot(time, rho_smoothed_exp_sg, label='Exponential Savitzky-Golay', linewidth=2)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Smoothing Oscillations in Data with Dynamic and Exponential Smoothing')
    plt.show()

##########################
    # Plotting epsilon_H

    # To use smoothed versions of rho, theta, theta_prime, uncomment the block you want. Uncommenting all of them is fine too.

    ## With moving average
#    rho_smoothed_ma = smoothers.dynamic_moving_average(rho_sol, phase, window_factor)
#    rho_prime_smoothed_ma = smoothers.dynamic_moving_average(rho_prime_sol, phase, window_factor)
#    theta_smoothed_ma = smoothers.dynamic_moving_average(theta_sol, phase, window_factor)
#    theta_prime_smoothed_ma = smoothers.dynamic_moving_average(theta_prime_sol, phase, window_factor)

    ## With Savitzky-Golay
    rho_prime_smoothed_sg = smoothers.smooth_savgol(rho_prime_sol, window_size, poly_order)
    theta_smoothed_sg = smoothers.smooth_savgol(theta_sol, window_size, poly_order)
    theta_prime_smoothed_sg = smoothers.smooth_savgol(theta_prime_sol, window_size, poly_order)

    ## With dynamic Savitzky-Golay
#    rho_dynamic_smoothed_sg= smoothers.dynamic_smooth_savgol(data, phase, window_factor, poly_order)
#    rho_prime_dynamic_smoothed_sg = smoothers.dynamic_smooth_savgol(rho_prime_sol, phase, window_factor, poly_order)
#    theta_dynamic_smoothed_sg = smoothers.dynamic_smooth_savgol(data, phase, window_factor, poly_order)
#    theta_prime_dynamic_smoothed_sg = smoothers.dynamic_smooth_savgol(data, phase, window_factor, poly_order)

    # Plot of theta, again. For reference.
    plt.figure(figsize=(14, 7))
    plt.plot(time, theta_sol, label=r'Original $\theta$', alpha=0.5)
#    plt.plot(time, rho_dynamic_ma, label='Dynamic Moving Average', linewidth=2)
    plt.plot(time, theta_smoothed_sg, label='Savitzky-Golay', linewidth=2)
#    plt.plot(time, rho_dynamic_smoothed_sg, label='Dynamic Savitzky-Golay', linewidth=2)
#    plt.plot(time, rho_smoothed_exp_sg, label='Exponential Savitzky-Golay', linewidth=2)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Smoothing Oscillations in Data with Dynamic and Exponential Smoothing')
    plt.show()

###########################
    # First try for epsilonH, not recommended because it probably leads to more numerical error.
    epsilonHtimesrhoprimesquared = (9/2) * (rho_prime_sol**2) - ((3/2) * (((theta_prime_sol**2 - 2*pi_0*theta_prime_sol + E**2))) - (3/2) * ((a * np.cos((l+1)*theta_sol) - b * np.sin((l+1)*theta_sol)) * rho_sol**(l-1)) ) * rho_sol**2

    temp1 = smoothers.smooth_savgol(epsilonHtimesrhoprimesquared, window_size, poly_order)
    temp2 = smoothers.smooth_savgol(rho_prime_sol**2, window_size, poly_order)

    eps = np.divide(temp1, temp2)
    epssmooth = smoothers.smooth_savgol(eps, window_size, poly_order)

############################
    # same same, but different
    # Pick *_sol, *_smoothed_sg, or others here to use in calculation below. Default works.
    rho = rho_sol
    rho_prime = rho_prime_sol
    theta = theta_sol
    theta_prime = theta_prime_sol

    # Calculate epsilonh
    x = ((theta_prime)**2 - 2*pi_0*theta_prime + E**2)*(rho**2) + (a*np.cos((l+1)*theta) - b*np.sin((l+1)*theta))*(rho**(l+1))
    x_smoothed_sg = smoothers.smooth_savgol(x, window_size, poly_order)
    temp = smoothers.smooth_savgol(rho_prime**2, window_size, poly_order)
    y = x_smoothed_sg/temp
    epsilonh = (9/2) - (3/2)*y
    # Variants:
    ## The result epsilonh may be smoothed out:
    epsilonh_smoothed_sg = smoothers.smooth_savgol(epsilonh, window_size, poly_order)
    ## Alternatively to epsilonh, to avoid singularities only the rho_prime**2 (denominator) needs to be smoothed out, and you can use x instead of x_smoothed_sg:
    epsilonh_smoothed_denominator_sg = (9/2) - (3/2)*x/temp

    # Plot. Comment out lines to reduce clutter.
    plt.figure(figsize=(14, 7))
    plt.plot(time, eps, label='eps', alpha=0.5)
    plt.plot(time, epssmooth, label='eps', alpha=0.5)
    plt.plot(time, epsilonh, label=r'$\epsilon_{H}$', alpha=0.5)
    plt.plot(time, epsilonh_smoothed_sg, label=r'$\epsilon_{H}$', alpha=0.5)
    plt.plot(time, epsilonh_smoothed_denominator_sg, label=r'$\epsilon_{H}$', alpha=0.5)
    plt.plot(time, np.ones_like(epsilonh), label='1', alpha=0.5)
    plt.grid()
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('')
    plt.show()
solver()
