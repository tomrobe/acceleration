################################
# ylim and xlim may need to be adjusted for each plot to make details visible.
################################


import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math

import smoothers
import perturbation

# Define the system of ODEs
def coupled_odes(x, y, phi, lamb, m, l, E):
    rho, rho_prime, theta, theta_prime = y
    rho_double_prime = (
        (theta_prime**2 + E**2) * rho
        - lamb * rho**l
    )
    theta_double_prime = (
        - 2 * rho_prime * theta_prime / rho
        - lamb * theta * rho**(l-1)
    )
    return [rho_prime, rho_double_prime, theta_prime, theta_double_prime]

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Main function
def solver():

    # Define parameters
#    pi_0 = 10
    E = 100.0
    phi = 0.0
    lamb = 10.0
    m = 1.0
    l = 5.0
    par = [phi,lamb,m,l,E]

    # Initial conditions
    rho_0 = 10.0
    rho_prime_0 = E * rho_0
#    rho_prime_0 = np.sqrt(lamb/3) * rho_0**3
    theta_0 = 0.0
    theta_prime_0 = -np.sqrt(lamb) * np.cos(theta_0) * rho_0**2
    y0 = [rho_0, rho_prime_0, theta_0, theta_prime_0]

    # The following two quantities need to be small
    ## The "perturbation_0" value needs to be small in order to start at the perturbative regime
    term_kinetic_0 = (theta_prime_0**2 + E**2) * rho_0
    term_potential_0 = - lamb * rho_0**l
    perturbation_0 = term_kinetic_0/term_potential_0
    print('perturbation parameter:',perturbation_0)

    ## Similarly, let's calculate the initial Hubble slow roll parameter
    rho_doubleprime_0 = term_kinetic_0 + term_potential_0
    epsilonh_0 = (9/2) - (3/2)*((rho_doubleprime_0*rho_0)/(rho_prime_0**2))
    print('initial epsilon_H:',epsilonh_0)
    
    # Solve the system over the interval x_span
    x_span = (0.0, 2.0)
    steps = 1000000
    x_eval = np.linspace(*x_span, steps)
    # Boundaries for plots
    blow = 0
    bup = steps

    sol= solve_ivp(coupled_odes, x_span, y0, method='RK45', args=(phi,lamb,m,l,E), t_eval=x_eval)

    # Check for discontinuity
    if (sol.t[-1]==x_span[1]):
        print('No discontinuity in range!')
    else:
        print('Discontinuity! Stopped at:',sol.t[-1])
        

    
    # Extract the solutions
    rho_sol= sol.y[0]
    rho_prime_sol= sol.y[1]
    theta_sol= sol.y[2]
    theta_prime_sol= sol.y[3]

    time = sol.t


    ################################ This may possibly be removed. Ignore for now
    # Apply smoothing methods
    window_factor = 10
    window_size = 100051  # Choose an odd number for the window size
    poly_order = 1

    # Pick an option
    ## With moving average
#    rho_smoothed_ma = smoothers.dynamic_moving_average(rho_sol, phase, window_factor)
#    rho_prime_smoothed_ma = smoothers.dynamic_moving_average(rho_prime_sol, phase, window_factor)
#    theta_smoothed_ma = smoothers.dynamic_moving_average(theta_sol, phase, window_factor)
#    theta_prime_smoothed_ma = smoothers.dynamic_moving_average(theta_prime_sol, phase, window_factor)

    ## With Savitzky-Golay
#    rho_smoothed_sg = smoothers.smooth_savgol(rho_sol, window_size, poly_order)
#    rho_prime_smoothed_sg = smoothers.smooth_savgol(rho_prime_sol, window_size, poly_order)
#    theta_smoothed_sg = smoothers.smooth_savgol(theta_sol, window_size, poly_order)
#    theta_prime_smoothed_sg = smoothers.smooth_savgol(theta_prime_sol, window_size, poly_order)
#    theta_prime_smoothed_sg_sg = smoothers.smooth_savgol(theta_prime_smoothed_sg, window_size, poly_order)

    ## With dynamic Savitzky-Golay
#    rho_dynamic_smoothed_sg= smoothers.dynamic_smooth_savgol(data, phase, window_factor, poly_order)
#    rho_prime_dynamic_smoothed_sg = smoothers.dynamic_smooth_savgol(rho_prime_sol, phase, window_factor, poly_order)
#    theta_dynamic_smoothed_sg = smoothers.dynamic_smooth_savgol(data, phase, window_factor, poly_order)
#    theta_prime_dynamic_smoothed_sg = smoothers.dynamic_smooth_savgol(data, phase, window_factor, poly_order)
    ################################

    # Pick *_sol, *_smoothed_sg, or others here to use in calculation below. Default works.
    rho = rho_sol
#    rho = rho_smoothed_sg
    rho_prime = rho_prime_sol
#    rho_prime = rho_prime_smoothed_sg
    theta = theta_sol
#    theta = theta_smoothed_sg
    theta_prime = theta_prime_sol
#    theta_prime = theta_prime_smoothed_sg

    # For exponential fit
#    popt, pcov = curve_fit(func, time[90000:90500], rho[90000:90500])
#    print(popt, pcov)

    kin_theta = theta_prime**2 * rho
    kin_E = E**2 * rho
    interaction = - lamb * rho**l
    rho_prime_prime = kin_theta + kin_E + interaction

    rho_prime_over_rho = rho_prime/rho
    rho_prime_prime_over_rho = rho_prime_prime/rho


    # Pick the time until which you want to take the average. Steps are rounded. I.e.
    avg_stop_time = 0.3
    avg_stop_step = round(steps*avg_stop_time /x_span[1])
    print('Average of rho prime/rho:',np.average(rho_prime_over_rho[:avg_stop_step]), 'at time:', avg_stop_time)
    print('Average of rho double prime/rho:',np.average(rho_prime_prime_over_rho[:avg_stop_step]), 'at time:',avg_stop_time)

    # Plot of rho
    plt.figure(figsize=(14, 7))
    plt.plot(time[blow:bup], rho_sol[blow:bup], label=r'$\rho(x)$')
##    plt.plot(time[blow:bup], rho_smoothed_sg[blow:bup], label='SG')
#    plt.plot(time[blow:bup], func(time, *popt), label='popt')
    plt.ylim(0,300)
    plt.grid()
    plt.legend()
    plt.xlabel('Clock')
    plt.ylabel('Amplitude')
    plt.title('Modulus of Condensate Wavefunction')
    plt.savefig('rho_oscillating')
    plt.show()

    # individual terms
    plt.figure(figsize=(14, 7))
    plt.plot(time[blow:bup], -rho_prime_prime[blow:bup], label='rho double prime')
    plt.plot(time[blow:bup], kin_theta[blow:bup], label='theta prime squared rho',alpha=0.5)
    plt.plot(time[blow:bup], kin_E[blow:bup], label='E squared rho')
    plt.plot(time[blow:bup], -interaction[blow:bup], label='interaction term')
    plt.ylim(-1000000,1000000)
    plt.grid()
    plt.legend()
    plt.xlabel('Clock')
    plt.ylabel('Amplitude')
    plt.title('Terms of Modulus Dynamics')
    plt.savefig('individualterms_oscillating')
    plt.show()

    # Plot of rho'
    plt.figure(figsize=(14, 7))
    plt.plot(time[blow:bup], rho_prime_sol[blow:bup], label=r'$\partial_{\chi} \rho(x)$')
#    plt.ylim(-10,10)
    plt.grid()
    plt.legend()
    plt.xlabel('Clock')
    plt.ylabel('Amplitude')
    plt.title('Derivative of Modulus of Condensate Wavefunction')
    plt.savefig('rhoprime_oscillating')
    plt.show()

    # Plot of rho'/rho
    plt.figure(figsize=(14,7))
    plt.plot(time[blow:bup], rho_prime_over_rho[blow:bup], label=r'$\frac{\partial_{\chi}\rho}{\rho}$')
    plt.ylim(-1000,1000)
    plt.grid()
    plt.legend()
    plt.xlabel('Clock')
    plt.ylabel('')
    plt.savefig('rhoprimeoverrho_oscillating')
    plt.show()

    # Plot of rho'''/rho
    plt.figure(figsize=(14,7))
    plt.plot(time[blow:bup], rho_prime_prime_over_rho[blow:bup], label=r'$\frac{\partial_{\chi}^{2}\rho}{\rho}$')
    plt.ylim(-1000000,1000000)
    plt.grid()
    plt.legend()
    plt.xlabel('Clock')
    plt.ylabel('')
    plt.savefig('rhoprimeprimeoverrho_oscillating')
    plt.show()

    # Plot of theta
    plt.figure(figsize=(14, 7))
    plt.plot(time[blow:bup], theta_sol[blow:bup], label=r'$\theta$')
#    plt.ylim(-10,10)
    plt.grid()
    plt.legend()
    plt.xlabel('Clock')
    plt.ylabel('Amplitude')
    plt.title('Phase of Condensate Wavefunction')
    plt.savefig('theta_oscillating')
    plt.show()

    # Plot of theta'
    plt.figure(figsize=(14, 7))
    plt.plot(time[blow:bup], theta_prime_sol[blow:bup], label=r'$\partial_{\chi} \theta$')
##    plt.plot(time[blow:bup], theta_prime_smoothed_sg_sg[blow:bup], label=r'$\partial_{\chi} \theta$')
    plt.ylim(-1000,1000)
    plt.grid()
    plt.legend()
    plt.xlabel('Clock')
    plt.ylabel('Amplitude')
    plt.title('Derivative of Phase of Condensate Wavefunction')
    plt.savefig('thetaprime_oscillating')
    plt.show()


    ################################    
    # Calculate epsilonh
    x = ((theta_prime)**2 + E**2) - lamb*(rho**(l-1))
    epsilonh = (9/2) - (3/2)*x*rho**2/(rho_prime**2)
    # Smooth rho_prime**2
    rho_prime_squared_sg = smoothers.smooth_savgol(rho_prime**2, window_size, poly_order)
    epsilonh_rhoprimesg = (9/2) - (3/2)*x*rho**2/(rho_prime_squared_sg)
#    epsilonh_sg = smoothers.smooth_savgol(epsilonh, window_size, poly_order)

    w = 2 - x*rho**2/(rho_prime**2)
    w_rhoprimesg = 2 - x*rho**2/(rho_prime_squared_sg )

    # Plot of slow roll parameter(s)
    plt.figure(figsize=(14, 7))
    plt.plot(time[blow:bup], epsilonh[blow:bup], label=r'$\epsilon_{H}$')
    plt.plot(time[blow:bup], epsilonh_rhoprimesg[blow:bup], label=r'$\epsilon_{H}$ w/ rho_prime**2 avgd')
#    plt.plot(time[blow:bup], epsilonh_sg[blow:bup], label=r'$\epsilon_{H}$ avgd')
    plt.plot(time[blow:bup], np.ones_like(epsilonh)[blow:bup], label='One', alpha=0.5)
#    plt.plot(time[blow:bup], epsilonh_perturbative[blow:bup], label='perturbed', alpha=0.5)
    plt.ylim(-20.0,20.0)
    plt.grid()
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Slow Roll parameter')
    plt.savefig('epsilonh_oscillating')
    plt.show()

    # w
    plt.figure(figsize=(14, 7))
    plt.plot(time[blow:bup], w[blow:bup], label='w')
    plt.plot(time[blow:bup], w_rhoprimesg[blow:bup], label='w w/ rho_prime**2 avgd')
    plt.plot(time[blow:bup], np.ones_like(w)[blow:bup], label='One', alpha=0.5)
    plt.plot(time[blow:bup], -np.ones_like(w)[blow:bup], label='-One', alpha=0.5)
    plt.ylim(-20.0,20.0)
    plt.grid()
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Equation of State parameter')
    plt.savefig('w_oscillating')
    plt.show()

solver()
