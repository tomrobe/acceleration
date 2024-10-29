################################
# 
# If the plot for epsilonh gets "spiky", uncomment the plt.ylim(,) line and adjust the arguments if necessary. It totally changes the appearence in those cases.
# Similarly, if a plot is very large somewhere and you feel like some detail may be lost due to scaling, you can use plt.ylim(,) or adjust the blow and bup values to look only at a certain range of chi.
#
################################

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

import smoothers
import perturbation

# define other constants that appear in perturbative analysis
c_2 = 10.0
d_1 = 0.0
d_2 = 0.0
pert_parameters = [c_2,d_1,d_2]

# Define the system of ODEs
def coupled_odes(x, y, pi_0, E, a, b, l):
    rho, rho_prime, theta, theta_prime = y
    rho_double_prime = (
        (theta_prime**2 - 2*pi_0*theta_prime + E**2) * rho
        + (a * np.cos((l+1)*theta) - b * np.sin((l+1)*theta)) * rho**l
    )
    theta_double_prime = (
        - 2 * rho_prime * (theta_prime - pi_0) / rho
        - (a * np.sin((l+1)*theta) + b * np.cos((l+1)*theta)) * rho**(l-1)
    )
    return [rho_prime, rho_double_prime, theta_prime, theta_double_prime]

# Main function
def solver():

    # Define parameters
    pi_0 = 1
    E = 2 # If E=pi_0, then \mu=0
    a = 10.0
    b = 10.0
    l = 5.0
    par = [pi_0,E,a,b,l]
    
    # Initial conditions
    rho_0 = 10.0
    rho_prime_0 = np.sqrt( np.sqrt(a**2 + b**2) / 3) * rho_0**3
    theta_0 = 0.13 #-math.pi / 24
 # Picked based on a and b, such that a*sin()+b*cos()~0, where a=b. Next value is ~0.39.
    theta_prime_0 = 1.0
    y0 = [rho_0, rho_prime_0, theta_0, theta_prime_0]

    # The following two quantities need to be small
    ## The "perturbation_0" value needs to be small in order to start at the perturbative regime
    term_kinetic_0 = (theta_prime_0**2 - 2*pi_0*theta_prime_0 + E**2)*rho_0
    term_potential_0 = (a*np.cos((l+1)*theta_0) - b*np.sin((l+1)*theta_0))*rho_0**l
    perturbation_0 = term_kinetic_0/term_potential_0
    print('perturbation parameter:',perturbation_0)

    ## Similarly, let's calculate the initial Hubble slow roll parameter
    rho_doubleprime_0 = term_kinetic_0 + term_potential_0
    epsilonh_0 = (9/2) - (3/2)*((rho_doubleprime_0*rho_0)/(rho_prime_0**2))
    print('initial epsilon_H:',epsilonh_0)
    
    # Solve the system over the interval x_span
    x_span = (0.0, 0.1) # Needs to be adjusted if the resolution drops
    x_eval = np.linspace(*x_span, 10000) # May need to be adjusted in case of error messages
    # Boundaries for plots
    blow = 0
    bup = 10000
    
    sol= solve_ivp(coupled_odes, x_span, y0, args=(pi_0, E, a, b, l), t_eval=x_eval)
    
    # Extract the solutions
    rho_sol= sol.y[0]
    rho_prime_sol= sol.y[1]
    theta_sol= sol.y[2]
    theta_prime_sol= sol.y[3]

    time = sol.t

    # Calculate results from perturbative analysis for reference
    rhobar_perturbative = perturbation.rhobar(time,[c_2,np.sqrt( np.sqrt(a**2 + b**2) / 3)])
    deltarho_perturbative = perturbation.deltarho(rhobar_perturbative,[E,np.sqrt( np.sqrt(a**2 + b**2) / 3),c_2,d_1,d_2])
    rho_perturbative = rhobar_perturbative + deltarho_perturbative 

    ################################ This may possibly be removed. Ignore for now
    # Apply smoothing methods
    window_factor = 10
    window_size = 51  # Choose an odd number for the window size
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

    # Plot of rho
    plt.figure(figsize=(14, 7))
    plt.plot(time[blow:bup], rho_sol[blow:bup], label=r'$\rho(x)$')
#    plt.ylim(-10,10)
    plt.legend()
    plt.xlabel('Clock')
    plt.ylabel('Amplitude')
    plt.title('Modulus of Condensate Wavefunction')
    plt.savefig('rho')
    plt.show()

    # Plot of theta
    plt.figure(figsize=(14, 7))
    plt.plot(time[blow:bup], theta_sol[blow:bup], label=r'$\theta$')
#    plt.ylim(-10,10)
    plt.legend()
    plt.xlabel('Clock')
    plt.ylabel('Amplitude')
    plt.title('Phase of Condensate Wavefunction')
    plt.savefig('theta')
    plt.show()

    ################################    
    # Calculate epsilonh
    x = ((theta_prime)**2 - 2*pi_0*theta_prime + E**2)*(rho**2) + (a*np.cos((l+1)*theta) - b*np.sin((l+1)*theta))*(rho**(l+1))
    epsilonh = (9/2) - (3/2)*x/(rho_prime**2)

    # Also for perturbative analysis
    epsilonh_perturbative = perturbation.epsilonh(time,par,y0,pert_parameters)

    # Plot of slow roll parameter(s)
    plt.figure(figsize=(14, 7))
    plt.plot(time[blow:bup], epsilonh[blow:bup], label=r'$\epsilon_{H}$')
    plt.plot(time[blow:bup], np.ones_like(epsilonh)[blow:bup], label='One', alpha=0.5)
#    plt.plot(time[blow:bup], epsilonh_perturbative[blow:bup], label='perturbed', alpha=0.5)
#    plt.ylim(-10,10)
    plt.grid()
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Slow Roll parameter')
    plt.savefig('epsilonh')
    plt.show()
solver()
