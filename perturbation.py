import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#import smoothers

def rhobar(time, parameters):
    c_2,A = parameters
    return (np.sqrt(c_2 - A*time))**(-1)

def deltarho(rhobar, parameters):
    E,A,c_2,d_1,d_2 = parameters
    g_1 = d_1 + 2 * E**2 * c_2**3 / (3 * A**2)
    return(g_1 * rhobar**3 - E**2 * rhobar**(-3)/ 12 + d_2 * rhobar**(-5))
#    return g_1 * rhobar**3

def epsilonh(time, parameters, init, other):
    
    # unpack parameters
    pi_0,E,a,b,l = parameters
    c_2,d_1,d_2 = other
    # Define parameters
#    pi_0 = 1.0
#    E = 2.0 # If E=pi_0, then \mu=0
#    a = 15.0
#    b = 15.0
#    l = 5.0
#    c_2 = 1.0
#    d_2 = 0.01
    A = np.sqrt(np.sqrt(a**2 + b**2)/3)
    par_rhobar = [c_2,A]
    par_deltarho = [E,A,c_2,d_1,d_2]
    
    # Initial conditions
#    rho_0 = 2.0
#    rho_prime_0 = A * rho_0**3
#    theta_0 = 0.0 # Picked based on a and b, such that a*sin()+b*cos()~0, where a=b. Next value is ~0.39.
#    theta_prime_0 = 10.0
#    y0 = [rho_0, rho_prime_0, theta_0, theta_prime_0]
    # unpack initial conditions
    rho_0,rho_prime_0,theta_0,theta_prime_0 = init

#    x_span = (0, 1) # Needs to be adjusted if the resolution drops
#    x_eval = np.linspace(*x_span, 1000) # May need to be adjusted in case of error messages

    # Calculate \epsilon_1=\epsilon_H
    epsilon_1_chi = 3 * rhobar(time,par_rhobar)**(-4) * E**2 / A**2 - 72 * d_2 * rhobar(time,par_rhobar)**(-6)
    epsilon_h_1 = 3 * rho_0**(-4) * E**2 / A**2
    epsilon_h_2 = - 72 * d_2 * rho_0**(-6)
#    epsilon_1_N = epsilon_h_1 * np.exp(-6*N) + epsilon_h_2 * np.exp(-9*N)
    # Calculate \epsilon_2

    # The following two quantities need to be small
    ## The "perturbation" value needs to be small in order to start at the perturbative regime
    term_kinetic_0 = (theta_prime_0**2 - 2*pi_0*theta_prime_0 + E**2)*rho_0
    term_potential_0 = (a*np.cos((l+1)*theta_0) - b*np.sin((l+1)*theta_0))*rho_0**l
    perturbation_0 = term_kinetic_0/term_potential_0
    print('perturbative perturbation parameter:',perturbation_0)

    ## Similarly, let's calculate the initial Hubble slow roll parameter
    rho_doubleprime_0 = term_kinetic_0 + term_potential_0
    epsilonh_0 = (9/2) - (3/2)*((rho_doubleprime_0*rho_0)/(rho_prime_0**2))
    print('perturbative epsilon_H:',epsilonh_0)

    return(epsilon_1_chi)
# 
#
#    # Plot with respect to \chi
#    plt.figure(figsize=(14, 7))
#    plt.plot(x_eval, epsilon_1_chi, label='Savitzky-Golay', linewidth=2)
#    plt.legend()
#    plt.xlabel('Time')
#    plt.ylabel('Amplitude')
#    plt.title('Smoothing Oscillations in Data with Dynamic and Exponential Smoothing')
#    plt.savefig('theta')
#    plt.show()
