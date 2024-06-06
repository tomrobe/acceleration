# packages

import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

# ODE system
def system(y:list[float], t:float, par:list[float]):
    """
    y : array-like - initial values
    t : float - clock
    par : tuple - parameters
    """
    # unpack variables
    rho, vrho, theta, vtheta = y

    # unpack parameters
    p, e, a = par

    # define derivatives
    drhodt = vrho
    dvrhodt: float = ( vtheta**2 - 2 * p * vtheta + e**2) * rho + 3 * a**2 * np.cos(6 * theta) * rho**5
    dthetadt = vtheta
    dvthetadt: float = ( - 2 * vrho * ( vtheta - p) / rho + 3 * a**2 * np.sin(6 * theta) * rho**4)

    return [drhodt, dvrhodt, dthetadt, dvthetadt]

def avg(f:list[float], t:float):
    window = 100000
    average_f = []
    for ind in range(len(f) - window + 1):
        average_f.append(np.mean(f[ind:ind+window]))
    for ind in range(window - 1):
        average_f.insert(0, np.nan)
    return average_f


def main():
# parameters
    PI_TILDE_ZERO = 1.0
    ENERGY = PI_TILDE_ZERO + 1.0
    SMALL_A = 3
    SMALL_B = 0
    PAR = [PI_TILDE_ZERO,ENERGY,np.sqrt(SMALL_A**2 + SMALL_B**2) / 3]

# initial conditions
    RHO0 = 1.0
    VRHO0 = RHO0 * 10.0**3
    THETA0 = 0.0
    VTHETA0 = 10.0
    Y0 = [RHO0,VRHO0,THETA0,VTHETA0]

# range
    t = np.linspace(1.0,1.044,1000000)

# solve
    solution = odeint(system, Y0, t, args=(PAR,))

# extract
    rho = solution[:,0]
    vrho = solution[:,1]
    theta= solution[:,2]
    vtheta = solution[:,3]

    print("A = ", np.sqrt(SMALL_A**2 + SMALL_B**2) / 3)

# plot
    plt.figure(figsize=(12,8))
#plt.plot(t,rho,label='rho')
#plt.plot(t,vrho,label='d rho / d chi')
#plt.plot(t,theta,label='theta')
    plt.plot(t,vtheta,label='d theta / d chi ')
#plt.plot(t,vtheta/theta,label='d theta / d chi / theta')
    plt.plot(t,vrho/rho,label='(d rho / d chi) / rho')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')
    plt.legend()
    plt.grid(True)
#plt.show()
#plt.savefig('1')

    plt.figure(figsize=(12,8))
    plt.plot(t,rho,label='rho')
    plt.plot(t,avg(rho,t),label='avg of rho')
#plt.plot(t,vrho,label='d rho / d chi')
#plt.plot(t,theta,label='theta')
#plt.plot(t,vtheta,label='d theta / d chi ')
#plt.plot(t,vtheta/theta,label='d theta / d chi / theta')
#plt.plot(t,vrho/rho,label='(d rho / d chi) / rho')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')
    plt.legend()
    plt.grid(True)
#plt.show()
#plt.savefig('2')


    plt.figure(figsize=(12,8))
#plt.plot(t,rho,label='rho')
#plt.plot(t,vrho,label='d rho / d chi')
    plt.plot(t,theta,label='theta')
#plt.plot(t,vtheta,label='d theta / d chi ')
#plt.plot(t,vtheta/theta,label='d theta / d chi / theta')
#plt.plot(t,vrho/rho,label='(d rho / d chi) / rho')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')
    plt.legend()
    plt.grid(True)
    plt.show()
#plt.savefig('3')

main()
