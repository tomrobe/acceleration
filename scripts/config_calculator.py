# This script is only meant to show how to produce json-compatible arrays from numpy arrays. Individual lines from the output can be pasted into the configuration file at the corresponding spots as they are.
import numpy as np
import json

N_init = 4

E = 1000 # phasespace

rho_0_base = 15.0 # phasespace

drho_0_base = E * rho_0_base
drho_0_arr = np.concatenate((np.ones(int(N_init/2))*(-0.5)*0.2, np.ones(int(N_init/2))*0.5)) * drho_0_base # phasespace
drho_0_list = drho_0_arr.tolist()

theta_0_base = 1.0
theta_0_arr = np.concatenate((np.linspace(-0.5,0.5,int(N_init/2))*theta_0_base, np.linspace(-0.5,0.5,int(N_init/2)) * theta_0_base)) # phasespace
theta_0_list = theta_0_arr.tolist()

dtheta_0_base = 1.0

print("----------------")
print("Used parameters:")
print("E:", E)
print("----------------")
print("Fixed:")
print("rho_0 :", rho_0_base)
print("dtheta_0:", dtheta_0_base)
print("----------------")
print("Sweeps:")
print("init.drho_0: ", json.dumps(drho_0_list))
print("init.theta_0: ", json.dumps(theta_0_list))
print("----------------")
