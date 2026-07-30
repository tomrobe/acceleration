# This script is only meant to show how to produce json-compatible arrays from numpy arrays. Individual lines from the output can be pasted into the configuration file at the corresponding spots as they are.
import numpy as np
import json

N_init = 4

x_span = (0.1,40.0) # yf

#E = 1000 # phasespace
E = 10 # yf
m = 0.1 # yf
lamb = 10 # yf

mu = np.sqrt(1 + 3 * m / 4)

rho_0_base = 15.0 # phasespace

## rho theta
drho_0_base = E * rho_0_base
drho_0_arr = np.concatenate((np.ones(int(N_init/2))*(-0.5)*0.2, np.ones(int(N_init/2))*0.5)) * drho_0_base # phasespace
drho_0_list = drho_0_arr.tolist()

theta_0_base = 1.0
theta_0_arr = np.concatenate((np.linspace(-0.5,0.5,int(N_init/2))*theta_0_base, np.linspace(-0.5,0.5,int(N_init/2)) * theta_0_base)) # phasespace
theta_0_list = theta_0_arr.tolist()

dtheta_0_base = 1.0

## YF
y_0_base = np.sqrt(lamb/3) * (1 - 2* m**2 * x_span[0]**2 * (3/8 - ((1-mu)**2)/m**2)/(1+2*mu))
y_0_arr = np.ones(N_init) * y_0_base
y_0_list = y_0_arr.tolist()

f_0_base = - 2 * np.sqrt(lamb/3) * (1 - mu) * x_span[0]
f_0_arr = np.ones(N_init) * f_0_base
f_0_list = f_0_arr.tolist()


print("----------------")
print("Span:")
print("x_span: ",x_span)
print("----------------")
print("Used parameters:")
print("E: ", E)
print("m: ", m)
print("lamb: ", lamb)
print("----------------")
print("Fixed: ")
print("rho_0 :", rho_0_base)
print("dtheta_0: ", dtheta_0_base)
print("y_0: ", y_0_base)
print("f_0: ", f_0_base)
print("----------------")
print("Sweeps:")
print("init.drho_0: ", json.dumps(drho_0_list))
print("init.theta_0: ", json.dumps(theta_0_list))
print("init.y_0: ", json.dumps(y_0_list))
print("init.f_0: ", json.dumps(f_0_list))
print("----------------")
