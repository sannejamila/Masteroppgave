import numpy as np
import torch
from scipy.optimize import newton


def newton_torch(func, guess, threshold=1e-7, max_iters=100, damping=1.0):
    guess = torch.tensor(guess, dtype=torch.float32, requires_grad=True)
    for i in range(max_iters):
        value = func(guess) 
        if torch.linalg.norm(value) < threshold:
            return guess
        J = torch.autograd.functional.jacobian(func, guess)  
        try:
            step = torch.linalg.solve(J, -value)  
        except RuntimeError:
            #print("Jacobian is singular, stopping.")
            return guess
        guess = guess + damping * step 
    return guess

def RK4_time_derivative(u_dot,u_start, dt, t_start = None, xspatial=None):
    #u_dot(self, u, t, xspatial=None):
    k1 = u_dot(u_start, t_start, xspatial=xspatial)
    k2 = u_dot(u_start + dt/2*k1, t_start + dt/2, xspatial)
    k3 = u_dot(u_start + dt/2*k2, t_start + dt/2, xspatial)
    k4 = u_dot(u_start+ dt*k3, t_start + dt, xspatial)

    return 1/6*(k1+2*k2+2*k3+k4)


def explicit_midpoint_time_derivative(u_dot,u_start, dt, t_start = None, xspatial=None):
    #u_dot(self, u, t, xspatial=None):
    u_temp = u_start + dt/2*u_dot(u_start, t_start, xspatial)
    lhs = u_dot(u_temp, t_start +dt/2, xspatial)
    return lhs



def symplectic_midpoint_time_derivative(u_dot, u_start, dt, t_start = None, t_end = None, xspatial=None, u_end=None):
    #u_dot(self, u, t, xspatial=None)
    if t_end == None:
        t_end = t_start + dt/2
    if u_end is None:
        def g(u):
            return u - u_start - dt * u_dot(u = 0.5 * (u + u_start), t = t_end, xspatial = xspatial)

        if isinstance(u_start, torch.Tensor):
            original_shape = u_start.shape
            u_start = u_start.squeeze(0)
            u_end = newton_torch(g, u_start)
            u_mid = 0.5 * (u_start + u_end)
            return u_dot(u_mid.view(original_shape), t_end, xspatial)
        else:
            u_end = newton(g, u_start)
    u_mid = 0.5 * (u_start + u_end)
    return u_dot(u_mid, t_end, xspatial)



def symplectic_euler(u_dot,u_start,dt, t_start = None, xspatial=None):
    g = u_dot(u_start,t_start, xspatial)
    f = u_dot(u_start+g*dt, t_start, xspatial)
    if isinstance(u_start,torch.Tensor):
        if u_start.ndim == 1:
            rhs = torch.cat((f[0:2], g[2:4]))
        else:
            rhs = torch.cat((f[:,0:2],g[:,2:4]),axis = 1)
    else:
        rhs = np.concatenate((f[0:2], g[2:4]))
    return rhs