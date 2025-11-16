import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import diags
from scipy.integrate import solve_ivp
from NeuralNetworks.numerical_integration import *
torch.set_default_dtype(torch.float32)


class ForwardPadding(nn.Module):
    """
    Performs periodic forward padding on the tensor u.
    Ads the first d elements of tensor u to the end of the tensor u.
    """
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, u):
        return torch.cat([u, u[..., : self.d]], dim=-1)
        


class Summation(nn.Module):
    """
    Sum over all dimensions except for the first one (batch or channel dimension) while keeping the dimensionality.
    Sums all other dimensions into a single value per item and returns the result with the same number of dimensions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, u):
        axis = tuple(range(1, u.ndim))
        return u.sum(axis=axis, keepdims=True)
    

class CentralPadding(nn.Module):
    """
    Symmetric periodic padding. Adds the first d elements to the end of the last dimension by repeating the first d elements.
    """
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, u):
        return torch.cat([u[..., -self.d :], u, u[..., : self.d]], dim=-1)
       
    


class BaseNN(torch.nn.Module):
    def __init__(self,nstates,noutputs=1, hidden_dim=50, timedependent = False, statedependent = True, spacedependent=False, act_1 = nn.Tanh(), act_2 = nn.Tanh()):
        #State dependent

        super().__init__()

        self.nstates = nstates
        self.noutputs = noutputs
        self.hidden_dim = hidden_dim

        self.timedependent = timedependent
        self.statedependent = statedependent
        self.spacedependent = spacedependent

        self.act_1 = act_1
        self.act_2 = act_2
      
        if not statedependent and not timedependent:
            input_dim = 1
            linear1 = nn.Linear(input_dim, hidden_dim)
            linear2 = nn.Linear(hidden_dim, hidden_dim)
            linear3 = nn.Linear(hidden_dim, noutputs)

            self.model = nn.Sequential(
                linear1,
                act_1,
                linear2,
                act_2,
                linear3,
            )
        else:
            input_dim = 1
            pad = ForwardPadding(d=1)
            conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=2)
            conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=None)
            summation = Summation()

            self.model = nn.Sequential(
                pad,
                conv1,
                act_1,
                conv2,
                act_2,
                conv3,
                summation,
            )

        forward_map = {
            (True,  True,  True):  self.forward_space_time_state,
            (True,  True,  False): self.forward_time_space,
            (True,  False, True):  self.forward_time_state,
            (False, True,  True):  self.forward_space_state,
            (True,  False, False): self.forward_time,
            (False, True,  False): self.forward_space,
            (False, False, True):  self.forward_state,
            (False, False, False): self.forward_general,
        }

        key = (timedependent, spacedependent, statedependent)
        self.forward = forward_map[key]
       
        

    def spatial_basis(self, xspatial = None):
        xsbasis = torch.cat(
            [
                torch.sin(2 * torch.pi / self.period * xspatial),
                torch.cos(2 * torch.pi / self.period * xspatial),
            ],
            axis=-2,
        )
        return xsbasis
    
    def forward_general(self, u=None, t=None, xspatial=None):
        return self.model

    def forward_state(self, u=None, t=None, xspatial=None):
        #Only state dependent, see PDEBaseNN in https://github.com/SINTEF/pseudo-hamiltonian-neural-networks/blob/091065ef3c1b730d56fd413b6373d0424d8114be/phlearn/phlearn/phnns/pde_models.py
        return self.model(u)
    
    def forward_space(self, u=None, t=None, xspatial=None):
        xsbasis = self.spatial_basis(xspatial=xspatial)
        return self.model(xsbasis)
    
    def forward_time(self, u=None, t=None, xspatial=None):
        return self.model(t)
    
    def forward_space_state(self, u=None, t=None, xspatial=None):
        xsbasis = self.spatial_basis(xspatial=xspatial)
        return self.model(torch.cat([u, xsbasis], dim=-2))
    
    def forward_time_state(self, u=None, t=None, xspatial=None):
        ts = t.repeat_interleave(u.shape[-1], dim=-1)
        return self.model(torch.cat([u, ts], dim=-2))
    
    def forward_time_space(self, u=None, t=None, xspatial=None):
        xsbasis = self.spatial_basis(xspatial=xspatial)
        ts = t.repeat_interleave(u.shape[-1], dim=-1)
        return self.model(torch.cat([xsbasis, ts], dim=-2))
    
    def forward_space_time_state(self, u=None, t=None, xspatial=None):
        xsbasis = self.spatial_basis(xspatial=xspatial)
        ts = t.repeat_interleave(u.shape[-1], dim=-1)
        return self.model(torch.cat([u, xsbasis, ts], dim=-2))


class PDEIntegralNN(BaseNN):
    def __init__(self, nstates, hidden_dim=100):
        super().__init__(nstates =nstates, noutputs = 1, hidden_dim=hidden_dim)


class PDEExternalForcesNN(BaseNN):
    def __init__(self,nstates,hidden_dim=100,period=20, timedependent=False, spacedependent=True, statedependent=False):
        noutputs = 1
        super().__init__(nstates,noutputs,hidden_dim,timedependent,statedependent,spacedependent=spacedependent)
        self.nstates = nstates
        self.noutputs = noutputs
        self.hidden_dim = hidden_dim
        self.spacedependent = spacedependent
        self.timedependent = timedependent
        self.statedependent = statedependent
        self.period = period


        if not statedependent and not timedependent and not spacedependent:
            self.model = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        else:
            input_dim = (
                int(statedependent) + int(timedependent) + 2 * int(spacedependent)
            )
            conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
            conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            conv3 = nn.Conv1d(hidden_dim, noutputs, kernel_size=1)

            self.model = nn.Sequential(
                conv1,
                nn.Tanh(),
                conv2,
                nn.Tanh(),
                conv3,
            )

            self.input_dim = input_dim


class SymConvOperator_estimator(torch.nn.Module):
    """Estimates a symmetric convolution operator applied to the left-hand side of a PDE.
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        d = int((kernel_size - 1) / 2)
        self.d = d
        self.ls = torch.nn.Parameter(torch.zeros(d), requires_grad=True)

    def forward(self, u=None):
        #Damping Matrix
        if self.kernel_size == 0:
            return torch.tensor([0]).reshape(1, 1, 1)
        else:
            return torch.concat([self.ls, torch.tensor([1]), self.ls]).reshape(1, 1, self.kernel_size)



class SkewSymConvOperator_estimator(torch.nn.Module):
   
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        if self.kernel_size > 1:
            d = int((kernel_size - 3) / 2)
            self.d = d
            self.ls = torch.nn.Parameter(torch.zeros(d), requires_grad=True)

    def forward(self, u=None):
        #Damping Matrix
        if self.kernel_size == 0 or self.kernel_size == 1:
            return torch.tensor([0]).reshape(1, 1, 1)
        else:
            return torch.concat([-self.ls, torch.tensor([-1.0, 0.0, 1.0]), self.ls]).reshape(1, 1, self.kernel_size)


class PDE_PHNN(torch.nn.Module):
    def __init__(self,nstates,kernel_sizes=[1, 3, 1, 0], init_sampler=None, V=None, H=None, f=None, R=None, S=None, A=None):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.nstates = nstates
        self.d_type = torch.float32

        
        if init_sampler is not None:
            self._initial_condition_sampler = init_sampler
      
        if V is not None:
            self.V = V
        else:
            self.V = PDEIntegralNN(nstates=self.nstates)
        if H is not None:
            self.H = H
        else:
            self.H = PDEIntegralNN(nstates=self.nstates)


        if S is not None:
            self.S = S
            self.is_trainable_S = False
        else:
            self.S = SkewSymConvOperator_estimator(kernel_size=self.kernel_sizes[1])
            self.is_trainable_S = True
        if R is not None:
            self.R = R
            self.is_trainable_R = False
        else:
            self.R = SymConvOperator_estimator(kernel_size=self.kernel_sizes[2])
            self.is_trainable_R = True
        if A is not None:
            self.A = A
            self.is_trainable_A = False
        else:
            self.A = SymConvOperator_estimator(kernel_size=self.kernel_sizes[0])
            self.is_trainable_A = True


        
    def dV(self, u):
        u = u.clone().requires_grad_(True)
        return torch.autograd.grad(self.V(u).sum(),u,retain_graph=self.training,create_graph=self.training)[0]
    
    def dV_correction(self):
        self.dV_original = self.dV
        self.dV = self.dV_corrected
    
    def dV_corrected(self, u):
        return self.dV_original(u) - self.dV_original(torch.tensor(((0,),)))
    
    def dV_corrected(self, u):
        u0 = torch.zeros_like(u, dtype=u.dtype, device=u.device)
        return self.dV_original(u) - self.dV_original(u0)


    def dH(self, u):
        u = u.clone().requires_grad_(True)
        return torch.autograd.grad(self.H(u).sum(),u,retain_graph=self.training,create_graph=self.training)[0]
    
    def u_dot(self, u, t, xspatial=None):

        rhs = torch.zeros_like(u)

        if self.kernel_sizes[1] != 0:
            if self.is_trainable_S:
                S = self.S()
            else:
                S = self.S
            if self.H is None:
                dH = torch.zeros_like(u)
            else:
                dH = self.dH(u)
    
            d = int((self.kernel_sizes[1]-1) / 2)
            dH_padded = torch.cat([dH[..., self.nstates -d:], dH, dH[..., :d]], dim=-1)
       
            rhs += torch.nn.functional.conv1d(dH_padded, S)

        if self.kernel_sizes[2] != 0:
            if self.is_trainable_R:
                R = self.R()
            else:
                R = self.R
            if self.V is None:
                dV = torch.zeros_like(u)
            else:
                dV = self.dV(u)
            d = int((self.kernel_sizes[2]-1) / 2)
            dV_padded = torch.cat([dV[..., self.nstates -d:], dV, dV[..., :d]], dim=-1)
            rhs += -torch.nn.functional.conv1d(dV_padded, R)

        if self.kernel_sizes[3] != 0:
            if self.f is not None:
                rhs += self.kernel_sizes[3] * self.f(u, t, xspatial)

        return rhs
    
    def lhs(self,dudt):
        if self.kernel_sizes[0] == 1:
            return dudt
        else:
            if self.is_trainable_A:
                A = self.A()
            else:
                A = self.A
            d = int((self.kernel_sizes[0] - 1) / 2)
            dudt_pad = torch.cat([dudt[..., self.nstates - d :], dudt, dudt[..., :d]], dim=-1)
            A_dudt = torch.nn.functional.conv1d(dudt_pad, A)
            return A_dudt
        

    def time_derivative_step(self, integrator, u_start, dt, t_start, t_end = None ,u_end=None, xspatial=None):
        if integrator == "RK4":
            return RK4_time_derivative(self.u_dot ,u_start, dt, t_start, xspatial)
        elif integrator == "midpoint":
            u_mid = (u_end + u_start) / 2
            if t_end == None:
                t_end = t_start + dt
            t_mid = (t_end + t_start) / 2
            return self.u_dot(u_mid, t_mid, xspatial=xspatial)

                                                      
     
    def simulate_trajectory(self, t_sample,u0=None, xspatial=None):
        u0 = torch.tensor(u0)
        if u0 is None:
            u0 = self._initial_condition_sampler(1)

        M = u0.shape[0]
        if self.kernel_sizes[0] == 1:
            if xspatial is not None:
                def u_dot(t, u):
                    u = torch.tensor(u.reshape(1, u.shape[-1]), dtype=torch.float32)
                    t = torch.tensor(np.array(t).reshape((1, 1)), dtype=torch.float32)
                    xspatial =torch.tensor(np.array(xspatial).reshape(1, xspatial.shape[-1]), dtype=torch.float32)
                    return self.u_dot(u,t,xspatial=xspatial).detach().numpy().flatten()
            else:
                def u_dot(t, u):
                    u = torch.tensor(u.reshape(1, u.shape[-1]), dtype=torch.float32)
                    t = torch.tensor(np.array(t).reshape((1, 1)), dtype=torch.float32)
                    return self.u_dot(u,t).detach().numpy().flatten()
        else:
            d = int((self.kernel_sizes[0] - 1) / 2)
            A = self.A().detach().numpy()
            diagonals = np.concatenate([A[0, :, (d + 1) :], A[0], A[0, :, : -(d + 1)]], axis=1).T.repeat(M, axis=1)
            offsets = np.concatenate([np.arange(-M + 1, -M + 1 + d), np.arange(-d, d + 1), np.arange(M - d, M)])
            D = diags(diagonals, offsets, (M, M)).toarray()

            if xspatial is not None:
                def u_dot(t, u):
                    u = torch.tensor(u.reshape(1, u.shape[-1]), dtype=torch.float32)
                    t = torch.tensor(np.array(t).reshape((1, 1)), dtype=torch.float32)
                    xspatial = torch.tensor(np.array(xspatial).reshape(1, xspatial.shape[-1]), dtype=torch.float32)
                    return np.linalg.solve(D,self.u_dot(u, t, xspatial).detach().numpy().flatten())
            else:
                def u_dot(t, u):
                    u = torch.tensor(u.reshape(1, u.shape[-1]), dtype=torch.float32)
                    t = torch.tensor(np.array(t).reshape((1, 1)), dtype=torch.float32)
                    return np.linalg.solve(D,self.u_dot(u, t).detach().numpy().flatten())
                
        out_ivp = solve_ivp(fun=u_dot,t_span=(t_sample[0], t_sample[-1]),y0=u0.detach().numpy().flatten(),t_eval=t_sample,rtol=1e-10)
        U = out_ivp["y"].T

        return U, None