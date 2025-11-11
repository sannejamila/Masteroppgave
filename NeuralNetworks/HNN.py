import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional, Sequence, Union
from activationfunctions import *
from numerical_integration import *
torch.set_default_dtype(torch.float32)


class BaseHamiltonianNeuralNetwork(nn.Module):
    def __init__(self, nstates: int,noutputs: int = 1,hidden_dim: int = 100, act_1: nn.Module = PAU(), act_2: nn.Module = nn.Softplus()):
        """Base Hamiltonian neural network. 

        Args:
            nstates (int): State dimension of the modeled system.
            noutputs (int, optional): Output dimension (dimension of the Hamiltonian function, which is a scalar). Defaults to 1.
            hidden_dim (int, optional): Number of hidden units per layer. Defaults to 100.
            act_1 (nn.Module, optional): First activation function. Defaults to PAU().
            act_2 (nn.Module, optional): Second activation function. Defaults to nn.Softplus().
        """
        super().__init__()
        self.nstates = nstates
        self.noutputs = 1
        self.hidden_dim = hidden_dim
        self.act_1 = act_1
        self.act_2 = act_2

        linear1 = nn.Linear(nstates, hidden_dim) 
        linear2 = nn.Linear(hidden_dim, hidden_dim)
        linear3 = nn.Linear(hidden_dim, noutputs)

        #Initializing weigths and biases by orthogonal weight initialization
        for lin in [linear1, linear2, linear3]:
            nn.init.orthogonal_(lin.weight) 

        self.model = nn.Sequential(
            linear1,
            self.act_1,
            linear2,
            self.act_2,
            linear3,
        )

    def forward(self, u: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            u (torch.Tensor, optional): Input state. Defaults to None.

        Returns:
            torch.Tensor: Learned Hamiltonian value in the input state u.
        """
        return self.model(u)
  


class HamiltonianNeuralNetwork(nn.Module):
    def __init__(self, nstates: int, S: Union[Sequence[float], np.ndarray, torch.Tensor], sys: object, initial_condition_sampler: Callable[..., Union[np.ndarray, torch.Tensor]],Hamiltonian_estimated: Optional[nn.Module] = None,**kwargs):
        """Full Hamiltonian neural network (HNN) framework for modeling physical systems.

        Args:
            nstates (int): State dimension of the modeled system.
            S (Union[Sequence[float], np.ndarray, torch.Tensor]): Skew-symmetric structure matrix.
            sys (object): Modeled Hamiltonian system. 
            initial_condition_sampler (Callable[..., Union[np.ndarray, torch.Tensor]]): Initial condition sampling function.
            Hamiltonian_estimated (Optional[nn.Module], optional): Base Hamiltonian neural network. Defaults to None.
        """
        super(HamiltonianNeuralNetwork,self).__init__()
        self.S = torch.tensor(S,dtype=torch.float32)
        self.Hamiltonian = None
        self.nstates = nstates
   
        if Hamiltonian_estimated is not None:
            self.Hamiltonian = Hamiltonian_estimated #HNN
        else:
            self.Hamiltonian = BaseHamiltonianNeuralNetwork(nstates = nstates, act_1 = Sin(), act_2 = nn.Softplus()) #HNN
        self.dH = self._dH_hamiltonian_est

        self.act1 = self.Hamiltonian.act_1
        self.act2 = self.Hamiltonian.act_2
        self.sys = sys

        if initial_condition_sampler is not None:
            self.initial_condition_sampler = initial_condition_sampler
    
    def _dH_hamiltonian_est(self, u: torch.Tensor) -> torch.Tensor:
        """Computes the gradient of the learned Hamiltonian w.r.t the state input.

        Args:
            u (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Gradient dH/du.
        """
        #u = u.requires_grad_()
        u = u.detach().requires_grad_()
        return torch.autograd.grad(
            self.Hamiltonian(u).sum(),
            u,
            retain_graph=self.training,
            create_graph=self.training,
        )[0]
    
    def _dH_hamiltonian_true(self, u: torch.Tensor) -> torch.Tensor:
        """Compute the true Hamiltonian gradient, detached from graph.

        Args:
            u (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Gradient of H (detached).
        """
        u = u.detach().requires_grad_()
        return torch.autograd.grad(
            self.Hamiltonian(u).sum(), u, retain_graph=False, create_graph=False
        )[0].detach()
    
    def u_dot(self, u: torch.Tensor) -> torch.Tensor:
        """ Compute time derivative of the state u using the learned Hamiltonian.

        Args:
            u (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Time derivative du/dt.
        """
        #S = (self.S).to(device=mps_device)
        return self.dH(u)@self.S.T
    
    def time_derivative_step(self, integrator: str, u_start: torch.Tensor, dt: torch.Tensor, u_end: Optional[torch.Tensor]=None, *args, **kwargs)-> torch.Tensor:
        """Estimates the time derivative of the state by numerical integraton (single step).

        Args:
            integrator (str): Name of numerical integrator.
            u_start (torch.Tensor): u_n.
            dt (torch.Tensor): Step length h.
            u_end (Optional[torch.Tensor], optional): u_n+1. Defaults to None.

        Returns:
            torch.Tensor: du/dt estimate by numerical integrator (single step).
        """
        if integrator == "RK4":
            return RK4_time_derivative(self.u_dot, u_start, dt)
        elif integrator == "midpoint":
            return explicit_midpoint_time_derivative(self.u_dot, u_start, dt, *args, **kwargs)
        elif integrator == "symplectic midpoint":
            return symplectic_midpoint_time_derivative(self.u_dot, u_start, dt, u_end, *args, **kwargs)
        
        elif integrator == "symplectic euler":
            return symplectic_euler(self.u_dot, u_start, dt)
        




    def simulate_trajectory(self,integrator: str,t_sample: Union[np.ndarray, torch.Tensor],dt: Union[np.ndarray, torch.Tensor],u0: Optional[Union[np.ndarray, torch.Tensor]] =None)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts a single trajectory by numerical integration utilizing the gradient of the learned Hamiltonian. 

        Args:
            integrator (str): Name of numerical integrator.
            t_sample (Union[np.ndarray, torch.Tensor]): Time array.
            dt (Union[np.ndarray, torch.Tensor]): Step length h.
            u0 (Optional[Union[np.ndarray, torch.Tensor]], optional): Initial state. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        if u0 is None:
            u0 = self.initial_condition_sampler()
 
        u0 = torch.tensor(u0,dtype = torch.float32)
        u0 = u0.reshape(1,u0.shape[-1])

        t_sample = torch.tensor(t_sample,dtype = torch.float32)
        t_shape = t_sample.shape[-1]

        #Initializing solution 
        u = torch.zeros([t_sample.shape[-1],self.nstates])
        dudt = torch.zeros_like(u)

        #Setting initial conditions
        u[0, :] = u0

        for i in range(t_shape-1):
            dudt[i,:] = self.time_derivative_step(integrator=integrator,u_start = u[i : i + 1, :], dt = dt)
            u[i+1,:] = u[i,:] + dt*dudt[i,:]

        return u,dudt,u0
    
    def generate_trajectories(self,ntrajectories: int, t_sample: Union[np.ndarray, torch.Tensor],integrator:str = "midpoint",u0s: Optional[Union[np.ndarray, torch.Tensor]] =None)-> tuple[torch.Tensor, torch.Tensor]:
        """Generates multiple trajectories  by numerical integration utilizing the gradient of the learned Hamiltonian. 

        Args:
            ntrajectories (int): Number of trajectories.
            t_sample (Union[np.ndarray, torch.Tensor]): Time array.
            integrator (str, optional): _description_. Defaults to "midpoint".
            u0s (Optional[Union[np.ndarray, torch.Tensor]], optional): Initial state. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: _description_
        """
        if u0s.any() == None:
            u0s = self.initial_condition_sampler(ntrajectories)
        
        #Reshaping
        u0s = torch.tensor(u0s,dtype = torch.float32)
        u0s = u0s.reshape(ntrajectories, self.nstates)
        t_sample = torch.tensor(t_sample,dtype = torch.float32)
        
        if len(t_sample.shape) == 1:
                #Reshaping time
                t_sample = np.tile(t_sample, (ntrajectories, 1))

        dt = t_sample[0, 1] - t_sample[0, 0]
        traj_length = t_sample.shape[-1]

        #Initializng u and setting initial conditions
        u = torch.zeros([ntrajectories, traj_length, self.nstates])
        u[:,0,:] = u0s

        for i in range(ntrajectories):
            u[i] = self.simulate_trajectory(integrator = integrator,t_sample = t_sample, u0 = u0s[i],dt=dt)[0]
   
        return u, t_sample