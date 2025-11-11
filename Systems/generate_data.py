import numpy as np
import torch
from tqdm import tqdm

def generate_data_PDE(system, ntrajectories, t_sample, integrator, xspatial = None, true_derivatives=False, u0s=None, data_type=torch.float32):
    nstates = system.nstates
    traj_length = t_sample.shape[0]

    u = np.zeros((ntrajectories,traj_length,nstates))
    dudt = np.zeros_like(u)
    t = np.zeros((ntrajectories,traj_length))
    u0_ = np.zeros((ntrajectories, nstates))

    for i in tqdm(range(ntrajectories)):
        if u0s is not None:
            u0 = np.array(u0s[i])
            u[i], dudt[i], t[i],u0_[i] = system.sample_trajectory(t=t_sample,u0=u0,integrator=integrator)
        else:
            u[i], dudt[i], t[i],u0_[i] = system.sample_trajectory(t=t_sample,integrator=integrator)

    dt = torch.tensor([t[0, 1] - t[0, 0]], dtype=data_type)


    u_start = torch.tensor(u[:, :-1], dtype=data_type).reshape(-1, 1, nstates)
    u_end = torch.tensor(u[:, 1:], dtype=data_type).reshape(-1, 1, nstates)
    t_start = torch.tensor(t[:, :-1], dtype=data_type).reshape(-1, 1, 1)
    t_end = torch.tensor(t[:, 1:], dtype=data_type).reshape(-1, 1, 1)
    dt = dt * torch.ones_like(t_start, dtype=data_type)

    dudt_tensor = torch.tensor(dudt[:, :-1], dtype=data_type).reshape(-1, 1, nstates) if true_derivatives else (u_end - u_start) / dt[0, 0]

    if xspatial is None:
        xspatial = system.x
   
    xspatial = torch.tensor(
        np.repeat(xspatial.reshape(1, 1, -1), dudt_tensor.shape[0], axis=0), dtype=data_type
    ).reshape(dudt_tensor.shape[0], 1, -1)

    

    return (u_start, u_end, t_start, t_end, dt, xspatial), dudt_tensor, u, u0_
