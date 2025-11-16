import numpy as np
import numpy.linalg as la
from scipy.sparse import diags
from scipy.linalg import lu_factor, lu_solve
import autograd.numpy as np
from scipy.integrate import solve_ivp
from NeuralNetworks.numerical_integration import *

def create_sparse_tridiagonal(n, a = 1, b = -2, c = 1):
    diagonals = [np.full(n-1, a), np.full(n, b), np.full(n-1, c)]
    return diags(diagonals, offsets=[-1, 0, 1]).toarray()


def midpoint_method(u_start, u_end, t, f, Df, dt, M, tol=1e-12, max_iter=5):
    I = np.eye(M)
    F = lambda u_hat: 1/dt*(u_hat-u_start) - f((u_start+u_hat)/2, t+.5*dt)
    J = lambda u_hat: 1/dt*I - 1/2*Df((u_start+u_hat)/2, t+.5*dt)

    error = la.norm(F(u_end))
    j = 0
    while error > tol:
        u_end = u_end - la.solve(J(u_end),F(u_end))
        error = la.norm(F(u_end))
        j += 1
        if j > max_iter:
            break
    return u_end

class HeatEquation:
    def __init__(self,L= 1, N = 10, dt = 0.0025, T = 0.25, A = None,B = None, C = None, init_sampler = None, nstates = None, seed = 42, BC = True):
        self.N = N #M = N+1
        if nstates is None:
            nstates = N+1
        self.x = np.linspace(0, L, N + 1)
        self.dx = L / N
        self.N_t = int(T / dt)
        self.nstates = nstates
        self.rng = np.random.default_rng(seed)
        self.name_system = "HeatEquation"

        if A is None:
            tridiag = create_sparse_tridiagonal(n=N+1)
            self.A = 1/self.dx**2*tridiag
        if B is None:
            twodiag = create_sparse_tridiagonal(n=N+1, a = 0, b = -1, c = 1) #N+1 hos sintef med -50 nederst til venstre.
            self.B = 1/self.dx*twodiag
        if C is None:
            two_diag = create_sparse_tridiagonal(n=N+1, a = -1, b = 0, c = 1) 
            self.C = 0.5*1/self.dx*two_diag

        if BC == True:
            self.SetBoundaryConditions()

        if init_sampler is not None:
            self._initial_condition_sampler = init_sampler

        self.skewsymmetric_matrix_flat = 0.5 / self.dx * np.array([[[-1, 0, 1]]])

    def SetBoundaryConditions(self):
        A, B, C = self.A.copy(), self.B.copy(), self.C.copy()

        #Boundary conditions
        A[0,-1] = 1/self.dx**2
        A[-1,0] = 1/self.dx**2
        
        B[-1,0] = 1/self.dx
    
        C[-1,0] = 0.5/self.dx
        C[0,-1] = -0.5/self.dx

        self.A = A
        self.B = B
        self.C = C
    
    def V(self,u):
        #dV(u) = u_xx -> V(u) = -0.5 int(u_x^2) dx ??
        return np.sum(-0.5 * (np.matmul(self.B, u.T) ** 2).T, axis=1)

    def dV(self,u):
        #N = -1, H(u) = 0.5 int(u_x^2) dx
        #Central diff: u_xx = Au -> dH(u) = - Au
        #u_xx = dudt = NdH(u) =- dV = - Au
        #-> dV = Au 
        #return  u @ self.A
        return  u @ self.A

    def ddV(self,u):
        #dV = Au -> ddV = A
        return self.A
    
    def _initial_condition_sampler(self):
        M = self.N+1
        x = self.x
        P = (x[-1] - x[0]) * M / (M - 1)
     

        d1, d2 = np.random.uniform(0.3, 3, 2)
        c1, c2 = np.random.uniform(0.5, 1.5, 2)
        k1 = np.random.uniform(0.5, 3.0, 1)
        k2 = np.random.uniform(10.0, 20.0, 1)
        n1 = np.random.uniform(20.0, 40.0, 1)
        n2 = np.random.uniform(0.05, 0.15, 1)
        u0 = 0
        u0 += (
            np.random.uniform(-5.0, 5.0, 1)
            - c1 * np.tanh(n1 * (x - d1))
            + c1 * np.tanh(n1 * (x - P + d1))
        )
        u0 += -c2 * np.tanh(n1 * (x - d2)) + c2 * np.tanh(n1 * (x - P + d2))
        u0 += n2 * np.sin(k1 * np.pi * x) ** 2 * np.sin(k2 * np.pi * x)
        return u0



    def u_dot(self, u, t, xspatial = None):
        """

        Computes du/dt = -grad[V(u)], i.e the time derivative of the state u
        The right hand side of the pseudo-Hamiltonian formulation.

        """
         
        return self.dV(u)  



    def u_dot_jacobian(self, u, t):
        """
        Computes the Jacobian of the right hand side of the pseudo-
        Hamiltonian equation.

        """
        return self.ddV(u)


    def sample_trajectory(self, t, u0=None, noise_std=0, add_noise=False, integrator=None):
        if u0 is None:
            u0 = self._initial_condition_sampler()

        if integrator == "midpoint":
            return self.sample_trajectory_midpoint(t, u0=u0, noise_std=0, add_noise=False)

        elif integrator == "scipy":
            def u_dot(t, u): 
                g = self.u_dot(u.reshape(1, u.shape[-1]), np.array(t).reshape((1, 1)))
                return g
            
            out_ivp = solve_ivp(
                fun=u_dot, t_span=(t[0], t[-1]), y0=u0, t_eval=t, rtol=1e-10
            )
            u, t = out_ivp["y"].T, out_ivp["t"].T
        else:
            dt = float(t[1] - t[0])                 #Assumes uniform time grid             
            I = np.eye(self.N+1)

            LHS = I - 0.5 * dt * self.A
            RHS = I + 0.5 * dt * self.A
            lu, piv = lu_factor(LHS)

            u = np.zeros((len(t), self.N + 1))
            u[0] = u0

            for n in range(len(t) - 1):
                u[n+1] = lu_solve((lu, piv), RHS @ u[n])
            t = np.asarray(t)

        dudt = self.u_dot(u, t)

        if add_noise:
            u += self.rng.normal(size=u.shape) * noise_std
            dudt += self.rng.normal(size=dudt.shape) * noise_std

        return u, dudt, t, u0
    

    def sample_trajectory_midpoint(self, t, u0=None, noise_std=0, add_noise=False):
        if u0 is None:
            u0 = self._initial_condition_sampler()
        M = u0.shape[-1]

        u = np.zeros([t.shape[0], u0.shape[-1]])
        dudt = np.zeros_like(u)
        u[0, :] = u0
       
        f = lambda u, t: self.u_dot(u, t)
        Df = lambda u, t: self.u_dot_jacobian(u, t)

        for i, t_step in enumerate(t[:-1]):
            dt = t[i + 1] - t[i]
            dudt[i, :] = f(u[i, :], t[i])
            u[i + 1, :] = midpoint_method(u[i, :], u[i, :], t[i], f, Df, dt, M, 1e-12, 5)

        #Add noise:
        if add_noise:
            u += self.rng.normal(size=u.shape) * noise_std
            dudt += self.rng.normal(size=dudt.shape) * noise_std

        return u, dudt, t, u0
