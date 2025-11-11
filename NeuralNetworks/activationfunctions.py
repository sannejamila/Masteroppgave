import numpy as np
import torch
import torch.nn as nn

class Sin(nn.Module):
    @staticmethod
    def forward(u: torch.Tensor) -> torch.Tensor:
        """Sin activation function.

        Args:
            u (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with sin applied.
        """
        return torch.sin(u)
    
class ParametricSin(nn.Module):
    """
    f(u) = alpha * sin(beta * u + phi)
    """
    def __init__(self, init_alpha: float = 1.0, init_beta: float = 1.0, init_phi: float = 0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))
        self.phi = nn.Parameter(torch.tensor(init_phi))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.sin(self.beta * u + self.phi)

class ParametricTanh(nn.Module):
    """
    f(u) = alpha * tanh(beta * u)
    """
    def __init__(self, init_alpha: float = 1.0, init_beta: float = 1.0, init_phi: float = 0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))
        self.phi = nn.Parameter(torch.tensor(init_phi))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.tanh(self.beta * u + self.phi)

def numerical_pade_coeffs(f, m: int , n: int, x_min: float=-2.0, x_max: float=2.0, num_points: int=500):
    """Computes the coefficients of the Pade approximation of an activation function f using least squares.

    Args:
        f (callable): The activation function to approximate.
        m (int): Order of the numerator polynomial.
        n (int): Order of the denominator polynomial.
        x_min (float, optional): Domain start point. Defaults to -2.0.
        x_max (float, optional): Domain end point. Defaults to 2.0.
        num_points (int, optional): Mesh size of domain. Defaults to 500.

    Returns:
        a: Coefficients of the numerator polynomial.
        b: Coefficients of the denominator polynomial.
    """
    x_t = torch.linspace(x_min, x_max, num_points, dtype=torch.double)
    y_t = f(x_t)
    x_np = x_t.detach().cpu().numpy()
    y_np = y_t.detach().cpu().numpy()

    #Build least-squares system
    X_num = np.vstack([x_np**i for i in range(m + 1)]).T
    X_den = np.vstack([-(y_np * (x_np**i)) for i in range(1, n + 1)]).T
    X = np.hstack([X_num, X_den])

    coeffs, *_ = np.linalg.lstsq(X, y_np, rcond=None)
    a = coeffs[:m + 1]
    b_rest = coeffs[m + 1:]
    b = np.concatenate(([1.0], b_rest))
    return a, b


    
class PAU(nn.Module):
    def __init__(self, n_numerator: int = 5, n_denominator: int = 4, init_func = None):
        """Constructs a rational approximation activation using Pade approximation.

        Args:
            n_numerator (int): Degree of numerator polynomial. Default is 5.
            n_denominator (int): Degree of denominator polynomial. Default is 4.
            init_func (callable, optional): Function to initialize the PAU. If None, random initialization is used. Defaults to None.
        """
        super(PAU, self).__init__()
        self.n_numerator = n_numerator
        self.n_denominator = n_denominator
        if init_func == None:
            self.numerator = nn.Parameter(torch.randn(n_numerator + 1))  #i = 0,...m
            self.denominator = nn.Parameter(torch.randn(n_denominator)) #i = 1,...n
        else:
            self.a, self.b = numerical_pade_coeffs(init_func, n_numerator, n_denominator)
            self.numerator = nn.Parameter(torch.tensor(self.a))  # i = 0,...,m
            self.denominator = nn.Parameter(torch.tensor(self.b[1:]))  # i = 1,...,n

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Applies the Pade activation unit (PAU). 

        Args:
            u (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with PAU applied.
        """
        num = sum(self.numerator[i] * u**i for i in range(self.n_numerator + 1))
        den = 1 + abs(sum(self.denominator[i] * u**(i+1) for i in range(self.n_denominator)))
        return num / den
    





class PLU(nn.Module):
    def __init__(self, shape=(1,), init_alpha=1.0, init_beta=1.0, init_rho_alpha=5.0, init_rho_beta=0.15):
        """
        The Periodic Linear Unit (PLU) is composed of a scaled linear sum of the sine function and x with α and β reparameterization, 
        as described in the paper "From Taylor Series to Fourier Synthesis: The Periodic Linear Unit".

        PLU(u, α, ρ_α, β, ρ_β) = x + (β_eff / (1 + |β_eff|)) * sin(|α_eff| * x)
        
        Where the effective parameters, α_eff and β_eff, are reparameterized using learnable repulsion terms, ρ_α and ρ_β as follows:

        α_eff = α + ρ_α / α
        β_eff = β + ρ_β / β

        The `x` component serves as a residual path. By including it, the function is identity-like at its core, 
        allowing gradients can flow through it easily just like a residual connection. It prevents the vanishing gradient problem and makes training much more stable.

        The `sin(x)` component then provides periodic non-linearity.
        
        Args:
            init_alpha (float):
                Period multiplier
            init_beta (float): 
                Signed, soft-bounded strength of periodic non-linearity
            init_rho_alpha (float):
                Repulsion term for keeping α away from 0.0, keeping α_eff >= sqrt(ρ_α)
            init_rho_beta (float):
                Repulsion term for keeping β away from 0.0, keeping β_eff >= sqrt(ρ_β)
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.full(shape, init_alpha))
        self.beta = nn.Parameter(torch.full(shape, init_beta))
        self.rho_alpha = nn.Parameter(torch.full(shape, init_rho_alpha))
        self.rho_beta = nn.Parameter(torch.full(shape, init_rho_beta))
    
    def forward(self, x):
        #repulsive reparameterization
        alpha_eff = self.alpha + self.rho_alpha / self.alpha
        beta_eff = self.beta + self.rho_beta / self.beta
        return x + (beta_eff / (1.0 + torch.abs(beta_eff))) * torch.sin(torch.abs(alpha_eff) * x)
    


class PLUExponential(nn.Module):
    def __init__(self, shape=(1,), init_alpha=0.0, init_beta=0.0):
        """
        An alternative formulation of the Periodic Linear Unit using e^x reparameterization in place of the original RR formulation.

        We start at a neutral point, where α=0.0 (α_eff=1.0) and β=0.0 (β_eff=1.0).
        As either α or β gets pushed above 0.0, the sine-wave begins oscillating faster and contributes more to the sum respectively.
        Since changing α to be higher, and thus increasing the oscillation, has a larger effect 
        on the final output and loss when compared to a change at the lower end, a "larger leap of faith" 
        is sometimes necessary to cross any hills in the loss plane to reach a better minimum.
        Since e^x has a higher and higher derivative above 0.0, this leap of faith is mathematically provided, as a small dα leads to a larger and larger dα_eff.
        As either α or β gets pushed below 0.0, the sine-wave begins to oscillate slower and slower and contributes less to the sum respectively.
        Since both of those lead to a collapse to linearity, we want to disincentivize the optimizer from taking a step in that direction more and more as α veers 
        closer to negative infinity.
        Since e^x has a lower and lower derivative below 0.0, each further dα step taken in that direction leads to a smaller and smaller dα_eff, thereby making each push 
        towards zero have less impact on the final loss, achieving our desired outcome.
        By using e^x reparameterization, we lose control over a specific lower bound for frequency and amplitude on the sine-wave component, but the benefit of 
        being able to have the model find a pretty good local minimum on its own without hyperparameter searching is well worth the trade-off in a variety of use-cases. 
        This is especially useful when using a separate learnable PLU activation for each hidden neuron, as hyperparameter searching is prohibitively expensive in that situation.
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.full(shape, init_alpha))
        self.beta = nn.Parameter(torch.full(shape, init_beta))
    
    def forward(self, x):
        # exponential reparameterization
        alpha_eff = torch.exp(self.alpha)
        beta_eff = torch.exp(self.beta)
        return x + beta_eff * torch.sin(alpha_eff * x)