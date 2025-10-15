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
    

def numerical_pade_coeffs(f, m, n, x_min=-2.0, x_max=2.0, num_points=500):
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