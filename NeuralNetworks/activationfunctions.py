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
    
class PAU(nn.Module):
    def __init__(self, n_numerator: int = 5, n_denominator: int = 4):
        """Constructs a rational approximation activation using Pade approximation.

        Args:
            n_numerator (int): Degree of numerator polynomial. Default is 5.
            n_denominator (int): Degree of denominator polynomial. Default is 4.
        """
        super(PAU, self).__init__()
        self.n_numerator = n_numerator
        self.n_denominator = n_denominator
        self.numerator = nn.Parameter(torch.randn(n_numerator + 1))  #i = 0,...m
        self.denominator = nn.Parameter(torch.randn(n_denominator)) #i = 1,...n
        

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