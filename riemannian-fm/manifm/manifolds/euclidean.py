"""Copyright (c) Meta Platforms, Inc. and affiliates."""
import math
import torch
from geoopt.manifolds import Euclidean as geoopt_Euclidean
import torch.distributions as D



class Euclidean(geoopt_Euclidean):


    def metric_normalized(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Euclidean space has a constant metric, so no normalization is needed."""
        return u 

    def random_base(self,batch_size, dim, std = 1.0):
        """Sample points from the Normal distribution centered in the origin.
        
        Args:
            batch_size (int): Number of points to sample
            dim (int): Dimension of the ball            
        Returns:
            torch.Tensor: sampled points in the ball [batch_size, dim]
        """
        # Gaussian sampling 
        mean = torch.zeros(dim, device="cuda")
        std = torch.tensor(std, device="cuda")
        #std = 0.3

        samples = []
        for _ in range(batch_size):
            x = self.random_normal(dim, mean=mean, std=std)
            samples.append(x)

        return torch.stack(samples, dim=0)
        

    def base_logprob(self, x, std=1.0):
        dist = D.Normal(
            loc=torch.zeros_like(x),
            scale=std
        )
        return dist.log_prob(x).sum(dim=-1)
    
    def random_gaussian_ring(self, dim, mean, std, radius):

        # gaussian_sample = mean + torch.randn(dim, device=mean.device) * std

        # direction = torch.randn(dim, device=mean.device)
        # direction = direction / (direction.norm() + 1e-8)

        # sample = gaussian_sample + radius * direction

        # return sample
        device = mean.device

        z = torch.randn(dim, device=device)
        u = z / z.norm()
        r = torch.normal(radius, std, size=(), device=device)

        return mean + r * u


    

