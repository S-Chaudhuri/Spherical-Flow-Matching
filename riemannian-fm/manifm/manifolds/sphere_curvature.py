import math
from geoopt.manifolds.base import Manifold
import numpy as np
import torch
from typing import Optional, Union, Tuple
import torch.linalg as linalg
from torch import broadcast_shapes

EPS = {torch.float32: 1e-4, torch.float64: 1e-7}

"""
Changed form Geoopt's SphereExact to SphereCurvature, which allows for arbitrary positive curvature (instead of just 1).
"""

class SphereCurvature(Manifold):
    # __doc__ = r"""{}

    # See Also
    # --------
    # :class:`SphereExact`
    # """.format(
    #     _sphere_doc
    # )
    ndim = 1
    name = "SphereCurvature"
    reversible = False

    def __init__(
        self, intersection: torch.Tensor = None, complement: torch.Tensor = None, c: float = 1.0
    ):
        super().__init__()
        if intersection is not None and complement is not None:
            raise TypeError(
                "Can't initialize with both intersection and compliment arguments, please specify only one"
            )
        elif intersection is not None:
            self._configure_manifold_intersection(intersection)
        elif complement is not None:
            self._configure_manifold_complement(complement)
        else:
            self._configure_manifold_no_constraints()
        if (
            self.projector is not None
            and (linalg.matrix_rank(self.projector) == 1).any()
        ):
            raise ValueError(
                "Manifold only consists of isolated points when "
                "subspace is 1-dimensional."
            )
        self.c = c
        self.radius = 1.0 / np.sqrt(self.c)

    def _check_shape(
        self, shape: Tuple[int], name: str
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok, reason = super()._check_shape(shape, name)
        if ok and self.projector is not None:
            ok = len(shape) < (self.projector.dim() - 1)
            if not ok:
                reason = "`{}` should have at least {} dimensions but has {}".format(
                    name, self.projector.dim() - 1, len(shape)
                )
            ok = shape[-1] == self.projector.shape[-2]
            if not ok:
                reason = "The [-2] shape of `span` does not match `{}`: {}, {}".format(
                    name, shape[-1], self.projector.shape[-1]
                )
        elif ok:
            ok = shape[-1] != 1
            if not ok:
                reason = (
                    "Manifold only consists of isolated points when "
                    "subspace is 1-dimensional."
                )
        return ok, reason

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Tuple[bool, Optional[str]]:
        norm = x.norm(dim=-1)
        # checking based on radius
        target = norm.new((1,)).fill_(self.radius)
        ok = torch.allclose(norm, target,atol=atol, rtol=rtol)
        # ok = torch.allclose(norm, norm.new((1,)).fill_(1), atol=atol, rtol=rtol)
        if not ok:
            return False, "`norm(x) != {}` with atol={}, rtol={}".format(self.radius, atol, rtol)
        ok = torch.allclose(self._project_on_subspace(x), x, atol=atol, rtol=rtol)
        if not ok:
            return (
                False,
                "`x` is not in the subspace of the manifold with atol={}, rtol={}".format(
                    atol, rtol
                ),
            )
        return True, None

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Tuple[bool, Optional[str]]:
        inner = self.inner(x, x, u, keepdim=True)
        ok = torch.allclose(inner, inner.new_zeros((1,)), atol=atol, rtol=rtol)
        if not ok:
            return False, "`<x, u> != 0` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        if v is None:
            v = u
        inner = (u * v).sum(-1, keepdim=keepdim)
        target_shape = broadcast_shapes(x.shape[:-1] + (1,) * keepdim, inner.shape)
        return inner.expand(target_shape)

    def pairwise_inner(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...id,...jd->...ij", u, v)

    # Changed by adding projection on the subspace, and scaling by the radius
    def projx(self, x: torch.Tensor) -> torch.Tensor:
        x = self._project_on_subspace(x) 
        # clamp norm to avoid numerical issues
        return x / x.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype]) * self.radius

    # We scale by radius since tangent space is defined relative to point with norm R
    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # proj_x(u) = u - <x, u> / R^2 * x
        u = u - (x * u).sum(dim=-1, keepdim=True) / (self.radius ** 2) * x 
        return self._project_on_subspace(u)

    # Changed by adding scaling by the radius, and using the curvature in the trigonometric functions
    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        norm_u = u.norm(dim=-1, keepdim=True)
        # on a sphere, the geodesic angle is arc length / radius
        theta = norm_u / self.radius
        # exp = x * torch.cos(norm_u) + u * torch.sin(norm_u) / norm_u (with clamping to avoid numerical issues)
        exp = x * torch.cos(theta) + u * torch.sin(theta) * (
            self.radius / norm_u.clamp_min(EPS[u.dtype])
        )
        retr = self.projx(x + u)
        cond = norm_u > EPS[u.dtype]
        return torch.where(cond, exp, retr)

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.projx(x + u)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.proju(y, v)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        u = self.proju(x, y - x)
        dist = self.dist(x, y, keepdim=True)
        cond = dist.gt(EPS[x.dtype])
        result = torch.where(
            cond, u * dist / u.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype]), u
        )
        return result

    # Original Log map is correct. Radius correction for curvature
    # is  taken care of by the projx() and proju() methods.
    # def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    #     cos_theta = (x * y).sum(dim=-1, keepdim=True) / (self.radius ** 2)
    #     cos_theta = cos_theta.clamp(-1 + EPS[x.dtype], 1 - EPS[x.dtype])
    #     theta = torch.acos(cos_theta)
    #     sin_theta = torch.sin(theta)

    #     # direction = self.proju(x, y - cos_theta * x)
    #     direction = y - cos_theta * x
    #     coef = (self.radius *theta) / sin_theta.clamp_min(EPS[x.dtype])
    #     result =  direction * coef
    #     # handle x close to y (since north pole origin)
    #     small = theta < EPS[x.dtype]
    #     fallback = self.proju(x, y - x)
    #     return torch.where(small, fallback, result)
    
    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        inner = self.inner(x, x, y, keepdim=keepdim) / (self.radius ** 2)
        inner = inner.clamp(
            -1 + EPS[x.dtype], 1 - EPS[x.dtype]
        )
        #tighter clamp as origin is fixed (floating point errors) optional
        # inner = inner.clamp(-1 + 1e-7, 1 - 1e-7)
        return torch.acos(inner) * self.radius

    def cdist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # first divide, then clamp to set to range [-R^2, R^2], 
        inner = self.pairwise_inner(x, y) / (self.radius ** 2)
        inner = inner.clamp(-1 + EPS[x.dtype], 1 - EPS[x.dtype])
        # scale by radius since the distance is defined relative to points with norm R
        return torch.acos(inner) * self.radius

    egrad2rgrad = proju

    def _configure_manifold_complement(self, complement: torch.Tensor):
        Q, _ = linalg.qr(complement)
        P = -Q @ Q.transpose(-1, -2)
        P[..., torch.arange(P.shape[-2]), torch.arange(P.shape[-2])] += 1
        self.register_buffer("projector", P)

    def _configure_manifold_intersection(self, intersection: torch.Tensor):
        Q, _ = linalg.qr(intersection)
        self.register_buffer("projector", Q @ Q.transpose(-1, -2))

    def _configure_manifold_no_constraints(self):
        self.register_buffer("projector", None)

    def _project_on_subspace(self, x: torch.Tensor) -> torch.Tensor:
        if self.projector is not None:
            return x @ self.projector.transpose(-1, -2)
        else:
            return x

    def random_uniform(self, *size, dtype=None, device=None) -> torch.Tensor:
        """
        Uniform random measure on Sphere manifold.

        Parameters
        ----------
        size : shape
            the desired output shape
        dtype : torch.dtype
            desired dtype
        device : torch.device
            desired device

        Returns
        -------
        ManifoldTensor
            random point on Sphere manifold

        Notes
        -----
        In case of projector on the manifold, dtype and device are set automatically and shouldn't be provided.
        If you provide them, they are checked to match the projector device and dtype
        """
        self._assert_check_shape(size2shape(*size), "x")
        if self.projector is None:
            tens = torch.randn(*size, device=device, dtype=dtype)
        else:
            if device is not None and device != self.projector.device:
                raise ValueError(
                    "`device` does not match the projector `device`, set the `device` argument to None"
                )
            if dtype is not None and dtype != self.projector.dtype:
                raise ValueError(
                    "`dtype` does not match the projector `dtype`, set the `dtype` arguement to None"
                )
            tens = torch.randn(
                *size, device=self.projector.device, dtype=self.projector.dtype
            )
        return ManifoldTensor(self.projx(tens), manifold=self)

    random = random_uniform

    # mass can wrap around the sphere when std is large
    def wrapped_normal(self, dim, mean, std):
        z = torch.randn_like(mean)
        z = self.proju(mean, z)
        z = std * z
        z = self.proju(mean, z)
        return self.expmap(mean, z)
    
    def transp(self, x, y, v):
        denom = 1 + self.inner(x, x, y, keepdim=True)
        res = v - self.inner(x, y, v, keepdim=True) / denom * (x + y)
        cond = denom.gt(1e-3)
        return torch.where(cond, res, -v)

    def uniform_logprob(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.shape[-1]
        logprob_unit_sphere = math.lgamma(dim / 2) - (math.log(2) + (dim / 2) * math.log(math.pi))
        
        # Account for the radius: Area scales by R^(dim-1)
        logprob_scaled = logprob_unit_sphere - (dim - 1) * math.log(self.radius)
        return torch.full_like(x[..., 0], logprob_scaled)

    #def random_base(self, *args, **kwargs):
    #    return self.random_uniform(*args, **kwargs)

    def base_logprob(self, *args, **kwargs):
        return self.uniform_logprob(*args, **kwargs)
