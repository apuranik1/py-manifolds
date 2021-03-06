import abc
from dataclasses import dataclass
from typing import Generic, Optional, Sequence, Tuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np


# P = generic Point type. Must not break differentiability.
P = TypeVar("P")


class OutOfDomain(Exception):
    """Indicates that a value is outside the domain of a map"""


class Chart(Generic[P], metaclass=abc.ABCMeta):
    """A single chart on a manifold.

    Determines a diffeomorphism from coordinates to points.
    """

    @abc.abstractmethod
    def coords_to_point(self, coords: np.ndarray) -> P:
        """Conversion from coordinates to a point in the domain.

        Raises OutOfDomain if the coords are not mapped to a valid point.
        """

    @abc.abstractmethod
    def point_to_coords(self, point: P) -> np.ndarray:
        """Convert a point to coordinates in this Chart.

        Raises OutOfDomain if the point is not in the domain of the chart.
        """


class Manifold(Generic[P], metaclass=abc.ABCMeta):
    """A smooth manifold equipped with an atlas"""

    @abc.abstractmethod
    def preferred_chart(self, point: P) -> Chart[P]:
        """Select some chart that "best" covers the point in terms of stability"""

    @abc.abstractmethod
    def charts(self, point: P) -> Sequence[Chart[P]]:
        """Enumerate all charts that cover the given point"""


@dataclass(frozen=True)
class Tangent(Generic[P]):
    """An element of the tangent bundle on a manifold in some chart"""
    chart: Chart[P]
    p_coords: np.ndarray
    v_coords: np.ndarray

    def to_chart(self, chart: Chart[P]):
        def convert_coords(c):
            return chart.point_to_coords(self.chart.coords_to_point(c))

        image, tangent = jax.jvp(convert_coords, [self.p_coords], [self.v_coords])
        return Tangent(chart, image, tangent)


@dataclass(frozen=True)
class Cotangent(Generic[P]):
    """An element of the cotangent bundle on a manifold in some chart"""
    chart: Chart[P]
    p_coords: np.ndarray
    v_coords: np.ndarray

    def to_chart(self, chart: Chart[P]):
        # need to pull back along inverse map
        image = chart.point_to_coords(self.chart.coords_to_point(self.p_coords))

        def convert_coords(c):
            return self.chart.point_to_coords(chart.coords_to_point(c))

        _, pullback_fun = jax.vjp(convert_coords, image)
        cotangent = pullback_fun(self.v_coords)
        return Cotangent(chart, image, cotangent)


@dataclass(frozen=True)
class Tensor(Generic[P]):
    """An element of a tensor bundle on a manifold in some chart.
    The contravariant dimensions must come before the covariant ones.
    """
    chart: Chart[P]
    p_coords: np.ndarray
    t_coords: np.ndarray
    n_contra: int

    @property
    def n_cov(self) -> int:
        return len(self.t_coords.shape) - self.n_contra

    @property
    def n_indices(self) -> int:
        return len(self.t_coords.shape)

    def tensorprod(self, other: "Tensor[P]") -> "Tensor[P]":
        """Tensor product of two tensors at the same point in the same chart.
        Equality of points and charts is not checked.
        """
        unordered_tensor = jnp.tensordot(self.t_coords, other.t_coords, 0)
        axis_order = [
            *range(0, self.n_contra),
            *range(self.n_indices, self.n_indices + other.n_contra),
            *range(self.n_contra, self.n_indices),
            *range(self.n_indices + other.n_contra, self.n_indices + other.n_indices)
        ]
        ordered = jnp.transpose(unordered_tensor, axis_order)
        return Tensor(self.chart, self.p_coords, ordered, self.n_contra + other.n_contra)

    def to_chart(self, chart: Chart[P]):
        def convert_forward(c):
            return chart.point_to_coords(self.chart.coords_to_point(c))

        def convert_backward(c):
            return self.chart.point_to_coords(chart.coords_to_point(c))

        image = convert_forward(self.p_coords)
        jacobian_backward = jax.jacfwd(convert_backward)(image)
        # at every step, we contract the first index of tensor
        # the indices we're done transforming are appended, so they end in the right order
        transformed_t = self.t_coords
        for _ in range(self.n_contra):
            # transform contravariant index by pulling back
            # in this case right multiplication is what we wanted anyway
            transformed_t = jnp.tensordot(transformed_t, jacobian_backward, axes=([0], [0]))

        jacobian_forward = jax.jacfwd(convert_forward)(self.p_coords)
        for _ in range(self.n_cov):
            # we actually want left multiplication, so contract axis 1 of jacobian
            transformed_t = jnp.tensordot(transformed_t, jacobian_forward, axes=([0], [1]))

        return Tensor(chart, image, transformed_t, self.n_contra)


class Metric(Generic[P]):
    """A metric on a manifold at some point in some chart"""
    chart: Chart[P]
    p_coords: np.ndarray
    matrix: np.ndarray


class RiemannianManifold(Manifold[P]):
    """A Riemannian manifold equipped with a metric"""

    @abc.abstractmethod
    def metric(self, point: P) -> Tuple[np.ndarray, Chart[P]]:
        """Compute the metric as a matrix in the specified chart at point"""

    def inv_metric(self, point: P) -> Tuple[np.ndarray, Chart[P]]:
        """Compute the inverse metric as a matrix in the specified chart at point.

        The inverse metric is an operations from cotangent x cotangent -> float
        """
        mat, chart = self.metric(point)
        return np.linalg.inv(mat), chart
