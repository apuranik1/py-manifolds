"""Types and general transformations on smooth manifolds.

All tangent spaces are expressed in the coordinate basis of a chart. Which chart will be
clear from context.

As many functions as possible are jit-friendly.
"""

import abc
from dataclasses import dataclass
from typing import Callable, Generic, Sequence, Tuple, TypeVar

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
        """Select some chart that "best" covers the point in terms of stability.

        If possible, this method should be JIT-able.
        """

    @abc.abstractmethod
    def charts(self, point: P) -> Sequence[Chart[P]]:
        """Enumerate all charts that cover the given point.

        Not generally JIT-able.
        """


@dataclass(frozen=True)
class ChartPoint(Generic[P]):
    """A point in coordinates in a chart"""

    coords: np.ndarray
    chart: Chart[P]

    @staticmethod
    def of_point(point: P, chart: Chart[P]) -> "ChartPoint[P]":
        return ChartPoint(chart.point_to_coords(point), chart)


P_ = TypeVar("P_")


T = TypeVar("T")
T_ = TypeVar("T_")


@dataclass(frozen=True)
class Diffeomorphism(Generic[P, P_]):
    """A diffeomorphism between manifolds, expressed as a smooth map"""

    forward: Callable[[P], P_]
    backward: Callable[[P_], P]

    # mypy breaks if we try to reuse P and P_ as typevars here

    def inverse(self: "Diffeomorphism[T, T_]") -> "Diffeomorphism[T_, T]":
        result = Diffeomorphism(forward=self.backward, backward=self.forward)
        return result

    @staticmethod
    def identity() -> "Diffeomorphism[T, T]":
        def ident(x):
            return x

        return Diffeomorphism(ident, ident)


@dataclass(frozen=True)
class Tensor(Generic[P]):
    """An element of a tensor bundle on a manifold in some chart.
    In the internal representation, contravariant dimensions come before the covariant
    ones, but in the interface, both are separately indexed starting at 0.
    """

    point: ChartPoint[P]
    t_coords: np.ndarray
    n_contra: int

    @property
    def n_cov(self) -> int:
        return len(self.t_coords.shape) - self.n_contra

    @property
    def n_indices(self) -> int:
        return len(self.t_coords.shape)

    def tensor_prod(self, other: "Tensor[P]") -> "Tensor[P]":
        """Tensor product of two tensors at the same ChartPoint.
        Equality of points and charts is not checked.
        """
        unordered_tensor = jnp.tensordot(self.t_coords, other.t_coords, 0)
        axis_order = [
            *range(0, self.n_contra),
            *range(self.n_indices, self.n_indices + other.n_contra),
            *range(self.n_contra, self.n_indices),
            *range(self.n_indices + other.n_contra, self.n_indices + other.n_indices),
        ]
        ordered = jnp.transpose(unordered_tensor, axis_order)
        return Tensor(self.point, ordered, self.n_contra + other.n_contra)

    def trace(self, contra_index: int, cov_index: int) -> "Tensor[P]":
        """Contract the contravariant [contra_index] with the covariant [cov_index].

        Requires 0 <= contra_index < n_contra, and 0 <= cov_index < n_cov
        """
        if contra_index < 0 or contra_index > self.n_contra:
            raise ValueError(f"contra_index out of bounds: {contra_index}")
        if cov_index < 0 or cov_index > self.n_cov:
            raise ValueError(f"cov_index out of bounds: {cov_index}")
        coords = jnp.trace(self.t_coords, contra_index, self.n_contra + cov_index)
        return Tensor(self.point, coords, self.n_contra - 1)

    def contract(self, contra_index: int, other: "Tensor[P]", cov_index: int):
        """Contract a contravariant index of self with a covariant index of other"""
        # it would be easier to implement this as tensor_prod then trace
        if contra_index < 0 or contra_index > self.n_contra:
            raise ValueError(f"contra_index out of bounds: {contra_index}")
        if cov_index < 0 or cov_index > other.n_cov:
            raise ValueError(f"cov_index out of bounds: {cov_index}")

        unordered = jnp.tensordot(
            self.t_coords,
            other.t_coords,
            ([contra_index], other.n_contra + cov_index),
        )
        # currently ordered self:contra, self:cov, other:contra, other:cov
        # want self:contra, other:contra, self:cov, other:cov
        # 1 missing each from self:contra, other:cov
        axis_order = [
            *range(self.n_contra - 1),
            *range(self.n_indices - 1, self.n_indices - 1 + other.n_contra),
            *range(self.n_contra - 1, self.n_indices - 1),
            *range(
                self.n_indices - 1 + other.n_contra,
                self.n_indices + other.n_indices - 2,
            ),
        ]
        ordered = jnp.transpose(unordered, axis_order)
        return Tensor(self.point, ordered, self.n_contra + other.n_contra - 1)

    def diffeo_pushforward(
        self, diffeo: Diffeomorphism[P, P_], chart: Chart[P_]
    ) -> "Tensor[P_]":
        """Compute the pushforward of this tensor along a diffeomorphism

        This isn't a true pushforward; it also requires pulling back contravariant
        indices along the inverse. It's more like a change of coordinates.
        """

        def coord_map_forward(c: np.ndarray) -> np.ndarray:
            return chart.point_to_coords(
                diffeo.forward(self.point.chart.coords_to_point(c))
            )

        def coord_map_backward(c: np.ndarray) -> np.ndarray:
            return self.point.chart.point_to_coords(
                diffeo.backward(chart.coords_to_point(c))
            )

        image = coord_map_forward(self.point.coords)
        jacobian_backward = jax.jacfwd(coord_map_backward)(image)
        # at every step, we contract the first index of tensor
        # the indices we're done transforming are appended, so they end in the right order
        transformed_t = self.t_coords
        for _ in range(self.n_contra):
            # transform contravariant index by pulling back
            # in this case right multiplication is what we wanted anyway
            transformed_t = jnp.tensordot(
                transformed_t, jacobian_backward, axes=([0], [0])
            )

        jacobian_forward = jax.jacfwd(coord_map_forward)(self.point.coords)
        for _ in range(self.n_cov):
            # we actually want left multiplication, so contract axis 1 of jacobian
            transformed_t = jnp.tensordot(
                transformed_t, jacobian_forward, axes=([0], [1])
            )

        return Tensor(ChartPoint(image, chart), transformed_t, self.n_contra)

    def to_chart(self, chart: Chart[P]):
        return self.diffeo_pushforward(Diffeomorphism.identity(), chart)


class CovariantTensor(Tensor[P]):
    def __init__(self, point: ChartPoint[P], t_coords: np.ndarray):
        super().__init__(point, t_coords, 0)

    def pushforward(
        self, morphism: Callable[[P], P_], chart: Chart[P_]
    ) -> "CovariantTensor[P_]":
        # see Tensor.diffeo_pushforward for explanation of details
        def coord_map(c: np.ndarray) -> np.ndarray:
            return chart.point_to_coords(morphism(self.point.chart.coords_to_point(c)))

        image = coord_map(self.point.coords)
        jacobian = jax.jacfwd(coord_map)(self.point.coords)
        transformed_t = self.t_coords
        for _ in range(self.n_cov):
            # we actually want left multiplication, so contract axis 1 of jacobian
            transformed_t = jnp.tensordot(transformed_t, jacobian, axes=([0], [1]))

        return CovariantTensor(ChartPoint(image, chart), transformed_t)

    def covariant_prod(self, other: "CovariantTensor[P]") -> "CovariantTensor[P]":
        tensor = self.tensor_prod(other)
        return CovariantTensor(tensor.point, tensor.t_coords)


class ContravariantTensor(Tensor[P]):
    def __init__(self, point: ChartPoint[P], t_coords: np.ndarray):
        super().__init__(point, t_coords, len(t_coords.shape))

    def pullback(
        self, morphism: Callable[[P_], P], preimage: ChartPoint[P_]
    ) -> "ContravariantTensor[P_]":
        """Compute a pullback of a covariant tensor along [morphism].

        Requires a preimage of this point to be provided. Correctness of the preimage
        is not checked.
        """
        # see Tensor.diffeo_pushforward for explanation of details

        def coord_map(c: np.ndarray) -> np.ndarray:
            return self.point.chart.point_to_coords(
                morphism(preimage.chart.coords_to_point(c))
            )

        jacobian = jax.jacfwd(coord_map)(preimage.coords)
        transformed_t = self.t_coords
        for _ in range(self.n_contra):
            transformed_t = jnp.tensordot(transformed_t, jacobian, axes=([0], [0]))

        return ContravariantTensor(preimage, transformed_t)

    def contravariant_prod(
        self, other: "ContravariantTensor[P]"
    ) -> "ContravariantTensor[P]":
        tensor = self.tensor_prod(other)
        return ContravariantTensor(tensor.point, tensor.t_coords)


@dataclass(frozen=True)
class Tangent(Generic[P]):
    """An element of the tangent bundle on a manifold in some chart"""

    point: ChartPoint[P]
    v_coords: np.ndarray

    def pushforward(
        self, morphism: Callable[[P], P_], chart: Chart[P_]
    ) -> "Tangent[P_]":
        """Compute pushforward along a morphism, expressed in the given chart"""

        def coord_map(c: np.ndarray):
            return chart.point_to_coords(morphism(self.point.chart.coords_to_point(c)))

        image, tangent = jax.jvp(coord_map, [self.point.coords], [self.v_coords])
        return Tangent(ChartPoint(image, chart), tangent)

    def to_chart(self, chart: Chart[P]):
        return self.pushforward(lambda p: p, chart)

    def as_tensor(self) -> CovariantTensor[P]:
        return CovariantTensor(self.point, self.v_coords)


@dataclass(frozen=True)
class Cotangent(Generic[P]):
    """An element of the cotangent bundle on a manifold in some chart"""

    point: ChartPoint[P]
    v_coords: np.ndarray

    def pullback(
        self, morphism: Callable[[P_], P], preimage: ChartPoint[P_]
    ) -> "Cotangent[P_]":
        """Compute pullback along a morphism in the given chart"""

        def coord_map(c: np.ndarray):
            return self.point.chart.point_to_coords(
                morphism(preimage.chart.coords_to_point(c))
            )

        _, pullback_fun = jax.vjp(coord_map, preimage.coords)
        cotangent = pullback_fun(self.v_coords)
        return Cotangent(preimage, cotangent)

    def to_chart(self, chart: Chart[P]):
        image = ChartPoint.of_point(
            self.point.chart.coords_to_point(self.point.coords), chart
        )
        return self.pullback(lambda p: p, image)

    def as_tensor(self) -> ContravariantTensor[P]:
        return ContravariantTensor(self.point, self.v_coords)


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
