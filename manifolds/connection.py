from typing import Callable, Generic, TypeVar

import jax.numpy as jnp

from .manifold import ChartPoint, Tangent, Tensor

P = TypeVar("P")


class Christoffel(Generic[P]):
    """Christoffel symbols at a point on a manifold.

    As with tensors, operations are only well-defined at the same point in the same
    chart. For performance and practicality, this condition is not checked.

    An explicitly defined function will usually ignore the chart.
    A tensor field's coordinates are always defined on a chart.
    We're happy if the charts coincide, but can't count on it?
    Well, maybe we can count on it, since tensor field is always in a chart.
    """

    def __init__(self, point: ChartPoint[P], coords: jnp.DeviceArray):
        """Initialize Christoffel symbols from coordinates.

        Order of indices must be upper, first lower, second lower.
        """
        if not len(coords.shape) == 3 and coords.shape:
            raise ValueError("Christoffel symbols require three indices")
        a, b, c = coords.shape
        if not a == b and a == c:
            raise ValueError("All axes should be of equal length")
        self.point = point
        self.coords = coords

    def __repr__(self) -> str:
        return f"Christoffel({self.point}, {self.coords})"

    def directional_derivative(
        self, direction: Tangent[P], field: Callable[[P], Tangent[P]]
    ) -> Tangent[P]:
        # in coordinates, D_X(Y) = (X Y^k)d_k + X_i Y_j Gamma^k_ij d_k
        field_at_point, coord_derivs = direction.derive_autodiff(field)
        second_term = self.coords @ direction.v_coords @ field_at_point.v_coords
        return Tangent(self.point, coord_derivs + second_term)

    def tcd_autodiff(self, tensor_field: Callable[[P], Tensor[P]]) -> Tensor[P]:
        """Compute the total covariant derivative of a tensor field in the same chart.

        The TCD is returned as a tensor at self.point with one new contravariant index.
        The new index is the last contravariant index.

        Preconditions:
          - tensor_field(p) is a tensor at p, in the same chart as self. If the chart
            may not match, use tcd_safe instead.
        """
        # the coordinate expression for this is an explosion of product rule
        # basically you have to differentiate every index separately, I think?
        # probably best to start with the covariant derivative of a covector
        raise NotImplementedError()

    def tcd_autodiff_safe(self, tensor_field: Callable[[P], Tensor[P]]) -> Tensor[P]:
        """Compute total covariant derivative of a tensor field in any chart"""
        return self.tcd_autodiff(lambda p: tensor_field(p).to_chart(self.point.chart))

    # changing the chart of a connection is a pain, so not implementing that yet
