import abc
from typing import Callable, Generic, TypeVar

import jax.numpy as jnp
import numpy as np

from manifolds.manifold import ChartPoint, Tensor

P = TypeVar("P")


class Connection(Generic[P]):
    """Connection on a manifold in a chart.

    As with tensors, operations are only well-defined at the same point in the same
    chart. For performance and practicality, this condition is not checked.

    An explicitly defined function will usually ignore the chart.
    A tensor field's coordinates are always defined on a chart.
    We're happy if the charts coincide, but can't count on it?
    Well, maybe we can count on it, since tensor field is always in a chart.
    """

    def __init__(self, point: ChartPoint[P], coords: np.ndarray):
        self.point = point
        self.coords = coords

    # need: directional derivative, total derivative
    def tcd(self, tensor_field: Callable[[P], Tensor[P]]) -> Tensor[P]:
        """Compute the total covariant derivative of a tensor field in the same chart.

        The TCD is returned as a tensor at self.point with one new contravariant index.
        The new index is the last contravariant index.

        Preconditions:
          - tensor_field(p) is a tensor at p, in the same chart as self. If the chart
            may not match, use tcd_safe instead.
        """
        raise NotImplementedError()

    def tcd_safe(self, tensor_field: Callable[[P], Tensor[P]]) -> Tensor[P]:
        """Compute total covariant derivative of a tensor field in any chart"""
        return self.tcd(lambda p: tensor_field(p).to_chart(self.point.chart))

    # changing the chart of a connection is a pain, so not implementing that yet
