from typing import Any, Sequence, TypeVar

import jax
import jax.numpy as jnp

from .manifold import Chart, ChartPoint
from .riemannian import PseudoMetric, PseudoRiemannianManifold


P = TypeVar("P")


class EuclideanPoint:
    def __init__(self, coords: jax.Array):
        self.coords = coords

    @property
    def dim(self):
        return self.coords.shape[0]


class IdChart(Chart[EuclideanPoint]):
    """Identity chart, mapping points to themselves"""

    def point_to_coords(self, point: EuclideanPoint) -> jax.Array:
        return point.coords

    def coords_to_point(self, coords: jax.Array) -> EuclideanPoint:
        return EuclideanPoint(coords)

    def to_array(self) -> jax.Array:
        return jnp.array(0.)

    @classmethod
    def of_array(cls: Any, arr: jax.Array) -> "IdChart":
        return IdChart()


class EuclideanSpace(PseudoRiemannianManifold[EuclideanPoint]):
    def preferred_chart(self, point: EuclideanPoint) -> IdChart:
        return IdChart()

    def charts(self, point: EuclideanPoint) -> Sequence[IdChart]:
        return [IdChart()]

    def metric(self, point: EuclideanPoint) -> PseudoMetric[EuclideanPoint]:
        # it's easier to provide metric instead of metric_in_chart
        return PseudoMetric(ChartPoint.of_point(point, IdChart()), jnp.eye(point.dim))

    def metric_in_chart(
        self, point: ChartPoint[EuclideanPoint]
    ) -> PseudoMetric[EuclideanPoint]:
        # 99% of the time we want the identity matrix
        if isinstance(point.chart, IdChart):
            return PseudoMetric(point, jnp.eye(point.coords.shape[0]))
        # some crazy person might use a custom chart
        return super().metric_in_chart(point)
