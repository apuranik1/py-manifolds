from typing import Sequence, TypeVar

import jax.numpy as jnp
import numpy as np

from manifolds.manifold import Chart, ChartPoint
from manifolds.riemannian import PseudoMetric, PseudoRiemannianManifold


P = TypeVar("P")


class EuclideanPoint:
    def __init__(self, coords: np.ndarray):
        self.coords = coords

    @property
    def dim(self):
        return self.coords.shape[0]


class IdChart(Chart[EuclideanPoint]):
    """Identity chart, mapping points to themselves"""

    def point_to_coords(self, point: EuclideanPoint) -> np.ndarray:
        return point.coords

    def coords_to_point(self, coords: np.ndarray) -> EuclideanPoint:
        return EuclideanPoint(coords)


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
        # I think this usually does unneeded multiplication by the identity matrix
        return super().metric_in_chart(point)
