from typing import Sequence

import numpy as np

from manifolds.manifold import Chart, Manifold


class EuclideanPoint:
    def __init__(self, coords: np.ndarray):
        self.coords = coords

class IdChart(Chart[EuclideanPoint]):
    """Identity chart, mapping points to themselves"""

    def point_to_coords(self, point: EuclideanPoint) -> np.ndarray:
        return point.coords

    def coords_to_point(self, coords: np.ndarray) -> EuclideanPoint:
        return EuclideanPoint(coords)


class EuclideanSpace(Manifold[EuclideanPoint]):

    def preferred_chart(self, point: EuclideanPoint) -> IdChart:
        return IdChart()

    def charts(self, point: EuclideanPoint) -> Sequence[IdChart]:
        return [IdChart()]
