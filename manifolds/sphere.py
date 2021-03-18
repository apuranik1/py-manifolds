from typing import Sequence

import jax.numpy as jnp

from manifolds.euclidean import EuclideanPoint, EuclideanSpace
from manifolds.manifold import Chart, ChartPoint
from manifolds.riemannian import (
    PseudoMetric,
    PseudoRiemannianManifold,
    metric_of_immersion,
)


class SpherePoint:
    """A point on an n-sphere, represented as an embedding into R^{n+1}"""

    def __init__(self, coords: jnp.DeviceArray):
        self.coords = coords


class StereographicChart(Chart[SpherePoint]):
    """Stereographic projection from north or south pole.

    Projects the sphere to the plane from point (0, ..., 0, signed_radius)
    """

    def __init__(self, signed_radius: jnp.DeviceArray) -> None:
        self.signed_radius = signed_radius

    def point_to_coords(self, point: SpherePoint) -> jnp.DeviceArray:
        (dim,) = point.coords.shape
        if dim < 2:
            raise ValueError("Sphere of dimension < 1 is not supported")
        u = point.coords[:-1]
        v = point.coords[-1]
        # ray from (0, R) through (u, v) to point (p, 0)
        # (u, v) - (0, R) = lambda * ((p, 0) - (0, R))
        # u = lambda p
        # v - R = -lambda * R
        # lambda = 1 - v/R?
        # p = u / lambda
        lam = -v / self.signed_radius + 1
        p = u / lam
        return p

    def coords_to_point(self, coords: jnp.DeviceArray) -> SpherePoint:
        (dim,) = coords.shape
        if dim < 1:
            raise ValueError("Sphere of dimension < 1 is not supported")
        # |u|^2 + v^2 = R^2
        # |u|^2 = (R - v)(R + v)
        # u = p * (1 - v/R)
        # Ru = p * (R - v)
        # R^2 |u|^2 = |p|^2 (R - v)^2
        # R^2 (R - v)(R + v) = |p|^2 (R - v)^2
        # R^2(R + v) = |p|^2(R - v)
        # R^3 + R^2 v = |p|^2 R - |p|^2 v
        # v(R^2 + |p|^2) = R(|p|^2 - R^2)
        # v = R * (|p|^2 - R^2)/(|p|^2 + R^2)
        p_norm_sq = coords @ coords
        v_over_R = (p_norm_sq - self.signed_radius ** 2) / (
            p_norm_sq + self.signed_radius ** 2
        )
        u = coords * (1 - v_over_R)
        v = self.signed_radius * v_over_R
        embed_coords = jnp.append(u, jnp.array([v]))
        return SpherePoint(embed_coords)

    def __repr__(self) -> str:
        return f"StereographicChart({self.signed_radius})"


def sphere_embedding(p: SpherePoint) -> EuclideanPoint:
    return EuclideanPoint(p.coords)


class Sphere(PseudoRiemannianManifold[SpherePoint]):
    def __init__(self, radius: float) -> None:
        self.radius = jnp.array(radius, dtype=jnp.float32)
        self._metric_impl = metric_of_immersion(
            sphere_embedding, self, EuclideanSpace()
        )

    def northpole_chart(self):
        return StereographicChart(self.radius)

    def southpole_chart(self):
        return StereographicChart(-self.radius)

    def charts(self, point: SpherePoint) -> Sequence[StereographicChart]:
        charts = []
        v: float = point.coords[-1].item()
        if v < self.radius * (1 - 1e4):
            charts.append(self.northpole_chart())
        if v > -self.radius * (1 - 1e-4):
            charts.append(self.southpole_chart())
        return charts

    def preferred_chart(self, point: SpherePoint) -> StereographicChart:
        signed_radius = jnp.where(point.coords[-1] >= 0, -self.radius, self.radius)
        return StereographicChart(signed_radius)

    def metric(self, point: SpherePoint) -> PseudoMetric[SpherePoint]:
        return self._metric_impl(point)

    def metric_in_chart(
        self, point: ChartPoint[SpherePoint]
    ) -> PseudoMetric[SpherePoint]:
        return super().metric_in_chart(point)

    def __repr__(self) -> str:
        return f"Sphere({self.radius})"
