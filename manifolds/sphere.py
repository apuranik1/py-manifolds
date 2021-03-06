from typing import Sequence, Optional

import numpy as np
import jax.numpy as jnp

from manifolds.manifold import Chart, Manifold


class SpherePoint:
    """A point on an n-sphere, represented as an embedding into R^{n+1}"""
    def __init__(self, coords: np.ndarray):
        self.coords = coords


class StereographicChart(Chart[SpherePoint]):
    """Stereographic projection from north or south pole.

    North pole is (0, ..., 0, radius). South pole is (0, ..., 0, -radius).
    """
    def __init__(self, radius: float, pole: str) -> None:
        self.radius = radius
        self.is_north = pole == "north"
        # TODO: rewrite point_to_coords and coords_to_point without branching

    def point_to_coords(self, point: SpherePoint) -> np.ndarray:
        dim, = point.coords.shape
        if dim < 2:
            raise ValueError("Sphere of dimension < 1 is not supported")
        u = point.coords[:-1]
        v = point.coords[-1]
        if self.is_north:
            # ray from (0, R) through (u, v) to point (p, 0)
            # (u, v) - (0, R) = lambda * ((p, 0) - (0, R))
            # u = lambda p
            # v - R = -lambda * R
            # lambda = 1 - v/R?
            # p = u / lambda
            lam = -v / self.radius + 1
        else:
            # symmetric: v -> -v
            lam = v / self.radius + 1
        p = u / lam
        return p

    def coords_to_point(self, coords: np.ndarray) -> SpherePoint:
        dim, = coords.shape
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
        # alternative is just replacing v with -v
        p_norm_sq = coords @ coords
        v_over_R = (p_norm_sq - self.radius ** 2) / (p_norm_sq + self.radius ** 2)
        u = coords * (1 - v_over_R)
        if self.is_north:
            v = self.radius * v_over_R
        else:
            v = -self.radius * v_over_R
        embed_coords = jnp.append(u, jnp.array([v]))
        return SpherePoint(embed_coords)


class Sphere(Manifold):
    def __init__(self, radius: float) -> None:
        self.radius = radius
        self.northpole_chart = StereographicChart(radius, "north")
        self.southpole_chart = StereographicChart(radius, "south")

    def charts(self, point: SpherePoint) -> Sequence[StereographicChart]:
        charts = []
        v: float = point.coords[-1].item()  # type: ignore
        if v < self.radius * (1 - 1e4):
            charts.append(self.northpole_chart)
        if v > -self.radius * (1 - 1e-4):
            charts.append(self.southpole_chart)
        return charts

    def preferred_chart(self, point: SpherePoint) -> StereographicChart:
        v = point.coords[-1].item()
        if v >= 0:
            return self.southpole_chart
        else:
            return self.northpole_chart
