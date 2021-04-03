from typing import Sequence

import jax.numpy as jnp
from jax import ops

from .manifold import Chart, ChartPoint
from .riemannian import PseudoMetric, PseudoRiemannianManifold


class SchwarzschildPoint:
    """Point in Schwarzschild space, represented in Cartesian coordinates.

    Coordinate order is (t, x, y, z) and follows the convention c = 1.
    """
    def __init__(self, coords: jnp.DeviceArray):
        if coords.shape[-1] != 4:
            raise ValueError("Only 4D spacetime is supported")
        self.coords = coords


class SchwarzschildChart(Chart[SchwarzschildPoint]):
    """Incomplete implementation of Schwarzschild spacetime charts.
    Currently no chart can cross the event horizon.
    """
    def __init__(self, radius: jnp.DeviceArray):
        self.radius = radius

    def coords_to_point(self, coords: jnp.DeviceArray) -> SchwarzschildPoint:
        return SchwarzschildPoint(coords)

    def point_to_coords(self, point: SchwarzschildPoint) -> jnp.DeviceArray:
        return point.coords

    def to_array(self) -> jnp.DeviceArray:
        return self.radius

    @classmethod
    def of_array(cls, arr: jnp.DeviceArray) -> "SchwarzschildChart":
        if not arr.shape == ():
            raise ValueError("SchwarzschildChart requires 0D radius")
        return SchwarzschildChart(arr)


class SchwarzschildSpacetime(PseudoRiemannianManifold[SchwarzschildPoint]):
    """Schwarzschild solution to the Einstein Field Equations.

    Partial implemention that does not cross the event horizon.
    """
    def __init__(self, radius: jnp.DeviceArray):
        self.radius = radius

    def preferred_chart(self, point: SchwarzschildPoint) -> Chart[SchwarzschildPoint]:
        return SchwarzschildChart(self.radius)

    def charts(self, point: SchwarzschildPoint) -> Sequence[Chart[SchwarzschildPoint]]:
        return [SchwarzschildChart(self.radius)]

    def metric_in_chart(self, point: ChartPoint[SchwarzschildPoint]) -> PseudoMetric[SchwarzschildPoint]:
        if isinstance(point.chart, SchwarzschildChart):
            # happy path
            # metric has time component, radial component, spherical component
            spatial = point.coords[1:4]
            r2 = spatial @ spatial
            r = jnp.sqrt(r2)
            a = 1. - self.radius / r  # this quantity appears twice in the metric
            time_comp = -a  # dt^2 component
            radial_comp = a ** -1  # dr^2 component
            angular_comp = 1.
            # dr = (x dx + y dy + z dz) / r
            # dr^2 = outer(space dspace, space dspace) / r^2
            dr2 = jnp.outer(spatial, spatial) / r2
            # spherical metric has [spatial] as kernel, other eigs 1
            # therefore equal to I - dr^2
            sphere_metric = jnp.eye(3) - dr2
            space_comp = radial_comp * dr2 + angular_comp * sphere_metric
            metric = ops.index_update(jnp.zeros((4, 4)), (0, 0), time_comp)
            metric = ops.index_update(metric, ops.index[1:4, 1:4], space_comp)
            return PseudoMetric(point, metric)
        else:
            # some psychopath might pass their own custom chart
            usual_chart = SchwarzschildChart(self.radius)
            # get metric in the usual chart
            usual_point = ChartPoint.of_point(point.to_point(), usual_chart)
            # then convert to whatever the user passed in
            return self.metric_in_chart(usual_point).to_chart(point.chart)

    def metric(self, point: SchwarzschildPoint) -> PseudoMetric[SchwarzschildPoint]:
        return super().metric(point)
