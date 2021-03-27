from typing import Callable, Tuple, Type, TypeVar

import jax
import jax.numpy as jnp

from .manifold import Chart, ChartPoint, Manifold, Tangent
from .connection import Christoffel


P = TypeVar("P")

# a connection maps a point to Christoffel symbols
Connection = Callable[[ChartPoint[P]], Christoffel[P]]

# a scheme defines a method of moving through time from one (x, v) pair to the next
# given a connection and a time interval
# see [explicit_step], for example
Scheme = Callable[
    [Connection[P], ChartPoint[P], Tangent[P], float], Tuple[ChartPoint[P], Tangent[P]]
]


def explicit_step(
    connection: Callable[[ChartPoint[P]], Christoffel[P]],
    x: ChartPoint[P],
    v: Tangent[P],
    dt: float,
) -> Tuple[ChartPoint[P], Tangent[P]]:
    """Given a connection, a point x, a tangent v, and a step size dt, compute the next values of x and v using an explicit scheme.

    Args:
      connection: a map from a point in a chart to Christoffel symbols at that point
      x: a point in the manifold
      v: a tangent vector anchored at x in any chart
      dt: time step size
    """
    # explicit scheme
    # advance x by v step
    x_new = ChartPoint(x.coords + dt * v.v_coords, x.chart)
    christoffel = connection(x)
    v_derivative = -christoffel.coords @ v.v_coords @ v.v_coords
    v_new = Tangent(x_new, v.v_coords + dt * v_derivative)
    return x_new, v_new


def midpoint_step(
    connection: Callable[[ChartPoint[P]], Christoffel[P]],
    x: ChartPoint[P],
    v: Tangent[P],
    dt: float,
) -> Tuple[ChartPoint[P], Tangent[P]]:
    """On the sphere, this scheme is both fast and very accurate at small step sizes.
    At some threshold (unclear exactly where) it quickly loses accuracy.
    """
    # step halfway to explicit step's x
    x_midpoint = ChartPoint(x.coords + 0.5 * dt * v.v_coords, x.chart)
    # use Christoffel symbols from approximate midpoint
    christoffel = connection(x_midpoint)
    v_derivative = -christoffel.coords @ v.v_coords @ v.v_coords
    # don't know where to anchor v_new yet, but need components
    v_new_coords = v.v_coords + dt * v_derivative
    # now take actual x step based on average of v and v_new_coords
    x_new = ChartPoint(x.coords + 0.5 * dt * (v.v_coords + v_new_coords), x.chart)
    v_new = Tangent(x_new, v_new_coords)
    return x_new, v_new


def make_exp_map_step(
    connection: Connection[P],
    scheme: Scheme[P],
    chart_cls: Type[Chart[P]],
):
    @jax.jit
    def exp_map_step(initial_x, initial_v, chart_coords, dt) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        """Run one step of the exponential map on the sphere.

        The JIT optimizes out essentially all the type-system fluff, leaving only DeviceArray operations.
        Some of those operations include autodiff so it's still not that fast, but it could run on a GPU.
        """
        x = ChartPoint(initial_x, chart_cls.of_array(chart_coords))
        # chart = sphere.preferred_chart(x_0)
        v = Tangent(x, initial_v)
        x_new, v_new = scheme(connection, x, v, dt)
        return x_new.coords, v_new.v_coords

    return exp_map_step
