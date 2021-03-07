import abc
from typing import Callable, TypeVar

import jax.numpy as jnp
import numpy as np

from manifolds.manifold import (
    Chart,
    ChartPoint,
    ContravariantTensor,
    Cotangent,
    CovariantTensor,
    Manifold,
    Tangent,
    Tensor,
)
from manifolds.util import assert_shape


P = TypeVar("P")
P_ = TypeVar("P_")


class PseudoMetric(ContravariantTensor[P]):
    """A psuedo-Riemannian metric on a manifold. For correct semantics, coords must be
    symmetric and non-singular, but this is not verified. If coords is
    positive-definite, then this will be an true Riemannian metric.

    Convenience functions are provided for operations on tangent and cotangent vectors.
    """

    def __init__(self, point: ChartPoint[P], coords: np.ndarray):
        """Create a metric from coordinates. """
        if not len(coords.shape) == 2:
            raise ValueError("Metric must have exactly two indices")
        super().__init__(point, coords)

    @staticmethod
    def of_tensor_exn(t: Tensor[P]) -> "PseudoMetric[P]":
        assert_shape(t, n_cov=0, n_contra=2)
        return PseudoMetric(t.point, t.t_coords)

    def pullback(
        self, morphism: Callable[[P_], P], preimage: ChartPoint[P_]
    ) -> "PseudoMetric[P_]":
        tensor = super().pullback(morphism, preimage)
        return PseudoMetric(tensor.point, tensor.t_coords)

    def to_chart(self, chart: Chart[P]) -> "PseudoMetric[P]":
        return PseudoMetric.of_tensor_exn(super().to_chart(chart))

    def flat(self, vector: Tangent[P]) -> Cotangent[P]:
        flat_tensor = self.contract(0, vector.as_tensor(), 0)
        return Cotangent.of_tensor_exn(flat_tensor)

    def sharp(self, vector: Cotangent[P]) -> Tangent[P]:
        coords = jnp.linalg.solve(self.t_coords, vector.v_coords)
        return Tangent(self.point, coords)

    def inverse(self) -> CovariantTensor[P]:
        """Returns a covariant 2-tensor"""
        coords = jnp.linalg.inv(self.t_coords)
        return CovariantTensor(self.point, coords)

    def inner(self, a: Tangent[P], b: Tangent[P]) -> np.ndarray:
        """Compute inner product of vectors, as 0-D array"""
        return jnp.dot(a.v_coords, jnp.dot(self.t_coords, b.v_coords))


class PseudoRiemannianManifold(Manifold[P]):
    """A Pseudo-Riemannian manifold equipped with a metric"""

    @abc.abstractmethod
    def metric_in_chart(self, point: ChartPoint[P]) -> PseudoMetric[P]:
        """Compute the metric at the point in the given chart"""
        return self.metric(point.to_point()).to_chart(point.chart)

    @abc.abstractmethod
    def metric(self, point: P) -> PseudoMetric[P]:
        """Compute the metric in any chart at point"""
        chart = self.preferred_chart(point)
        return self.metric_in_chart(ChartPoint.of_point(point, chart))


def metric_of_immersion(
    immersion: Callable[[P], P_],
    domain: Manifold[P],
    codomain: PseudoRiemannianManifold[P_],
) -> Callable[[P], PseudoMetric[P]]:
    """Construct a metric as a pullback.

    Provides an implementation of metric_in_chart."""

    def metric(point: P) -> PseudoMetric[P]:
        image = immersion(point)
        image_metric = codomain.metric(image)
        return image_metric.pullback(
            immersion, ChartPoint.of_point(point, domain.preferred_chart(point))
        )

    return metric
