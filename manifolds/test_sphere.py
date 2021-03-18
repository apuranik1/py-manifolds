import jax.numpy as jnp
import pytest

# from manifolds.euclidean import EuclideanPoint, IdChart, EuclideanSpace
from manifolds.manifold import (
    ChartPoint,
    Cotangent,
    ContravariantTensor,
    CovariantTensor,
    Tangent,
    Tensor,
)
from manifolds.sphere import Sphere, StereographicChart, SpherePoint


def sphere() -> Sphere:
    return Sphere(5.0)


def northpole() -> SpherePoint:
    return SpherePoint(jnp.array([0.0, 0.0, 5.0]))


def southpole() -> SpherePoint:
    return SpherePoint(jnp.array([0.0, 0.0, -5.0]))


def north_point() -> SpherePoint:
    return SpherePoint(jnp.array([3.0, 0.0, 4.0]))


def north_projection() -> StereographicChart:
    return StereographicChart(5.0)


def south_projection() -> StereographicChart:
    return StereographicChart(-5.0)


def test_preferred_chart() -> None:
    """Test preferred chart of points"""
    s = sphere()
    northpole_chart = s.preferred_chart(northpole())
    n_radius = northpole_chart.signed_radius.item()  # type: ignore
    assert n_radius == -5.0

    southpole_chart = s.preferred_chart(southpole())
    s_radius = southpole_chart.signed_radius.item()  # type: ignore
    assert s_radius == 5.0

    north_point_chart = s.preferred_chart(north_point())
    np_radius = north_point_chart.signed_radius.item()
    assert np_radius == -5.0


def test_tangent_transformation() -> None:
    """Test a tangent vector in different charts"""
    t = Tangent(
        ChartPoint.of_point(north_point(), south_projection()), jnp.array([1.0, 1.0])
    )
    t_alt = t.to_chart(north_projection())
    assert t_alt.point.coords.tolist() == pytest.approx([15.0, 0.0])
    assert t_alt.v_coords.tolist() == pytest.approx([-9.0, 9.0])


def test_cotangent_transformation() -> None:
    """Test a cotangent vector in different charts"""
    ct = Cotangent(
        ChartPoint.of_point(north_point(), south_projection()), jnp.array([1.0, 1.0])
    )
    ct_alt = ct.to_chart(north_projection())
    assert ct_alt.point.coords.tolist() == pytest.approx([15.0, 0.0])
    assert ct_alt.v_coords.tolist() == pytest.approx([-1 / 9, 1 / 9])


def test_tensor_transformations() -> None:
    """Test tensor in different charts"""
    # tensor products of covariants don't commute
    t1 = Tensor(
        ChartPoint.of_point(north_point(), south_projection()), jnp.array([1.0, 1.0]), 0
    )
    t2 = Tensor(
        ChartPoint.of_point(north_point(), south_projection()),
        jnp.array([1.0, -1.0]),
        0,
    )
    tprod1 = t1.tensor_prod(t2)
    assert tprod1.t_coords.shape == (2, 2)
    assert tprod1.t_coords.ravel().tolist() == pytest.approx([1.0, -1.0, 1.0, -1.0])
    tprod2 = t2.tensor_prod(t1)
    assert tprod2.t_coords.shape == (2, 2)
    assert tprod2.t_coords.ravel().tolist() == pytest.approx([1.0, 1.0, -1.0, -1.0])

    # tensor product of covariant and contravariant commutes (sort of)
    # because we always put contravariant indices first
    ct1 = Tensor(
        ChartPoint.of_point(north_point(), south_projection()), jnp.array([1.0, 2.0]), 1
    )
    tprod3 = t1.tensor_prod(ct1)
    assert tprod3.t_coords.ravel().tolist() == pytest.approx([1.0, 1.0, 2.0, 2.0])
    tprod4 = ct1.tensor_prod(t1)
    assert tprod4.t_coords.ravel().tolist() == pytest.approx([1.0, 1.0, 2.0, 2.0])

    prod_then_chart = t1.tensor_prod(ct1).to_chart(north_projection())
    chart_then_prod = t1.to_chart(north_projection()).tensor_prod(
        ct1.to_chart(north_projection())
    )
    assert prod_then_chart.t_coords.shape == (2, 2)
    assert chart_then_prod.t_coords.shape == (2, 2)
    assert jnp.allclose(prod_then_chart.t_coords, chart_then_prod.t_coords).item()


def test_derivative() -> None:
    def x_projection(p: SpherePoint) -> float:
        return p.coords[0]

    # derivative at north pole in direction (1, 0) proj from south is 2
    northpole = ChartPoint(jnp.array([0.0, 0.0]), south_projection())
    t1 = Tangent(northpole, jnp.array([1.0, 0.0]))
    assert t1.derive_autodiff(x_projection).item() == pytest.approx(2.0)
    # derivative at north pole in direction (0, 1) proj from south is 0
    t2 = Tangent(northpole, jnp.array([0.0, 1.0]))
    assert t2.derive_autodiff(x_projection).item() == pytest.approx(0.0)
    # derivative at (5, 0) in any direction is 0
    t3 = Tangent(
        ChartPoint(jnp.array([5.0, 0.0]), south_projection()), jnp.array([10.0, -6.0])
    )
    assert t3.derive_autodiff(x_projection).item() == pytest.approx(0.0)
