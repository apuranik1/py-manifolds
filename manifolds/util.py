from typing import Any, Optional, TYPE_CHECKING

# circular import only needed for type checking
if TYPE_CHECKING:
    from manifolds.manifold import Tensor
else:
    Tensor = Any


def assert_shape(
    t: Tensor,
    n_cov: Optional[int] = None,
    n_contra: Optional[int] = None,
    n_indices: Optional[int] = None,
):
    if n_cov is not None and t.n_cov != n_cov:
        raise ValueError(f"Got n_cov={t.n_cov}. Expected n_cov={n_cov}.")
    if n_contra is not None and t.n_contra != n_contra:
        raise ValueError(f"Got n_contra={t.n_contra}. Expected n_contra={n_contra}.")
    if n_indices is not None and t.n_indices != n_indices:
        raise ValueError(
            f"Got n_indices={t.n_indices}. Expected n_indices={n_indices}."
        )
