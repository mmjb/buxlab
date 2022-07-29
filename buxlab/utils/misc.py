import dataclasses
import math
from typing import Callable, Sequence, TypeVar

import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure

PyTreeT = TypeVar("PyTreeT")


def aggregate_pytree_leaves(
    pytrees: Sequence[PyTreeT], aggr: Callable[[jnp.ndarray], jnp.ndarray] = jnp.sum
) -> PyTreeT:
    assert len(pytrees) > 0, "Cannot aggregate empty sequence"
    treedef = tree_structure(pytrees[0])
    flat_vals = [tree_flatten(v)[0] for v in pytrees]
    return tree_unflatten(treedef, aggr(jnp.array(flat_vals), axis=0))


class DataclassPyTreeInterface:
    def tree_flatten(self):
        return (dataclasses.astuple(self), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def round_to_nearest_mult_of_power_of_2(num: int):
    if num <= 0:
        return 0
    magnitude = 2 ** math.floor(math.log(num, 2))
    return math.ceil(num / magnitude) * magnitude
