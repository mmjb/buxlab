from functools import partial
from typing import TypeVar, Callable, Union
import re

ValueType = TypeVar("ValueType")


LINEAR_INTERPOLATION_REGEX = re.compile(
    r"""
        interpolate\(
            (?P<num_epochs>[0-9]+)                     # always match a number of epochs
            (?:,\s?(?P<init_weight>[0-9]*\.[0-9]+))?   # allow a middle argument for initial weight (defaults to 1.0)
            (?:,\s?(?P<final_weight>[0-9]*\.[0-9]+))?  # final argument for final weight (defaults to 0.0)
        \)""",
    re.VERBOSE,
)


def const_schedule(step_idx: int, const: ValueType) -> float:
    return const


def linear_interpolation(step_idx: int, num_steps: int, init_weight: float, final_weight: float) -> float:
    if step_idx < num_steps:
        return init_weight + (final_weight - init_weight) * (step_idx / num_steps)

    return final_weight


def make_weight_schedule(weight_spec: Union[float, str]) -> Callable[[int], float]:
    """Return a (serializable) function with the appropriate schedule"""
    if isinstance(weight_spec, float):
        return partial(const_schedule, const=weight_spec)

    match = LINEAR_INTERPOLATION_REGEX.match(weight_spec)
    if match:
        num_epochs = int(match.group("num_epochs"))
        init_weight = float(match.group("init_weight") or "1.0")
        final_weight = float(match.group("final_weight") or "0.0")
        return partial(
            linear_interpolation,
            num_epochs=num_epochs,
            init_weight=init_weight,
            final_weight=final_weight,
        )

    raise Exception(f"Unrecognized weight schedule spec `{weight_spec}`")
