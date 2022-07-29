# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for sparse (segment) operations in Jax; taken from the jraph library.
Started from github commit 8b0536c0; then `black`-ed."""

from typing import Optional

import jax
from jax import lax
import jax.numpy as jnp


def segment_sum(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
):
    """Computes the sum within segments of an array.

    Jraph alias to `jax.ops.segment_sum
    <https://jax.readthedocs.io/en/latest/_autosummary/jax.ops.segment_sum.html>`_.
    Note that other segment operations in jraph are not aliases, but are rather
    defined inside the package. Similar to TensorFlow's `segment_sum
    <https://www.tensorflow.org/api_docs/python/tf/math/segment_sum>`_.

    Args:
      data: an array with the values to be summed.
      segment_ids: an array with integer dtype that indicates the segments of
        `data` (along its leading axis) to be summed. Values can be repeated and
        need not be sorted. Values outside of the range [0, num_segments) are
        dropped and do not contribute to the sum.
      num_segments: optional, an int with nonnegative value indicating the number
        of segments. The default is set to be the minimum number of segments that
        would support all indices in ``segment_ids``, calculated as
        ``max(segment_ids) + 1``. Since `num_segments` determines the size of the
        output, a static value must be provided to use ``segment_sum`` in a
        ``jit``-compiled function.
      indices_are_sorted: whether ``segment_ids`` is known to be sorted.
      unique_indices: whether ``segment_ids`` is known to be free of duplicates.

    Returns:
      An array with shape :code:`(num_segments,) + data.shape[1:]` representing
      the segment sums.

    Examples:
      Simple 1D segment sum:

      >>> data = jnp.arange(5)
      >>> segment_ids = jnp.array([0, 0, 1, 1, 2])
      >>> segment_sum(data, segment_ids)
      DeviceArray([1, 5, 4], dtype=int32)

      Using JIT requires static `num_segments`:

      >>> from jax import jit
      >>> jit(segment_sum, static_argnums=2)(data, segment_ids, 3)
      DeviceArray([1, 5, 4], dtype=int32)
    """
    return jax.ops.segment_sum(
        data=data,
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )


def segment_mean(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
):
    """Returns mean for each segment.

    Args:
      data: the values which are averaged segment-wise.
      segment_ids: indices for the segments.
      num_segments: total number of segments.
      indices_are_sorted: whether ``segment_ids`` is known to be sorted.
      unique_indices: whether ``segment_ids`` is known to be free of duplicates.
    """
    nominator = segment_sum(
        data,
        segment_ids,
        num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )
    denominator = segment_sum(
        jnp.ones_like(data),
        segment_ids,
        num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )
    return nominator / jnp.maximum(denominator, jnp.ones(shape=[], dtype=denominator.dtype))


def segment_variance(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
):
    """Returns the variance for each segment.

    Args:
      data: values whose variance will be calculated segment-wise.
      segment_ids: indices for segments
      num_segments: total number of segments.
      indices_are_sorted: whether ``segment_ids`` is known to be sorted.
      unique_indices: whether ``segment_ids`` is known to be free of duplicates.

    Returns:
      num_segments size array containing the variance of each segment.
    """
    means = segment_mean(
        data,
        segment_ids,
        num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )[segment_ids]
    counts = segment_sum(
        jnp.ones_like(data),
        segment_ids,
        num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )
    counts = jnp.maximum(counts, jnp.ones_like(counts))
    variances = (
        segment_sum(
            jnp.power(data - means, 2),
            segment_ids,
            num_segments,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
        )
        / counts
    )
    return variances


def segment_normalize(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    eps=1e-8,
):
    """Normalizes data within each segment.

    Args:
      data: values whose z-score normalized values will be calculated.
        segment-wise.
      segment_ids: indices for segments.
      num_segments: total number of segments.
      indices_are_sorted: whether ``segment_ids`` is known to be sorted.
      unique_indices: whether ``segment_ids`` is known to be free of duplicates.
      eps: epsilon for numerical stability.

    Returns:
      array containing data normalized segment-wise.
    """

    means = segment_mean(
        data,
        segment_ids,
        num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )[segment_ids]
    variances = segment_variance(
        data,
        segment_ids,
        num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )[segment_ids]
    normalized = (data - means) * lax.rsqrt(jnp.maximum(variances, jnp.array(eps, dtype=variances.dtype)))
    return normalized


def segment_max(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
):
    """Alias for jax.ops.segment_max.

    Args:
      data: an array with the values to be maxed over.
      segment_ids: an array with integer dtype that indicates the segments of
        `data` (along its leading axis) to be maxed over. Values can be repeated
        and need not be sorted. Values outside of the range [0, num_segments) are
        dropped and do not contribute to the result.
      num_segments: optional, an int with positive value indicating the number of
        segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
        jnp.max(-segment_ids))`` but since `num_segments` determines the size of
        the output, a static value must be provided to use ``segment_max`` in a
        ``jit``-compiled function.
      indices_are_sorted: whether ``segment_ids`` is known to be sorted
      unique_indices: whether ``segment_ids`` is known to be free of duplicates

    Returns:
      An array with shape ``(num_segments,) + data.shape[1:]`` representing
      the segment maxs.
    """
    return jax.ops.segment_max(data, segment_ids, num_segments, indices_are_sorted, unique_indices)


def _replace_empty_segments_with_constant(
    aggregated_segments: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: Optional[int] = None,
    constant: float = 0.0,
):
    """Replaces the values of empty segments with constants."""
    result_shape = (len(segment_ids),) + aggregated_segments.shape[1:]
    num_elements_in_segment = segment_sum(
        jnp.ones(result_shape, dtype=jnp.int32), segment_ids, num_segments=num_segments
    )
    return jnp.where(
        num_elements_in_segment > 0,
        aggregated_segments,
        jnp.array(constant, dtype=aggregated_segments.dtype),
    )


def segment_min_or_constant(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    constant: float = 0.0,
):
    """As segment_min, but returns a constant for empty segments.

    `segment_min` returns `-inf` for empty segments, which can cause `nan`s in the
    backwards pass of a neural network, even with masking. This method overrides
    the default behaviour of `segment_min` and returns a constant for empty
    segments.

    Args:
      data: an array with the values to be maxed over.
      segment_ids: an array with integer dtype that indicates the segments of
        `data` (along its leading axis) to be maxed over. Values can be repeated
        and need not be sorted. Values outside of the range [0, num_segments) are
        dropped and do not contribute to the result.
      num_segments: optional, an int with positive value indicating the number of
        segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
        jnp.max(-segment_ids))`` but since `num_segments` determines the size of
        the output, a static value must be provided to use ``segment_min`` in a
        ``jit``-compiled function.
      indices_are_sorted: whether ``segment_ids`` is known to be sorted
      unique_indices: whether ``segment_ids`` is known to be free of duplicates
      constant: The constant to replace empty segments with, defaults to zero.

    Returns:
      An array with shape ``(num_segments,) + data.shape[1:]`` representing
      the segment maxs.
    """
    mins_ = segment_min(data, segment_ids, num_segments, indices_are_sorted, unique_indices)
    return _replace_empty_segments_with_constant(mins_, segment_ids, num_segments, constant)


def segment_max_or_constant(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    constant: float = 0.0,
):
    """As segment_max, but returns a constant for empty segments.

    `segment_max` returns `-inf` for empty segments, which can cause `nan`s in the
    backwards pass of a neural network, even with masking. This method overrides
    the default behaviour of `segment_max` and returns a constant for empty
    segments.

    Args:
      data: an array with the values to be maxed over.
      segment_ids: an array with integer dtype that indicates the segments of
        `data` (along its leading axis) to be maxed over. Values can be repeated
        and need not be sorted. Values outside of the range [0, num_segments) are
        dropped and do not contribute to the result.
      num_segments: optional, an int with positive value indicating the number of
        segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
        jnp.max(-segment_ids))`` but since `num_segments` determines the size of
        the output, a static value must be provided to use ``segment_max`` in a
        ``jit``-compiled function.
      indices_are_sorted: whether ``segment_ids`` is known to be sorted
      unique_indices: whether ``segment_ids`` is known to be free of duplicates
      constant: The constant to replace empty segments with, defaults to zero.

    Returns:
      An array with shape ``(num_segments,) + data.shape[1:]`` representing
      the segment maxs.
    """
    maxs_ = segment_max(data, segment_ids, num_segments, indices_are_sorted, unique_indices)
    return _replace_empty_segments_with_constant(maxs_, segment_ids, num_segments, constant)


def segment_min(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
):
    """Computes the min within segments of an array.

    Similar to TensorFlow's segment_min:
    https://www.tensorflow.org/api_docs/python/tf/math/segment_min

    Args:
      data: an array with the values to be maxed over.
      segment_ids: an array with integer dtype that indicates the segments of
        `data` (along its leading axis) to be min'd over. Values can be repeated
        and need not be sorted. Values outside of the range [0, num_segments) are
        dropped and do not contribute to the result.
      num_segments: optional, an int with positive value indicating the number of
        segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
        jnp.max(-segment_ids))`` but since `num_segments` determines the size of
        the output, a static value must be provided to use ``segment_max`` in a
        ``jit``-compiled function.
      indices_are_sorted: whether ``segment_ids`` is known to be sorted
      unique_indices: whether ``segment_ids`` is known to be free of duplicates

    Returns:
      An array with shape ``(num_segments,) + data.shape[1:]`` representing
      the segment mins.
    """
    return jax.ops.segment_min(data, segment_ids, num_segments, indices_are_sorted, unique_indices)


def segment_softmax(
    logits: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> jnp.ndarray:
    """Computes a segment-wise softmax.

    For a given tree of logits that can be divded into segments, computes a
    softmax over the segments.

      logits = jnp.ndarray([1.0, 2.0, 3.0, 1.0, 2.0])
      segment_ids = jnp.ndarray([0, 0, 0, 1, 1])
      segment_softmax(logits, segments)
      >> DeviceArray([0.09003057, 0.24472848, 0.66524094, 0.26894142, 0.7310586],
      >> dtype=float32)

    Args:
      logits: an array of logits to be segment softmaxed.
      segment_ids: an array with integer dtype that indicates the segments of
        `data` (along its leading axis) to be maxed over. Values can be repeated
        and need not be sorted. Values outside of the range [0, num_segments) are
        dropped and do not contribute to the result.
      num_segments: optional, an int with positive value indicating the number of
        segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
        jnp.max(-segment_ids))`` but since ``num_segments`` determines the size of
        the output, a static value must be provided to use ``segment_sum`` in a
        ``jit``-compiled function.
      indices_are_sorted: whether ``segment_ids`` is known to be sorted
      unique_indices: whether ``segment_ids`` is known to be free of duplicates

    Returns:
      The segment softmax-ed ``logits``.
    """
    # First, subtract the segment max for numerical stability
    maxs = segment_max(logits, segment_ids, num_segments, indices_are_sorted, unique_indices)
    logits = logits - maxs[segment_ids]
    # Then take the exp
    logits = jnp.exp(logits)
    # Then calculate the normalizers
    normalizers = segment_sum(logits, segment_ids, num_segments, indices_are_sorted, unique_indices)
    normalizers = normalizers[segment_ids]
    softmax = logits / normalizers
    return softmax


def segment_log_softmax(
    logits: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> jnp.ndarray:
    # First, subtract the segment max for numerical stability
    maxs = segment_max(logits, segment_ids, num_segments, indices_are_sorted, unique_indices)
    logits = logits - maxs[segment_ids]
    # Then calculate the normalizers
    normalizers = segment_sum(jnp.exp(logits), segment_ids, num_segments, indices_are_sorted, unique_indices)
    normalizers = jnp.log(normalizers)[segment_ids]
    log_softmax = logits - normalizers
    return log_softmax
