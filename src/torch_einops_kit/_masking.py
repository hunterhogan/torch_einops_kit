from __future__ import annotations

from einops import rearrange
from functools import reduce
from torch import arange, Tensor
from torch_einops_kit import exists, safe, shift_left, shift_right
from torch_einops_kit.scaleValues import reverse_cumsum
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
	from collections.abc import Callable, Sequence
	from torch.types import Number

def and_masks(masks: Sequence[Tensor | None]) -> Tensor | None:
	"""Reduce a `Sequence` of boolean mask `Tensor` values to a single mask using element-wise logical AND.

	You can use `and_masks` to combine multiple boolean masks so that the result is `True` only where
	all non-`None` input masks are `True`. `and_masks` calls `reduce_masks` [1] with
	`torch.logical_and` [2]. `None` values in `masks` are filtered out before reduction. If all values
	in `masks` are `None`, `and_masks` returns `None`.

	Parameters
	----------
	masks : Sequence[Tensor | None]
		A `Sequence` of boolean `Tensor` or `None` values. `None` values are treated as absent and
		filtered out before reduction. All non-`None` `Tensor` values must have the same shape.

	Returns
	-------
	mask : Tensor | None
		A boolean `Tensor` that is `True` only at positions where every non-`None` input mask is
		`True`. Returns `None` if `masks` contains no non-`None` values.

	See Also
	--------
	or_masks : Reduce masks using element-wise logical OR.
	reduce_masks : Reduce masks using a caller-supplied binary operator.

	Examples
	--------
	From sdft_pytorch [4], intersecting an end-of-sequence mask with an initial-token mask to exclude
	padding and masked prefix positions from the loss calculation:

	```python
		mask = and_masks([eos_mask, init_tokens_mask])
	```

	References
	----------
	[1] torch_einops_kit.reduce_masks

	[2] torch.logical_and - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.logical_and.html
	[4] lucidrains/sdft-pytorch
		https://github.com/lucidrains/sdft-pytorch
	"""
	return reduce_masks(masks, torch.logical_and)

def lens_to_mask(lens: Tensor, max_len: Number | None = None) -> Tensor:
	"""Convert a sequence of length values into a boolean mask `Tensor`.

	You can use `lens_to_mask` to create sequence masks from integer length values. For each scalar in
	`lens`, `lens_to_mask` produces a row of `True` values for positions less than that length value
	and `False` values for all positions equal to or greater than it. The output `Tensor` has one more
	dimension than `lens`, appended at the last axis, with length `max_len`.

	Parameters
	----------
	lens : Tensor
		A `Tensor` of non-negative integers representing sequence lengths. `lens` may have any shape;
		the output shape is `(*lens.shape, max_len)`.
	max_len : int | None = None
		The size of the last dimension of the output `Tensor`. If `None`, `max_len` is set to
		`int(lens.amax().item())`.

	Returns
	-------
	mask : Tensor
		A boolean `Tensor` of shape `(*lens.shape, max_len)`. Position `i` along the last axis is
		`True` if `i < lens[...]` for the corresponding element of `lens`.

	Examples
	--------
	From dreamer4 [2], masking padded time steps in variable-length rollouts:

	```python
		mask_for_gae = lens_to_mask(experience.lens, time)
	```

	References
	----------
	[2] lucidrains/dreamer4
		https://github.com/lucidrains/dreamer4
	"""
	device: torch.device = lens.device

	if not exists(max_len):
		max_len = lens.amax().item()

	seq: Tensor = arange(max_len, device=device)
	lens = rearrange(lens, '... -> ... 1')
	return seq < lens

def mask_after(t: Tensor, value: Tensor | Number, dim: int = -1, *, inclusive: bool = True) -> Tensor:
	"""Compute a boolean mask over `t` that stays `True` through the first `value` along `dim`.

	You can use `mask_after` to keep positions in `t` along `dim` until the first occurrence of
	`value`. `mask_after` returns a boolean `Tensor` with the same shape as `t`. When
	`inclusive=True`, the first matching position remains `True`. When `inclusive=False`, `mask_after`
	sets the first matching position to `False` together with all later positions. `mask_after` uses
	`shift_right` [1] and `Tensor.cumsum` [2] to build the result.

	Parameters
	----------
	t : Tensor
		The input `Tensor` to scan for `value`.
	value : Tensor | Number
		The delimiter value to match against `t`. `value` may be a scalar `Number` or a `Tensor`
		broadcastable to `t` for the element-wise comparison.
	dim : int = -1
		The dimension along which `mask_after` scans for the first occurrence of `value`.
	inclusive : bool = True
		When `True`, the first matching position in each slice of `t` remains `True`. When `False`,
		the first matching position becomes `False`.

	Returns
	-------
	mask : Tensor
		A boolean `Tensor` with the same shape as `t`. For each slice of `t` along `dim`, `mask` is
		`True` before the first occurrence of `value`, and also at the first occurrence when
		`inclusive=True`. If a slice contains no occurrence of `value`, `mask` is `True` for the
		entire slice.

	See Also
	--------
	mask_before : Compute the complementary mask from the reverse direction.
	lens_to_mask : Convert length values into prefix masks without searching for a delimiter.

	Examples
	--------
	From `SAC_pytorch.SAC` [3], excluding critic-loss positions after the first terminal step in
	each rollout sequence:

	```python
		loss_mask = mask_after(done, True)
	```

	From `train_chunked_fql.py` [4], combining the terminal-step mask with `lens_to_mask` [5] so
	the loss excludes both post-terminal positions and padded positions:

	```python
		loss_mask = mask_after(terminal, True) & lens_to_mask(n_step_lens, seq_len)
	```

	References
	----------
	[1] torch_einops_kit.shift_right

	[2] torch.Tensor.cumsum - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.Tensor.cumsum.html
	[3] lucidrains/SAC-pytorch `SAC_pytorch/SAC.py`
		https://github.com/lucidrains/SAC-pytorch/blob/main/SAC_pytorch/SAC.py
	[4] lucidrains/rectified-flow-pytorch `train_chunked_fql.py`
		https://github.com/lucidrains/rectified-flow-pytorch/blob/main/train_chunked_fql.py
	[5] torch_einops_kit.lens_to_mask
	"""
	mask: Tensor = t == value
	if inclusive:
		mask = shift_right(mask, amount = 1, dim = dim, pad_value = False)
	return mask.float().cumsum(dim = dim) == 0.

def mask_before(t: Tensor, value: Tensor | Number, dim: int = -1, *, inclusive: bool = True) -> Tensor:
	"""Compute a boolean mask over `t` that stays `True` from the last `value` onward along `dim`.

	You can use `mask_before` to keep positions in `t` along `dim` from the last occurrence of `value`
	to the end of each slice. `mask_before` returns a boolean `Tensor` with the same shape as `t`.
	When `inclusive=True`, the last matching position remains `True`. When `inclusive=False`,
	`mask_before` sets the last matching position to `False` and keeps only later positions `True`.
	`mask_before` uses `shift_left` [1] and `reverse_cumsum` [2] to build the result.

	Parameters
	----------
	t : Tensor
		The input `Tensor` to scan for `value`.
	value : Tensor | Number
		The delimiter value to match against `t`. `value` may be a scalar `Number` or a `Tensor`
		broadcastable to `t` for the element-wise comparison.
	dim : int = -1
		The dimension along which `mask_before` scans for the last occurrence of `value`.
	inclusive : bool = True
		When `True`, the last matching position in each slice of `t` remains `True`. When `False`, the
		last matching position becomes `False`.

	Returns
	-------
	mask : Tensor
		A boolean `Tensor` with the same shape as `t`. For each slice of `t` along `dim`, `mask` is
		`True` after the last occurrence of `value`, and also at the last occurrence when
		`inclusive=True`. If a slice contains no occurrence of `value`, `mask` is `True` for the
		entire slice.

	Delimiter handling
	------------------
	repeated matches : behavior
		`mask_before` uses `reverse_cumsum` [2], so the final occurrence of `value` along each slice
		of `t` determines where `mask` becomes `True`. Earlier occurrences of `value` do not change
		the final boundary.

	See Also
	--------
	mask_after : Compute the complementary mask from the forward direction.
	lens_to_mask : Convert length values into prefix masks without searching for a delimiter.

	References
	----------
	[1] torch_einops_kit.shift_left

	[2] torch_einops_kit.reverse_cumsum
	"""
	mask: Tensor = t == value
	if inclusive:
		mask = shift_left(mask, amount = 1, dim = dim, pad_value = False)
	return reverse_cumsum(mask.float(), dim = dim) == 0.

def or_masks(masks: Sequence[Tensor | None]) -> Tensor | None:
	"""Reduce a sequence of boolean mask `Tensor` values to a single mask using element-wise logical OR.

	You can use `or_masks` to combine multiple boolean masks so that the result is `True` wherever at
	least one non-`None` input mask is `True`. `or_masks` calls `reduce_masks` [1] with
	`torch.logical_or` [2]. `None` values in `masks` are filtered out before reduction. If all values
	in `masks` are `None`, `or_masks` returns `None`.

	Parameters
	----------
	masks : Sequence[Tensor | None]
		A sequence of boolean `Tensor` or `None` values. `None` values are treated as absent and
		filtered out before reduction. All non-`None` `Tensor` values must have the same shape.

	Returns
	-------
	mask : Tensor | None
		A boolean `Tensor` that is `True` at any position where at least one non-`None` input mask is
		`True`. Returns `None` if `masks` contains no non-`None` values.

	See Also
	--------
	and_masks : Reduce masks using element-wise logical AND.
	reduce_masks : Reduce masks using a caller-supplied binary operator.

	References
	----------
	[1] torch_einops_kit.reduce_masks

	[2] torch.logical_or - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.logical_or.html
	"""
	return reduce_masks(masks, torch.logical_or)

@safe
def reduce_masks(masks: Sequence[Tensor], op: Callable[[Tensor, Tensor], Tensor]) -> Tensor | None:
	"""Reduce a `Sequence` of `Tensor` values to a single `Tensor` using a binary operator.

	You can use `reduce_masks` to apply any binary element-wise callable reduction over a sequence of
	`Tensor` values. The `safe` [1] decorator filters out `None` values from `masks` before `op` is
	applied. If `masks` contains no non-`None` values, `reduce_masks` returns `None`. Reduction
	proceeds left-to-right over the elements of `masks`.

	Parameters
	----------
	masks : Sequence[Tensor | None]
		A `Sequence` of `Tensor` or `None` values. `None` values are filtered out before `op` is
		applied.
	op : Callable[[Tensor, Tensor], Tensor]
		A binary callable that accepts two `Tensor` arguments and returns a `Tensor`.

	Returns
	-------
	mask : Tensor | None
		The result of applying `op` cumulatively, left-to-right, over the non-`None` elements of
		`masks`. Returns `None` if no non-`None` values remain after filtering.

	See Also
	--------
	and_masks : Reduce masks using element-wise logical AND.
	or_masks : Reduce masks using element-wise logical OR.

	References
	----------
	[1] torch_einops_kit.safe
	"""
	return reduce(op, masks) if masks else None

"""
Some of the logic in this module may be protected by the following.

MIT License

Copyright (c) 2026 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
