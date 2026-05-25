"""Provide `einops` pack, unpack, and paired inverse utilities.

You can use this module to pack one or more `torch.Tensor` objects using an `einops` pattern [1]
and to recover the original shapes through paired inverse functions. `pack_one` and `unpack_one`
handle the single-tensor case. `pack_with_inverse` handles both the single-tensor and
list-of-tensors case and is re-exported from the package root [2].

Contents
--------
Functions
	pack_one
		Pack one `Tensor` and return shape metadata for paired reconstruction.
	pack_with_inverse
		Pack `t` with an einops pattern and return a paired inverse unpacking function.
	unpack_one
		Unpack one `Tensor` using packed-shape metadata produced by `pack_one`.

References
----------
[1] einops pack/unpack API
	https://einops.rocks/api/pack_unpack/
[2] torch_einops_kit
"""

from __future__ import annotations

from einops import pack, unpack
from torch import is_tensor, Tensor
from torch_einops_kit import default, first
from typing import overload, TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Sequence
	from torch_einops_kit import InversePackListTensors, InversePackTensor

def pack_one(t: Tensor, pattern: str) -> tuple[Tensor, Sequence[tuple[int, ...] | list[int]]]:
	"""Pack one `Tensor` and return shape metadata for paired reconstruction.

	You can use `pack_one` to wrap a single `Tensor` in the `einops.pack` [1] interface,
	capturing packed-shape metadata needed to restore the original shape with `unpack_one` [2].

	Parameters
	----------
	t : Tensor
		Input `Tensor` to pack.
	pattern : str
		Einops packing pattern string passed to `einops.pack` [1].

	Returns
	-------
	packedTensorAndPackedShape : tuple[Tensor, Sequence[tuple[int, ...] | list[int]]]
		Packed `Tensor` output and packed-shape metadata for reconstruction.

	See Also
	--------
	unpack_one : Reconstruct one `Tensor` using packed-shape metadata from `pack_one`.

	References
	----------
	[1] einops pack/unpack API
		https://einops.rocks/api/pack_unpack/
	[2] torch_einops_kit.einops.unpack_one
	"""
	return pack([t], pattern)

@overload
def pack_with_inverse(t: Tensor, pattern: str) -> tuple[Tensor, InversePackTensor]: ...
@overload
def pack_with_inverse(t: list[Tensor], pattern: str) -> tuple[Tensor, InversePackListTensors]: ...
def pack_with_inverse(t: Tensor | list[Tensor], pattern: str) -> tuple[Tensor, InversePackTensor] | tuple[Tensor, InversePackListTensors]:
	"""Pack `t` with `pattern` and return a paired inverse unpacking function.

	You can use this function to merge one `Tensor` or one `list[Tensor]` into one packed `Tensor`
	with `einops.pack` [1] and to carry forward an inverse callable tied to the captured packed-shape
	metadata. The returned `inverse` callable accepts a packed or transformed `Tensor` `out`,
	optionally accepts an override unpacking pattern `inv_pattern`, and reconstructs either one
	`Tensor` or `list[Tensor]` to match the kind of `t`.

	Parameters
	----------
	t : Tensor | list[Tensor]
		One `Tensor` or one `list[Tensor]` to pack.
	pattern : str
		Einops pack pattern string passed to `einops.pack` [1]. The `*` axis marks the dimensions to
		collect into the packed axis.

	Returns
	-------
	packed : Tensor
		The packed tensor produced by `einops.pack` [1].
	inverse : InversePackTensor | InversePackListTensors
		Paired inverse callable. When `t` is one `Tensor`, `inverse` has type `InversePackTensor` [2]
		and returns one `Tensor`. When `t` is a `list[Tensor]`, `inverse` has type
		`InversePackListTensors` [3] and returns `list[Tensor]`.

	Output Kind
	-----------
	tensor input : dispatch
		Passing one `Tensor` returns an `inverse` callable that unwraps the single unpacked result.
	list input : dispatch
		Passing `list[Tensor]` returns an `inverse` callable that preserves the unpacked list
		structure.

	Unpacking Pattern
	-----------------
	`inv_pattern` override : pattern reuse
		The returned `inverse` reuses `pattern` when `inv_pattern` is `None`. Passing `inv_pattern`
		forwards `inv_pattern` to `einops.unpack` [4] so `inverse` can unpack a derived `Tensor`, as
		long as `inv_pattern` remains compatible with the packed-shape metadata captured during
		packing.

	See Also
	--------
	pack_one :
		Pack one `Tensor` and return shape metadata for paired reconstruction.
	tree_flatten_with_inverse :
		Flatten a PyTree and return an inverse reconstruction function.
	unpack_one :
		Unpack one `Tensor` using metadata produced by `pack_one`.

	Examples
	--------
	Pack one tensor and recover the original shape.

	```python
		import torch

		from torch_einops_kit import pack_with_inverse

		t = torch.randn(3, 12, 2, 2)
		packed, inverse = pack_with_inverse(t, 'b * d')

		assert packed.shape == (3, 24, 2)

		recovered = inverse(packed)
		assert recovered.shape == (3, 12, 2, 2)
	```

	References
	----------
	[1] einops.pack - einops documentation
		https://einops.rocks/api/pack/
	[2] torch_einops_kit.InversePackTensor

	[3] torch_einops_kit.InversePackListTensors

	[4] einops pack/unpack API
		https://einops.rocks/api/pack_unpack/

	"""
	if is_tensor(t):
		is_one = True
		sequenceT: Sequence[Tensor] = [t]
	else:
		is_one = False
		sequenceT = t

	packed, packed_shape = pack(sequenceT, pattern)

	def inverse_is_one(out: Tensor, inv_pattern: str | None = None) -> Tensor:
		inv_pattern = default(inv_pattern, pattern)
		unpacked: list[Tensor] = unpack(out, packed_shape, inv_pattern)

		return first(unpacked)

	def inverse_is_list(out: Tensor, inv_pattern: str | None = None) -> list[Tensor]:
		inv_pattern = default(inv_pattern, pattern)
		unpacked: list[Tensor] = unpack(out, packed_shape, inv_pattern)

		return unpacked

	if is_one:
		return packed, inverse_is_one
	return packed, inverse_is_list

def unpack_one(t: Tensor, ps: Sequence[tuple[int, ...] | list[int]], pattern: str) -> Tensor:
	"""Unpack one `Tensor` using packed-shape metadata produced by `pack_one`.

	You can use `unpack_one` with metadata from a paired `pack_one` [1] call to reconstruct the
	original `Tensor` shape. The function delegates to `einops.unpack` [2] and extracts the single
	unpacked output.

	Parameters
	----------
	t : Tensor
		Packed `Tensor` produced by `pack_one` [1].
	ps : Sequence[tuple[int, ...] | list[int]]
		Packed-shape metadata returned by `pack_one` [1].
	pattern : str
		Einops unpacking pattern string passed to `einops.unpack` [2].

	Returns
	-------
	unpackedTensor : Tensor
		Reconstructed `Tensor` matching the original shape before packing.

	See Also
	--------
	pack_one : Pack one `Tensor` and return shape metadata for paired reconstruction.

	References
	----------
	[1] torch_einops_kit.einops.pack_one

	[2] einops pack/unpack API
		https://einops.rocks/api/pack_unpack/
	"""
	return unpack(t, list(ps), pattern)[0]

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
