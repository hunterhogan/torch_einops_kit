from __future__ import annotations

from torch import Tensor
from torch_einops_kit import (
	align_dims_left, pad_left_ndim, pad_left_ndim_to, pad_ndim, pad_right_ndim, pad_right_ndim_to, pad_right_ndim_to_and_expand_as,
	repeat_interleave_to_match)
import pytest
import torch

with torch.sparse.check_sparse_tensor_invariants(False):
	sparseVectorPaddingSource = torch.sparse_coo_tensor(torch.tensor([[0, 2]]), torch.tensor([5.0, 7.0]), (3,))  # pyright: ignore[reportUnknownMemberType]
	sparseMatrixPaddingSource = torch.sparse_coo_tensor(torch.tensor([[0, 1], [1, 2]]), torch.tensor([11.0, 13.0]), (2, 3))  # pyright: ignore[reportUnknownMemberType]

@pytest.mark.parametrize('ndim', [pytest.param(None, id='infer-ndim'), pytest.param(1, id='ndim-one'), pytest.param(3, id='ndim-three')])
def test_align_dims_left(t: Tensor, ndim: int | None) -> None:
	tensor_sequence = (t,)

	if ndim is not None and ndim < t.ndim:
		with pytest.raises(ValueError, match='greater than or equal to `0`'):
			align_dims_left(tensor_sequence, ndim=ndim)
		return

	aligned = align_dims_left(tensor_sequence, ndim=ndim)
	expected_ndim = t.ndim if ndim is None else max(ndim, t.ndim)
	expected_shape = (*t.shape, *(1,) * (expected_ndim - t.ndim))

	assert len(aligned) == 1, f'align_dims_left returned {len(aligned)} tensors, expected 1 for {ndim=} and {tuple(t.shape)=}.'
	assert tuple(aligned[0].shape) == expected_shape, (
		f'align_dims_left returned shape {tuple(aligned[0].shape)}, expected {expected_shape} for {ndim=} and {tuple(t.shape)=}.'
	)
	assert torch.equal(aligned[0].reshape(t.shape), t), f'align_dims_left changed tensor values for {ndim=} and {tuple(t.shape)=}.'

@pytest.mark.parametrize(
	('explicit_ndim', 'expected_length', 'expected_exception'),
	[pytest.param(None, None, ValueError, id='empty-inferred-ndim'), pytest.param(3, 0, None, id='empty-explicit-ndim')],
)
def test_align_dims_left_empty_sequence(
	explicit_ndim: int | None, expected_length: int | None, expected_exception: type[Exception] | None
) -> None:
	empty: tuple[Tensor, ...] = ()

	if expected_exception is not None:
		with pytest.raises(expected_exception):
			align_dims_left(empty, ndim=explicit_ndim)
	else:
		result = align_dims_left(empty, ndim=explicit_ndim)
		assert len(result) == expected_length, (
			f'align_dims_left returned {len(result)} tensors for empty input, expected {expected_length} with {explicit_ndim=}.'
		)

@pytest.mark.parametrize(
	('tensors', 'explicit_ndim'),
	[
		pytest.param(
			(torch.tensor([127.0, 131.0]), torch.tensor([[137.0, 139.0], [149.0, 151.0]])), 1, id='explicit-ndim-smaller-than-matrix-rank'
		)
	],
)
def test_align_dims_left_raises_when_explicit_ndim_too_small(tensors: tuple[Tensor, ...], explicit_ndim: int) -> None:
	with pytest.raises(ValueError, match='greater than or equal to `0`'):
		align_dims_left(tensors, ndim=explicit_ndim)

@pytest.mark.parametrize('ndims', [pytest.param(2, id='two-dims'), pytest.param(3, id='three-dims')])
def test_pad_left_ndim(t: Tensor, ndims: int) -> None:
	expected_shape = (*(1,) * ndims, *t.shape)
	padded = pad_left_ndim(t, ndims)

	assert tuple(padded.shape) == expected_shape, (
		f'pad_left_ndim returned shape {tuple(padded.shape)}, expected {expected_shape} for {ndims=} and {tuple(t.shape)=}.'
	)
	assert torch.equal(padded.reshape(t.shape), t), f'pad_left_ndim changed tensor values for {ndims=} and {tuple(t.shape)=}.'

@pytest.mark.parametrize(
	'target_ndim_delta',
	[
		pytest.param(-1, id='target-below-current'),
		pytest.param(0, id='target-equals-current'),
		pytest.param(2, id='target-exceeds-by-two'),
		pytest.param(5, id='target-exceeds-by-five'),
	],
)
def test_pad_left_ndim_to(t: Tensor, target_ndim_delta: int) -> None:
	target_ndim = t.ndim + target_ndim_delta
	added = max(0, target_ndim_delta)
	expected_shape = (*(1,) * added, *t.shape)
	result = pad_left_ndim_to(t, target_ndim)

	assert tuple(result.shape) == expected_shape, (
		f'pad_left_ndim_to returned shape {tuple(result.shape)}, expected {expected_shape} for {target_ndim=} and {tuple(t.shape)=}.'
	)
	assert torch.equal(result.reshape(t.shape), t), f'pad_left_ndim_to changed tensor values for {target_ndim=} and {tuple(t.shape)=}.'

@pytest.mark.parametrize(
	('left_padding', 'right_padding'),
	[pytest.param(0, 0, id='no-padding'), pytest.param(2, 1, id='two-left-one-right'), pytest.param(1, 3, id='one-left-three-right')],
)
def test_pad_ndim(t: Tensor, left_padding: int, right_padding: int) -> None:
	expected_shape = (*(1,) * left_padding, *t.shape, *(1,) * right_padding)
	padded = pad_ndim(t, (left_padding, right_padding))

	assert tuple(padded.shape) == expected_shape, (
		f'pad_ndim returned shape {tuple(padded.shape)}, expected {expected_shape} '
		f'for {left_padding=}, {right_padding=}, and {tuple(t.shape)=}.'
	)
	assert torch.equal(padded.reshape(t.shape), t), (
		f'pad_ndim changed tensor values for {left_padding=}, {right_padding=}, and {tuple(t.shape)=}.'
	)

def test_pad_ndim_raises_for_negative_padding(tensor_malformed_padding: tuple[Tensor, int, int]) -> None:
	tensor_value, left_padding, right_padding = tensor_malformed_padding
	with pytest.raises(ValueError, match='greater than or equal to `0`'):
		pad_ndim(tensor_value, (left_padding, right_padding))

@pytest.mark.parametrize('ndims', [pytest.param(2, id='two-dims'), pytest.param(3, id='three-dims')])
def test_pad_right_ndim(t: Tensor, ndims: int) -> None:
	expected_shape = (*t.shape, *(1,) * ndims)
	padded = pad_right_ndim(t, ndims)

	assert tuple(padded.shape) == expected_shape, (
		f'pad_right_ndim returned shape {tuple(padded.shape)}, expected {expected_shape} for {ndims=} and {tuple(t.shape)=}.'
	)
	assert torch.equal(padded.reshape(t.shape), t), f'pad_right_ndim changed tensor values for {ndims=} and {tuple(t.shape)=}.'

@pytest.mark.parametrize(
	'target_ndim_delta',
	[
		pytest.param(-1, id='target-below-current'),
		pytest.param(0, id='target-equals-current'),
		pytest.param(2, id='target-exceeds-by-two'),
		pytest.param(5, id='target-exceeds-by-five'),
	],
)
def test_pad_right_ndim_to(t: Tensor, target_ndim_delta: int) -> None:
	target_ndim = t.ndim + target_ndim_delta
	added = max(0, target_ndim_delta)
	expected_shape = (*t.shape, *(1,) * added)
	result = pad_right_ndim_to(t, target_ndim)

	assert tuple(result.shape) == expected_shape, (
		f'pad_right_ndim_to returned shape {tuple(result.shape)}, expected {expected_shape} for {target_ndim=} and {tuple(t.shape)=}.'
	)
	assert torch.equal(result.reshape(t.shape), t), f'pad_right_ndim_to changed tensor values for {target_ndim=} and {tuple(t.shape)=}.'

@pytest.mark.parametrize(
	('t', 'target', 'expected'),
	[
		pytest.param(
			torch.tensor([2.0, 3.0])
			, torch.full((5, 2, 3), 7.0)
			, torch.tensor([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]])
			, id='vector-expands-over-target-trailing-axes'
		)
		, pytest.param(
			torch.tensor([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
			, torch.full((17, 19, 2), 23.0)
			, torch.tensor([[[2.0, 2.0], [3.0, 3.0], [5.0, 5.0]], [[7.0, 7.0], [11.0, 11.0], [13.0, 13.0]]])
			, id='matrix-expands-over-one-target-trailing-axis'
		)
		, pytest.param(
			torch.tensor([[29.0, 31.0], [37.0, 41.0], [43.0, 47.0]])
			, torch.full((53, 59), 61.0)
			, torch.tensor([[29.0, 31.0], [37.0, 41.0], [43.0, 47.0]])
			, id='same-rank-keeps-source-shape'
		)
		, pytest.param(
			torch.tensor([[67.0, 71.0, 73.0], [79.0, 83.0, 89.0]])
			, torch.full((97,), 101.0)
			, torch.tensor([[67.0, 71.0, 73.0], [79.0, 83.0, 89.0]])
			, id='lower-rank-target-keeps-source-shape'
		)
	],
)
def test_pad_right_ndim_to_and_expand_as(t: Tensor, target: Tensor, expected: Tensor) -> None:
	result = pad_right_ndim_to_and_expand_as(t, target)

	assert tuple(result.shape) == tuple(expected.shape), (
		f'pad_right_ndim_to_and_expand_as returned shape {tuple(result.shape)}, expected {tuple(expected.shape)} '
		f'for {tuple(t.shape)=} and {tuple(target.shape)=}.'
	)
	assert result.dtype == expected.dtype, (
		f'pad_right_ndim_to_and_expand_as returned dtype {result.dtype}, expected {expected.dtype} '
		f'for {tuple(t.shape)=} and {tuple(target.shape)=}.'
	)
	assert torch.equal(result, expected), (
		f'pad_right_ndim_to_and_expand_as returned {result}, expected {expected} for {tuple(t.shape)=} and {tuple(target.shape)=}.'
	)

@pytest.mark.parametrize(
	('t', 'target', 'expected'),
	[
		pytest.param(sparseVectorPaddingSource, torch.full((3, 5), 7.0), RuntimeError, id='sparse-vector-cannot-reshape-for-padding'),
		pytest.param(sparseMatrixPaddingSource, torch.full((2, 3, 5), 17.0), RuntimeError, id='sparse-matrix-cannot-reshape-for-padding'),
	],
)
def test_pad_right_ndim_to_and_expand_asError(t: Tensor, target: Tensor, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		pad_right_ndim_to_and_expand_as(t, target)

@pytest.mark.parametrize(
	('t', 'target', 'dim', 'target_dim', 'expected'),
	[
		pytest.param(
			torch.tensor([[2.0, 3.0], [5.0, 7.0]])
			, 6
			, 0
			, None
			, torch.tensor([[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [5.0, 7.0], [5.0, 7.0], [5.0, 7.0]])
			, id='integer-target-repeats-leading-dimension'
		)
		, pytest.param(
			torch.tensor([[11.0, 13.0], [17.0, 19.0]])
			, torch.full((23, 29, 6), 31.0)
			, 1
			, 2
			, torch.tensor([[11.0, 11.0, 11.0, 13.0, 13.0, 13.0], [17.0, 17.0, 17.0, 19.0, 19.0, 19.0]])
			, id='tensor-target-repeats-source-dim-from-explicit-target-dim'
		)
		, pytest.param(
			torch.tensor([[37.0, 41.0], [43.0, 47.0]])
			, torch.full((4, 53), 59.0)
			, 0
			, None
			, torch.tensor([[37.0, 41.0], [37.0, 41.0], [43.0, 47.0], [43.0, 47.0]])
			, id='tensor-target-repeats-source-dim-from-default-target-dim'
		)
		, pytest.param(
			torch.tensor([[61.0, 67.0, 71.0], [73.0, 79.0, 83.0]])
			, 3
			, -1
			, None
			, torch.tensor([[61.0, 67.0, 71.0], [73.0, 79.0, 83.0]])
			, id='matching-integer-target-keeps-values'
		)
	],
)
def test_repeat_interleave_to_match(t: Tensor, target: Tensor | int, dim: int, target_dim: int | None, expected: Tensor) -> None:
	result = repeat_interleave_to_match(t, target, dim=dim, target_dim=target_dim)
	resolved_target_dim = dim if target_dim is None else target_dim
	target_length = target if isinstance(target, int) else target.shape[resolved_target_dim]

	assert tuple(result.shape) == tuple(expected.shape), (
		f'repeat_interleave_to_match returned shape {tuple(result.shape)}, expected {tuple(expected.shape)} '
		f'for {tuple(t.shape)=}, {target_length=}, {dim=}, and {target_dim=}.'
	)
	assert result.dtype == expected.dtype, (
		f'repeat_interleave_to_match returned dtype {result.dtype}, expected {expected.dtype} '
		f'for {tuple(t.shape)=}, {target_length=}, {dim=}, and {target_dim=}.'
	)
	assert torch.equal(result, expected), (
		f'repeat_interleave_to_match returned {result}, expected {expected} '
		f'for {tuple(t.shape)=}, {target_length=}, {dim=}, and {target_dim=}.'
	)
	if target_length == t.shape[dim]:
		assert result is t, (
			f'repeat_interleave_to_match returned a new tensor for equal lengths, expected original tensor for {tuple(t.shape)=} and {dim=}.'
		)

@pytest.mark.parametrize(
	('t', 'target', 'dim', 'target_dim', 'expected'),
	[
		pytest.param(torch.tensor([[2.0, 3.0], [5.0, 7.0]]), 5, 0, None, ValueError, id='integer-target-not-divisible')
		, pytest.param(
			torch.tensor([[11.0, 13.0, 17.0], [19.0, 23.0, 29.0]])
			, torch.full((31, 5), 37.0)
			, 0
			, 1
			, ValueError
			, id='tensor-target-dimension-not-divisible'
		)
		, pytest.param(torch.tensor([[41.0, 43.0], [47.0, 53.0]]), 5, 3, None, IndexError, id='source-dimension-out-of-range')
		, pytest.param(
			torch.tensor([[59.0, 61.0], [67.0, 71.0]]), torch.full((73, 79), 83.0), 0, 3, IndexError, id='target-dimension-out-of-range'
		)
	],
)
def test_repeat_interleave_to_matchError(
	t: Tensor, target: Tensor | int, dim: int, target_dim: int | None, expected: type[Exception]
) -> None:
	with pytest.raises(expected):
		repeat_interleave_to_match(t, target, dim=dim, target_dim=target_dim)
