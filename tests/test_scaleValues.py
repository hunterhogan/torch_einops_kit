from __future__ import annotations

from torch import Tensor
from torch_einops_kit.scaleValues import exclusive_cumsum, l2norm, masked_mean, reverse_cumsum, RMSNorm
import pytest
import torch

SCALE_VALUES_TENSORS: dict[str, Tensor] = {
	'rank-1-primes-a': torch.tensor([2.0, 3.0, 5.0, 7.0]),
	'rank-1-primes-b': torch.tensor([11.0, 13.0, 17.0]),
	'rank-2-primes-a': torch.tensor([[19.0, 23.0, 29.0], [31.0, 37.0, 41.0]]),
	'rank-2-primes-b': torch.tensor([[43.0, 47.0, 53.0], [59.0, 61.0, 67.0]]),
}

SCALE_VALUES_EXPECTED_TENSORS: dict[str, Tensor] = {
	'rank-1-primes-a-exclusive-trailing': torch.tensor([0.0, 2.0, 5.0, 10.0]),
	'rank-1-primes-b-exclusive-leading': torch.tensor([0.0, 11.0, 24.0]),
	'rank-1-primes-a-reverse-trailing-keepdim': torch.tensor([17.0, 15.0, 12.0, 7.0]),
	'rank-1-primes-b-reverse-leading-no-keepdim': torch.tensor([41.0, 30.0, 17.0]),
	'rank-2-primes-a-exclusive-trailing': torch.tensor([[0.0, 19.0, 42.0], [0.0, 31.0, 68.0]]),
	'rank-2-primes-a-reverse-trailing-keepdim': torch.tensor([[71.0, 52.0, 29.0], [109.0, 78.0, 41.0]]),
	'rank-2-primes-b-exclusive-leading': torch.tensor([[0.0, 0.0, 0.0], [43.0, 47.0, 53.0]]),
	'rank-2-primes-b-reverse-leading-no-keepdim': torch.tensor([[102.0, 108.0, 120.0], [59.0, 61.0, 67.0]]),
}

@pytest.fixture
def scale_values_tensor(request: pytest.FixtureRequest) -> Tensor:
	tensor_key: str = request.param
	return SCALE_VALUES_TENSORS[tensor_key]

@pytest.fixture
def scale_values_expected_tensor(request: pytest.FixtureRequest) -> Tensor:
	expected_tensor_key: str = request.param
	return SCALE_VALUES_EXPECTED_TENSORS[expected_tensor_key]

@pytest.mark.parametrize(
	('scale_values_tensor', 'reductionDim', 'scale_values_expected_tensor'),
	[
		pytest.param('rank-1-primes-a', -1, 'rank-1-primes-a-exclusive-trailing', id='rank-1-trailing'),
		pytest.param('rank-1-primes-b', 0, 'rank-1-primes-b-exclusive-leading', id='rank-1-leading'),
		pytest.param('rank-2-primes-a', -1, 'rank-2-primes-a-exclusive-trailing', id='rank-2-trailing'),
		pytest.param('rank-2-primes-b', 0, 'rank-2-primes-b-exclusive-leading', id='rank-2-leading'),
	],
	indirect=['scale_values_tensor', 'scale_values_expected_tensor'],
)
def test_exclusive_cumsum(scale_values_tensor: Tensor, reductionDim: int, scale_values_expected_tensor: Tensor) -> None:
	inputTensor = scale_values_tensor
	expectedTensor = scale_values_expected_tensor
	resultTensor = exclusive_cumsum(inputTensor, dim=reductionDim)

	assert resultTensor.shape == expectedTensor.shape, (
		f'exclusive_cumsum returned shape {tuple(resultTensor.shape)}, expected {tuple(expectedTensor.shape)} '
		f'for input shape {tuple(inputTensor.shape)} and {reductionDim=}.'
	)
	assert torch.equal(resultTensor, expectedTensor), (
		f'exclusive_cumsum returned {resultTensor}, expected {expectedTensor} for input {inputTensor} and {reductionDim=}.'
	)

@pytest.mark.parametrize(
	('scale_values_tensor', 'reductionDim', 'keepdim', 'scale_values_expected_tensor'),
	[
		pytest.param('rank-1-primes-a', -1, True, 'rank-1-primes-a-reverse-trailing-keepdim', id='rank-1-trailing-keepdim'),
		pytest.param('rank-1-primes-b', 0, False, 'rank-1-primes-b-reverse-leading-no-keepdim', id='rank-1-leading-no-keepdim'),
		pytest.param('rank-2-primes-a', -1, True, 'rank-2-primes-a-reverse-trailing-keepdim', id='rank-2-trailing-keepdim'),
		pytest.param('rank-2-primes-b', 0, False, 'rank-2-primes-b-reverse-leading-no-keepdim', id='rank-2-leading-no-keepdim'),
	],
	indirect=['scale_values_tensor', 'scale_values_expected_tensor'],
)
def test_reverse_cumsum(scale_values_tensor: Tensor, reductionDim: int, keepdim: bool, scale_values_expected_tensor: Tensor) -> None:
	inputTensor = scale_values_tensor
	expectedTensor = scale_values_expected_tensor
	resultTensor = reverse_cumsum(inputTensor, dim=reductionDim, keepdim=keepdim)

	assert resultTensor.shape == expectedTensor.shape, (
		f'reverse_cumsum returned shape {tuple(resultTensor.shape)}, expected {tuple(expectedTensor.shape)} '
		f'for input shape {tuple(inputTensor.shape)}, {reductionDim=}, and {keepdim=}.'
	)
	assert torch.equal(resultTensor, expectedTensor), (
		f'reverse_cumsum returned {resultTensor}, expected {expectedTensor} '
		f'for input {inputTensor}, {reductionDim=}, and {keepdim=}.'
	)

@pytest.mark.parametrize('tolerance', [pytest.param(1e-5, id='tolerance-1e-5')])
def test_l2norm(t: Tensor, tolerance: float) -> None:
	inputTensor = t.to(dtype=torch.float64)
	resultTensor = l2norm(inputTensor)

	assert resultTensor.shape == inputTensor.shape, (
		f'l2norm returned shape {tuple(resultTensor.shape)}, expected {tuple(inputTensor.shape)} for input shape {tuple(t.shape)}.'
	)

	vectorNorms = torch.sqrt((resultTensor * resultTensor).sum(dim=-1))
	expectedNorms = torch.ones_like(vectorNorms)
	assert torch.allclose(vectorNorms, expectedNorms, atol=tolerance), (
		f'l2norm produced non-unit vector norms {vectorNorms} for input shape {tuple(t.shape)} with {tolerance=}.'
	)

@pytest.mark.parametrize('eps', [pytest.param(1e-5, id='eps-default')])
def test_masked_mean(t: Tensor, boolean_mask_like_t: Tensor, reduction_dim: int | None, eps: float) -> None:
	tensor_value = t.to(dtype=torch.float64)
	mask_value = boolean_mask_like_t

	expanded_mask = mask_value
	if expanded_mask.ndim < tensor_value.ndim:
		expanded_mask = expanded_mask.reshape((*expanded_mask.shape, *(1,) * (tensor_value.ndim - expanded_mask.ndim)))
	expanded_mask = expanded_mask.expand_as(tensor_value)

	if reduction_dim is None:
		selected_values = tensor_value[expanded_mask]
		expected = selected_values.mean() if bool(expanded_mask.any()) else selected_values.sum()
	else:
		numerator = (tensor_value * expanded_mask).sum(dim=reduction_dim)
		denominator = expanded_mask.sum(dim=reduction_dim)
		expected = numerator / denominator.clamp(min=eps)

	result = masked_mean(tensor_value, mask=mask_value, dim=reduction_dim, eps=eps)

	assert result.shape == expected.shape, (
		f'masked_mean returned shape {tuple(result.shape)}, expected {tuple(expected.shape)} '
		f'for {tuple(t.shape)=}, {tuple(mask_value.shape)=}, and {reduction_dim=}.'
	)
	assert torch.allclose(result, expected), (
		f'masked_mean returned values {result} that do not match expected {expected} '
		f'for {tuple(t.shape)=}, {tuple(mask_value.shape)=}, and {reduction_dim=}.'
	)

@pytest.mark.parametrize(
	('tensor_dtype', 'tolerance'), [pytest.param(torch.float32, 1e-5, id='float32'), pytest.param(torch.float64, 1e-10, id='float64')]
)
def test_RMSNorm(t: Tensor, tensor_dtype: torch.dtype, tolerance: float) -> None:
	inputTensor = t.to(dtype=tensor_dtype)
	featureDimension = inputTensor.shape[-1]
	module = RMSNorm(featureDimension).to(dtype=tensor_dtype)

	resultTensor = module(inputTensor)
	expectedTensor = torch.nn.functional.normalize(inputTensor, dim=-1) * (featureDimension**0.5) * module.gamma
	resultRootMeanSquare = torch.sqrt(resultTensor.pow(2).mean(dim=-1))
	expectedRootMeanSquare = torch.ones_like(resultRootMeanSquare)

	assert tuple(module.gamma.shape) == (featureDimension,), (
		f'RMSNorm initialized gamma with shape {tuple(module.gamma.shape)}, expected {(featureDimension,)} '
		f'for input shape {tuple(inputTensor.shape)}.'
	)
	assert resultTensor.shape == inputTensor.shape, (
		f'RMSNorm returned shape {tuple(resultTensor.shape)}, expected {tuple(inputTensor.shape)} '
		f'for {tensor_dtype=} and input shape {tuple(inputTensor.shape)}.'
	)
	assert torch.allclose(resultTensor, expectedTensor, atol=tolerance, rtol=tolerance), (
		f'RMSNorm returned values {resultTensor} that do not match expected {expectedTensor} '
		f'for {tensor_dtype=}, input shape {tuple(inputTensor.shape)}, and {tolerance=}.'
	)
	assert torch.allclose(resultRootMeanSquare, expectedRootMeanSquare, atol=tolerance, rtol=tolerance), (
		f'RMSNorm produced root-mean-square values {resultRootMeanSquare}, expected {expectedRootMeanSquare} '
		f'for {tensor_dtype=}, input shape {tuple(inputTensor.shape)}, and {tolerance=}.'
	)
