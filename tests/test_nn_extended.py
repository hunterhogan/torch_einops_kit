from __future__ import annotations

from operator import is_
from torch import nn, Tensor
from torch_einops_kit.nn import Identity, Lambda, Sequential
from typing import TYPE_CHECKING
import pytest
import torch

if TYPE_CHECKING:
	from collections.abc import Callable

def _lambda_affine_with_bias_tensor(tensor_value: Tensor, bias_tensor: Tensor, *, scale: float) -> Tensor:
	return tensor_value.to(dtype=torch.float64) * scale + bias_tensor.to(dtype=torch.float64)

@pytest.fixture
def identity_call_arguments(t: Tensor) -> tuple[Tensor, tuple[Tensor, str], dict[str, str | int]]:
	extra_tensor = torch.full_like(t, 89)
	extra_args: tuple[Tensor, str] = (extra_tensor, 'north')
	extra_kwargs: dict[str, str | int] = {'marker': 'east', 'count': 13}
	return t, extra_args, extra_kwargs

@pytest.fixture
def lambda_unary_function() -> Callable[[Tensor], Tensor]:
	def transform_tensor(tensor_value: Tensor) -> Tensor:
		return tensor_value.to(dtype=torch.float64) * 2.0 + 13.0

	return transform_tensor

@pytest.fixture
def lambda_invocation_case(t: Tensor) -> tuple[Callable[..., Tensor], tuple[Tensor, Tensor], dict[str, float], Tensor]:
	bias_tensor = torch.full_like(t, 21, dtype=torch.float64)
	args: tuple[Tensor, Tensor] = (t, bias_tensor)
	kwargs: dict[str, float] = {'scale': 2.0}
	expected_tensor = _lambda_affine_with_bias_tensor(*args, **kwargs)
	return _lambda_affine_with_bias_tensor, args, kwargs, expected_tensor

@pytest.fixture
def sequential_case(t: Tensor, lambda_unary_function: Callable[[Tensor], Tensor]) -> tuple[tuple[nn.Module | None, ...], Tensor, Tensor]:
	modules: tuple[nn.Module | None, ...] = (Identity(), None, Lambda(lambda_unary_function), None)
	expected_tensor = lambda_unary_function(t)
	return modules, t, expected_tensor

@pytest.mark.parametrize('label', [pytest.param('ignores-extra-arguments', id='ignores-extra-arguments')])
def test_Identity(identity_call_arguments: tuple[Tensor, tuple[Tensor, str], dict[str, str | int]], label: str) -> None:
	identity_module = Identity()
	primary_input, extra_args, extra_kwargs = identity_call_arguments
	result = identity_module(primary_input, *extra_args, **extra_kwargs)

	assert result is primary_input, f'{id(result)=}, expected the original object id {id(primary_input)} for {label}.'
	assert torch.equal(result, primary_input), f'{result=}, expected {primary_input} for {label}.'

@pytest.mark.parametrize('label', [pytest.param('forwards-args-and-kwargs', id='forwards-args-and-kwargs')])
def test_Lambda(lambda_invocation_case: tuple[Callable[..., Tensor], tuple[Tensor, Tensor], dict[str, float], Tensor], label: str) -> None:
	wrapped_function, args, kwargs, expected_tensor = lambda_invocation_case
	lambda_module: Lambda[..., Tensor] = Lambda(wrapped_function)
	result = lambda_module(*args, **kwargs)

	assert lambda_module.fn is wrapped_function, f'{lambda_module.fn=}, expected {wrapped_function!r} for {label}.'
	assert torch.equal(result, expected_tensor), f'{result=}, expected {expected_tensor} for {label}.'

@pytest.mark.parametrize('label', [pytest.param('filters-none-and-preserves-order', id='filters-none-and-preserves-order')])
def test_Sequential(sequential_case: tuple[tuple[nn.Module | None, ...], Tensor, Tensor], label: str) -> None:
	modules, input_tensor, expected_tensor = sequential_case
	sequential_module = Sequential(*modules)
	expected = tuple(filter(None, modules))
	actual = tuple(sequential_module.children())
	result = sequential_module(input_tensor)

	assert isinstance(sequential_module, nn.Sequential), f'{type(sequential_module).__name__=}, expected Sequential for {label}.'
	assert len(actual) == len(expected), f'{len(actual)=}, expected {len(expected)} for {label}.'
	assert all(module is not None for module in actual), f'{actual=}, expected all modules to be non-None for {label}.'
	assert all(map(is_, actual, expected)), f'{label=}: {actual=}, {expected=}.'
	assert torch.equal(result, expected_tensor), f'{result=}, expected {expected_tensor} for {label}.\n'
