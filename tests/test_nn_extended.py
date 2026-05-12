from __future__ import annotations

from torch import nn, Tensor
from torch_einops_kit.nn import Identity, Lambda, Sequential
from typing import TYPE_CHECKING
import pytest
import torch

if TYPE_CHECKING:
	from collections.abc import Callable

@pytest.mark.parametrize('identity_case_label', [pytest.param('ignores-extra-arguments', id='ignores-extra-arguments')])
def test_identity_returns_first_argument_unchanged(
	identity_call_arguments: tuple[Tensor, tuple[Tensor, str], dict[str, str | int]], identity_case_label: str
) -> None:
	identity_module = Identity()
	primary_input, extra_args, extra_kwargs = identity_call_arguments
	result = identity_module(primary_input, *extra_args, **extra_kwargs)

	assert result is primary_input, (
		f'Identity returned object id {id(result)}, expected the original object id {id(primary_input)} for {identity_case_label}.'
	)
	assert torch.equal(result, primary_input), (
		f'Identity returned tensor values {result}, expected {primary_input} for {identity_case_label}.'
	)

@pytest.mark.parametrize('lambda_case_label', [pytest.param('forwards-args-and-kwargs', id='forwards-args-and-kwargs')])
def test_lambda_calls_wrapped_function(
	lambda_invocation_case: tuple[Callable[..., Tensor], tuple[Tensor, Tensor], dict[str, float], Tensor], lambda_case_label: str
) -> None:
	wrapped_function, args, kwargs, expected_tensor = lambda_invocation_case
	lambda_module: Lambda[..., Tensor] = Lambda(wrapped_function)
	result = lambda_module(*args, **kwargs)

	assert lambda_module.fn is wrapped_function, (
		f'Lambda stored callable {lambda_module.fn!r}, expected {wrapped_function!r} for {lambda_case_label}.'
	)
	assert torch.equal(result, expected_tensor), f'Lambda returned tensor {result}, expected {expected_tensor} for {lambda_case_label}.'

@pytest.mark.parametrize('sequential_case_label', [pytest.param('filters-none-and-preserves-order', id='filters-none-and-preserves-order')])
def test_sequential_filters_none_modules_and_runs_remaining_modules(
	sequential_case: tuple[tuple[nn.Module | None, ...], Tensor, Tensor], sequential_case_label: str
) -> None:
	modules, input_tensor, expected_tensor = sequential_case
	sequential_module = Sequential(*modules)
	expected_filtered_modules = tuple(module for module in modules if module is not None)
	actual_modules = tuple(sequential_module.children())
	result = sequential_module(input_tensor)

	assert isinstance(sequential_module, nn.Sequential), (
		f'Sequential returned {type(sequential_module).__name__}, expected Sequential for {sequential_case_label}.'
	)
	assert len(actual_modules) == len(expected_filtered_modules), (
		f'Sequential kept {len(actual_modules)} modules, expected {len(expected_filtered_modules)} for {sequential_case_label}.'
	)
	assert all(module is not None for module in actual_modules), (
		f'Sequential retained a None module for {sequential_case_label}: {actual_modules!r}.'
	)
	assert all(
		actual_module is expected_module for actual_module, expected_module in zip(actual_modules, expected_filtered_modules, strict=True)
	), (
		f'Sequential changed module order or instances for {sequential_case_label}: got {actual_modules!r}, expected {expected_filtered_modules!r}.'
	)
	assert torch.equal(result, expected_tensor), (
		f'Sequential returned tensor {result}, expected {expected_tensor} for {sequential_case_label}.'
	)
