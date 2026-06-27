from __future__ import annotations

from operator import is_
from torch import nn, Tensor
from torch_einops_kit.nn import count_parameters, Identity, Lambda, Residual, Sequential
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


@pytest.mark.parametrize('label', [pytest.param('residual-varied-structures', id='residual-varied-structures')])
def test_Residual_varied_structures(t: Tensor, label: str) -> None:
	"""Residual should add the original input to the *first* leaf returned by `fn`,
	preserve the overall pytree shape, accept args/kwargs, and must not mutate the input.
	"""

	# simple tensor -> tensor
	def double(x: Tensor) -> Tensor:
		return x * 2

	res = Residual(double)
	expected = t + double(t)
	out = res(t)
	assert torch.allclose(out, expected), f'{out=}, expected {expected} for {label} (simple tensor).'

	# nested pytree structure (tuple, dict, non-tensor leaf)
	def nested(x: Tensor) -> tuple[Tensor, dict[str, Tensor], str]:
		return x * 2, {'inner': x * 3}, 'marker'

	res_nested: Residual[Tensor, [], tuple[Tensor, dict[str, Tensor], str]] = Residual(nested)
	out_nested: tuple[Tensor, dict[str, Tensor], str] = res_nested(t)
	assert isinstance(out_nested, tuple)
	assert torch.allclose(out_nested[0], t * 3), f'{out_nested[0]=}, expected {t * 3} for {label} (nested first leaf).'
	assert torch.allclose(out_nested[1]['inner'], t * 3), f"{out_nested[1]['inner']=}, expected {t * 3} for {label} (nested inner)."
	assert out_nested[2] == 'marker'

	# confirm args/kwargs are forwarded
	def affine(x: Tensor, *, scale: float = 2.0) -> Tensor:
		return x * scale

	res_affine = Residual(affine)
	out_affine = res_affine(t, scale=3.0)
	assert torch.allclose(out_affine, t + t * 3.0), f'{out_affine=}, expected {t + t * 3.0} for {label} (kwargs).'

	# ensure original input is not mutated
	orig = t.clone()
	_ = res(orig)
	assert torch.equal(orig, t), f'input tensor was mutated for {label}.'


@pytest.mark.parametrize('label', [pytest.param('count-parameters-accurate-and-decorator', id='count-parameters-accurate-and-decorator')])
def test_count_parameters_accuracy_and_decorator(label: str) -> None:
	"""Validate counting, requires_grad filtering, and decorator behavior.
	"""

	class Custom(nn.Module):
		def __init__(self) -> None:
			super().__init__()
			self.l1 = nn.Linear(4, 6)
			self.extra = nn.Parameter(torch.zeros(5))
			self.register_buffer('buf', torch.ones(3))

	model = Custom()

	# compute ground-truth counts from parameters iterator (buffers are excluded)
	params = list(model.parameters())
	ground_total = sum(p.numel() for p in params)
	assert count_parameters(model) == ground_total, f'{count_parameters(model)=}, expected {ground_total} for {label} (total).'

	# toggle one parameter's requires_grad and validate filtering
	model.extra.requires_grad_(False)
	expected_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	expected_non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
	assert count_parameters(model, requires_grad=True) == expected_trainable, f'trainable mismatch for {label}.'
	assert count_parameters(model, requires_grad=False) == expected_non_trainable, f'non-trainable mismatch for {label}.'

	# test decorator attaching a property that counts all parameters
	@count_parameters
	class DecoratedAll(nn.Module):
		def __init__(self) -> None:
			super().__init__()
			self.a = nn.Linear(2, 3)
			self.b = nn.Parameter(torch.randn(1))

	inst_all = DecoratedAll()
	assert hasattr(inst_all, 'num_parameters')
	assert inst_all.num_parameters == sum(p.numel() for p in inst_all.parameters()), f'property mismatch for {label} (all).'

	# test decorator with requires_grad=True captures the kwarg
	@count_parameters(requires_grad=True)
	class DecoratedTrainable(nn.Module):
		def __init__(self) -> None:
			super().__init__()
			self.a = nn.Linear(2, 3)
			self.b = nn.Parameter(torch.randn(2))
			self.b.requires_grad_(False)

	inst_trainable = DecoratedTrainable()
	expected_trainable_inst = sum(p.numel() for p in inst_trainable.parameters() if p.requires_grad)
	assert inst_trainable.num_parameters == expected_trainable_inst, f'property mismatch for {label} (trainable).'
