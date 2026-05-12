from __future__ import annotations

from torch_einops_kit import compact, default, divisible_by, exists, first, identity, maybe, once, safe
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from collections.abc import Sequence
	from torch import Tensor

@pytest.mark.parametrize(
	('input_value', 'expected_output'),
	[
		pytest.param([None, None], [], id='all-none'),
		pytest.param([2, None, 3, None, 5], [2, 3, 5], id='integers-with-none'),
		pytest.param(('alpha', None, 'gamma'), ['alpha', 'gamma'], id='tuple-of-strings-with-none'),
		pytest.param([0, None, False, '', 13], [0, False, '', 13], id='falsy-non-none-preserved'),
	],
)
def test_compact(input_value: Sequence[object | None], expected_output: list[object]) -> None:
	result = compact(input_value)
	assert result == expected_output, f'compact returned {result}, expected {expected_output} for {input_value=}.'

@pytest.mark.parametrize(
	('input_value', 'fallback_value', 'expected'),
	[
		pytest.param(None, 42, 42, id='none-returns-fallback-int'),
		pytest.param(None, 'default_string', 'default_string', id='none-returns-fallback-str'),
		pytest.param(7, 42, 7, id='exists-returns-value-int'),
		pytest.param('alpha', 'default_string', 'alpha', id='exists-returns-value-str'),
	],
)
def test_default(input_value: int | str | None, fallback_value: int | str, expected: int | str) -> None:
	result = default(input_value, fallback_value)
	assert result == expected, f'default returned {result}, expected {expected} for {input_value=} and {fallback_value=}.'

@pytest.mark.parametrize(
	('numerator', 'denominator', 'expected'),
	[
		pytest.param(12, 3, True, id='int-divisible'),
		pytest.param(13, 3, False, id='int-not-divisible'),
		pytest.param(12.0, 3.0, True, id='float-divisible'),
		pytest.param(13.0, 3.0, False, id='float-not-divisible'),
		pytest.param(12, 0, False, id='zero-division'),
	],
)
def test_divisible_by(numerator: float, denominator: float, expected: bool) -> None:
	result = divisible_by(numerator, denominator)
	assert result is expected, f'divisible_by returned {result}, expected {expected} for {numerator=} and {denominator=}.'

@pytest.mark.parametrize(
	('input_value', 'expected'),
	[
		pytest.param(None, False, id='none'),
		pytest.param(0, True, id='zero'),
		pytest.param(False, True, id='boolean'),
		pytest.param('alpha', True, id='string'),
		pytest.param([], True, id='empty-list'),
	],
)
def test_exists(input_value: int | str | list[int] | None, expected: bool) -> None:
	result = exists(input_value)
	assert result is expected, f'exists returned {result}, expected {expected} for {input_value=}.'

@pytest.mark.parametrize(
	('sequence_value', 'expected_first'),
	[
		pytest.param([2, 3, 5], 2, id='list-of-int'),
		pytest.param((7, 11, 13), 7, id='tuple-of-int'),
		pytest.param(['alpha', 'beta', 'gamma'], 'alpha', id='list-of-str'),
	],
)
def test_first(sequence_value: Sequence[int] | Sequence[str], expected_first: int | str) -> None:
	result = first(sequence_value)
	assert result == expected_first, f'first returned {result}, expected {expected_first} for {sequence_value=}.'

@pytest.mark.parametrize(
	'input_value', [pytest.param(42, id='int'), pytest.param('alpha', id='string'), pytest.param([2, 3, 5], id='list')]
)
def test_identity(input_value: int | str | list[int]) -> None:
	result = identity(input_value, 'ignored_arg', kwarg='ignored')
	assert result is input_value, f'identity returned different object reference for {input_value=}.'

@pytest.mark.parametrize(
	('input_value', 'extra_arguments', 'extra_keyword_arguments', 'expected_output', 'expected_called', 'expect_identity'),
	[
		pytest.param(13, ('ignored',), {'keyword': 'value'}, 13, False, True, id='none-function-int'),
		pytest.param('alpha', (21,), {'keyword': 'marker'}, 'alpha', False, True, id='none-function-string'),
		pytest.param(None, (5,), {'multiplier': 2}, None, False, False, id='none-input-short-circuit'),
		pytest.param(21, (5,), {'multiplier': 2}, 52, True, False, id='value-input-applies-function'),
		pytest.param(None, (), {}, None, False, False, id='none-input-not-called'),
		pytest.param(8, (), {}, 11, True, False, id='value-input-called'),
	],
)
def test_maybe(
	input_value: int | str | None,
	extra_arguments: tuple[object, ...],
	extra_keyword_arguments: dict[str, object],
	expected_output: int | str | None,
	expected_called: bool,
	expect_identity: bool,
) -> None:
	class TransformProbe:
		def __init__(self) -> None:
			self.called: bool = False

		def __call__(self, value: int, *args: object, **kwargs: object) -> int:
			self.called = True
			additional = args[0] if len(args) != 0 else 3
			multiplier = kwargs.get('multiplier', 1)

			assert isinstance(additional, int), f'test_maybe expected int additional, got {additional!r} for {args=}.'
			assert isinstance(multiplier, int), f'test_maybe expected int multiplier, got {multiplier!r} for {kwargs=}.'

			return (value + additional) * multiplier

	transform_probe = TransformProbe()
	result = maybe(None if expect_identity else transform_probe)(input_value, *extra_arguments, **extra_keyword_arguments) # pyright: ignore[reportArgumentType]

	assert transform_probe.called is expected_called, (
		f'maybe call state was {transform_probe.called}, expected {expected_called} for {input_value=}, {extra_arguments=}, and {extra_keyword_arguments=}.'
	)

	if expect_identity:
		assert result is input_value, (
			f'maybe(None) returned {result}, expected identity passthrough for {input_value=} with {extra_arguments=} and {extra_keyword_arguments=}.'
		)
	else:
		assert result == expected_output, (
			f'maybe returned {result}, expected {expected_output} for {input_value=}, {extra_arguments=}, and {extra_keyword_arguments=}.'
		)

@pytest.mark.parametrize(
	('callableInput', 'expectedFirstResult'),
	[pytest.param(5, 10, id='input-five-expected-ten'), pytest.param(13, 26, id='input-thirteen-expected-twenty-six')],
)
def test_once(callableInput: int, expectedFirstResult: int) -> None:
	invocationCount: dict[str, int] = {'count': 0}

	def doubleValue(value: int) -> int:
		invocationCount['count'] += 1
		return value * 2

	wrappedDoubleValue = once(doubleValue)

	firstResult = wrappedDoubleValue(callableInput)
	secondResult = wrappedDoubleValue(callableInput)
	thirdResult = wrappedDoubleValue(callableInput)

	assert firstResult == expectedFirstResult, (
		f'once first call returned {firstResult}, expected {expectedFirstResult} for {callableInput=}.'
	)
	assert secondResult is None, f'once second call returned {secondResult}, expected None for {callableInput=}.'
	assert thirdResult is None, f'once third call returned {thirdResult}, expected None for {callableInput=}.'
	assert invocationCount['count'] == 1, (
		f'once wrapped function was invoked {invocationCount["count"]} times, expected exactly 1 for {callableInput=}.'
	)

@pytest.mark.parametrize(
	('tensor_indexes', 'leading_none_count', 'trailing_none_count'),
	[
		pytest.param((), 2, 0, id='all-none'),
		pytest.param((0,), 1, 1, id='single-active-with-sandwich-none'),
		pytest.param((0, 1), 0, 1, id='two-active-with-middle-and-trailing-none'),
		pytest.param((2, 0, 1), 1, 0, id='three-active-order-preserved'),
		pytest.param((1, 1, 2), 0, 0, id='duplicate-active-preserved'),
	],
)
def test_safe(
	tuple_t_len_3: list[tuple[Tensor, ...]],
	empty_optional_tensor_sequence: list[Tensor | None],
	tensor_indexes: tuple[int, ...],
	leading_none_count: int,
	trailing_none_count: int,
) -> None:
	active_tensors: tuple[Tensor, ...] = tuple(tuple_t_len_3[0][tensor_index] for tensor_index in tensor_indexes)
	tensors_list: list[Tensor | None] = [*empty_optional_tensor_sequence, *([None] * leading_none_count)]

	for tensor_index, tensor_value in enumerate(active_tensors):
		if tensor_index != 0:
			tensors_list.append(None)
		tensors_list.append(tensor_value)

	tensors_list.extend([None] * trailing_none_count)

	@safe
	def dummy_func(tensors_arg: Sequence[Tensor]) -> tuple[Tensor, ...]:
		return tuple(tensors_arg)

	result = dummy_func(tensors_list)

	assert len(result) == len(active_tensors), (
		f'safe-decorated function returned length {len(result)}, expected {len(active_tensors)} for {tensors_list=}.'
	)
	assert all(result_tensor is expected_tensor for result_tensor, expected_tensor in zip(result, active_tensors, strict=True)), (
		f'safe-decorated function returned tensors {result}, expected identical ordered tensors {active_tensors} for {tensors_list=}.'
	)
