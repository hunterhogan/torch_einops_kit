from __future__ import annotations

from torch import nn, Tensor
from torch_einops_kit.nn import count_parameters, Identity, Lambda, Residual, Sequential
import torch

def test_sequential() -> None:
	# Test that it filters out None
	seq: nn.Sequential = Sequential(nn.Linear(10, 10), None, nn.ReLU())
	assert all(module is not None for module in seq)

	# Test forward pass
	x: Tensor = torch.randn(2, 10)
	out: Tensor = seq(x)
	assert out.shape == (2, 10)

def test_lambda() -> None:
	def fn(x: Tensor) -> Tensor:
		return x * 2

	lam: Lambda[..., Tensor] = Lambda(fn)
	x: Tensor = torch.tensor([1.0, 2.0, 3.0])
	assert torch.allclose(lam(x), torch.tensor([2.0, 4.0, 6.0]))

def test_identity() -> None:
	ident: Identity = Identity()
	x: Tensor = torch.tensor([1.0, 2.0, 3.0])
	assert torch.allclose(ident(x), x)

def test_residual() -> None:
	def fn(x: Tensor) -> Tensor:
		return x * 2
	res = Residual(fn)
	x: Tensor = torch.tensor([1., 2., 3.])
	assert torch.allclose(res(x), torch.tensor([3., 6., 9.]))

	def fn_tuple(x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
		return x * 2, x * 3, dict(a=x * 4)

	res_tuple = Residual(fn_tuple)
	out1, out2, out3 = res_tuple(x)
	assert torch.allclose(out1, torch.tensor([3., 6., 9.]))
	assert torch.allclose(out2, torch.tensor([3., 6., 9.]))
	assert torch.allclose(out3['a'], torch.tensor([4., 8., 12.]))

def test_count_parameters() -> None:
	model: nn.Linear = nn.Linear(10, 10)
	assert count_parameters(model) == 110

	# Test requires_grad filter
	model.bias.requires_grad_(False)
	assert count_parameters(model) == 110
	assert count_parameters(model, requires_grad=True) == 100
	assert count_parameters(model, requires_grad=False) == 10

	# Test as a decorator
	@count_parameters
	class MyModel(nn.Module):
		def __init__(self) -> None:
			super().__init__()
			self.linear: nn.Linear = nn.Linear(10, 10)
			self.linear.bias.requires_grad_(False)

	my_model: MyModel = MyModel()
	assert my_model.num_parameters == 110

	# Test as a decorator with kwargs
	@count_parameters(requires_grad=True)
	class MyModelTrainable(nn.Module):
		def __init__(self) -> None:
			super().__init__()
			self.linear: nn.Linear = nn.Linear(10, 10)
			self.linear.bias.requires_grad_(False)

	my_model_trainable: MyModelTrainable = MyModelTrainable()
	assert my_model_trainable.num_parameters == 100

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
