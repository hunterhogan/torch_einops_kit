"""Construct `nn.Sequential` instances and adapt callables to the `nn.Module` interface.

Contents
--------
Functions
	Sequential
		Construct an `nn.Sequential` instance from `modules`, ignoring each value that is `None`.

Classes
	Identity
		Adapt `identity` to the `nn.Module` interface.
	Lambda
		Adapt `fn` to the `nn.Module` interface.
"""

from __future__ import annotations

from torch import nn
from torch_einops_kit import compact, identity, PSpec, RVar
from typing import Generic, TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable

def Sequential(*modules: nn.Module | None) -> nn.Sequential:
	"""Construct an `nn.Sequential` instance from `modules`, ignoring each value that is `None`.

	You can call `Sequential` in place of `nn.Sequential` [1] when `modules` may contain `None`.
	`Sequential` passes `modules` to `compact` [2], then passes the resulting `list` of `nn.Module`
	instances to `nn.Sequential` [1].

	Parameters
	----------
	*modules : nn.Module | None
		Each `nn.Module` instance to place in the returned `nn.Sequential` instance. Each value that
		is `None` is discarded before construction.

	Returns
	-------
	sequential : nn.Sequential
		`nn.Sequential` instance that contains each value from `modules` that is not `None`, in
		argument order.

	See Also
	--------
	compact : Filter `None` values from `arr` and return the remaining elements as a `list`.

	References
	----------
	[1] torch.nn.Sequential - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
	[2] torch_einops_kit.compact
	"""
	return nn.Sequential(*compact(modules))

class Identity(nn.Module):
	"""Adapt `identity` to the `nn.Module` interface.

	You can instantiate `Identity` when code requires an `nn.Module` [1] instance but the required
	computation is `identity` [2]. `Identity` binds `forward` to `identity`, so `Identity.forward`
	accepts the same arguments as `identity` and returns the first positional argument unchanged.
	Unlike `torch.nn.Identity` [3], `Identity.forward` accepts additional positional arguments and
	keyword arguments.

	Attributes
	----------
	forward : staticmethod
		Class attribute that delegates `Identity.forward` to `identity` [2].

	See Also
	--------
	identity : Return `t` unchanged, ignoring all other arguments.
	torch.nn.Identity : Return the input unchanged.

	References
	----------
	[1] torch.nn.Module - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.nn.Module.html
	[2] torch_einops_kit.identity

	[3] torch.nn.Identity - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.nn.Identity.html
	"""

	forward = staticmethod(identity)
	"""Return the first positional argument unchanged and ignore each additional argument.

	You can call `Identity.forward` when `Identity` must satisfy the `nn.Module` interface and the
	required computation is `identity` [1]. `Identity.forward` delegates to `identity`, so
	`Identity.forward` returns the first positional argument unchanged and discards each additional
	positional argument and keyword argument.

	See Also
	--------
	identity : Return `t` unchanged, ignoring all other arguments.

	References
	----------
	[1] torch_einops_kit.identity
	"""

class Lambda(nn.Module, Generic[PSpec, RVar]):
	"""Adapt `fn` to the `nn.Module` interface.

	You can instantiate `Lambda` when code requires an `nn.Module` [1] instance but the required
	computation already exists as `fn`. `Lambda` stores `fn` on `self.fn`, and `Lambda.forward` passes
	each positional argument and keyword argument to `fn` without changing the argument structure.

	Attributes
	----------
	fn : Callable[PSpec, RVar]
		Callable object called by `Lambda.forward`.

	References
	----------
	[1] torch.nn.Module - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.nn.Module.html
	"""

	def __init__(self, fn: Callable[PSpec, RVar]) -> None:
		"""Store `fn` on `self.fn`.

		You can instantiate `Lambda` with `fn` when `fn` already implements the required computation.
		`Lambda.__init__` stores `fn` on `self.fn` without modifying `fn`.

		Parameters
		----------
		fn : Callable[PSpec, RVar]
			Callable object stored on `self.fn`.
		"""
		super().__init__()
		self.fn: Callable[PSpec, RVar] = fn

	def forward(self, *args: PSpec.args, **kwargs: PSpec.kwargs) -> RVar:
		"""Call `self.fn` with `args` and `kwargs`.

		You can call `Lambda.forward` to delegate `args` and `kwargs` to `self.fn`. `Lambda.forward`
		returns the value returned by `self.fn` without modifying `args` or `kwargs`.

		Parameters
		----------
		*args : PSpec.args
			Positional arguments passed to `self.fn`.
		**kwargs : PSpec.kwargs
			Keyword arguments passed to `self.fn`.

		Returns
		-------
		result : RVar
			Value returned by `self.fn`.
		"""
		return self.fn(*args, **kwargs)

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
