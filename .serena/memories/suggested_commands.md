Preferred workflow
- Prefer MCP / Serena tools over terminal commands in this repository.
- Use Serena symbolic search/edit tools for code navigation and precise changes.
- Use MCP testing tools for pytest runs.
- Use Pylance MCP tools for syntax, diagnostics, import analysis, and signature compatibility.
- Use Context7 / Microsoft docs tools for up-to-date package and platform documentation.

Terminal fallback commands
- Sync dependencies: `uv sync`
- Run all tests: `pytest`
- Run lint checks: `ruff check .`
- Import smoke test: `python -c "import torch_einops_kit"`
- Git status: `git status`
- Git diff summary: `git diff --stat`

Windows shell reminders
- Current directory: `Get-Location`
- List files: `Get-ChildItem`
- Recursive file search: `Get-ChildItem -Recurse`
- Text search: `Select-String -Path <path> -Pattern <pattern>`

Entrypoint note
- No CLI entrypoint or `python -m torch_einops_kit` runner is declared in `pyproject.toml`.