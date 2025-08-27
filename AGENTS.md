# Repository Guidelines

## Project Structure & Module Organization
- Core: `streamlit_app.py` (UI), `simple_rag.py` (RAG), `config.py` (env/config).
- Data: `policies/` (input PDFs), `faiss_index_hr/` (generated index; ignored).
- Tooling: `requirements*.txt`, `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml`, `Dockerfile*`.

## Build, Run, and Dev Commands
- `make setup-dev`: Install dev deps and create `.env` from template.
- `make serve` | `streamlit run streamlit_app.py`: Run the demo UI locally.
- `make format` | `make lint` | `make type-check`: Code quality tasks.
- Docker: `make docker-build` and `make docker-run` (mount `policies/` read-only).

## Coding Style & Naming
- Formatting: Black (88) + Isort (black profile).
- Linting: Flake8; types with mypy (relaxed for demo).
- Naming: modules `snake_case.py`; classes `CamelCase`; funcs/vars `snake_case`.

## Testing Guidelines
- Minimal POC: no test suite maintained. If adding tests, use pytest and place files under `tests/` named `test_*.py`.

## Commit & Pull Requests
- Commits: Conventional style (`feat:`, `fix:`, `chore:`). Keep changes small and focused.
- PRs: describe the change, include repro steps and screenshots of UI changes.

## Security & Configuration Tips
- Secrets: never commit `.env`; set `OPENAI_API_KEY` locally or via Streamlit secrets.
- Data: do not commit generated `faiss_index_hr/` or large PDFs. Keep `policies/` small for demos.
