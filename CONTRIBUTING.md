# Contributing to CrossMind

CrossMind welcomes contributions that improve the runtime, documentation, tests, deployment assets, or security posture.

## Development setup

1. Use Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install runtime and development dependencies:

   ```bash
   python -m pip install -r requirements.txt
   python -m pip install -r requirements-dev.txt
   ```

4. Copy `.env.example` to `.env` and set only the values required for the work being tested.
5. Run the focused test module before opening a pull request.

## Pull request checklist

- Add or update tests for behavior changes.
- Run the relevant unit and integration tests.
- Keep secrets out of source, logs, fixtures, and documentation.
- Update configuration documentation when adding environment variables.
- Keep API changes versioned and reflected in the OpenAPI documentation.
- Update the architecture, deployment, or operations guide when changing runtime boundaries.
- Use descriptive commit messages and keep changes scoped.

## Test commands

```bash
python -m pytest tests
python -m pytest tests/integration
python -m compileall app ingestion reasoning vector_store dashboard
```

The repository also contains archived legacy files. Do not treat `archive/` as active runtime code.

## Security

- Never commit API keys, tokens, passwords, private keys, or cloud credentials.
- Report suspected vulnerabilities through the project's private security channel before public disclosure.
- Run the security workflow or `scripts/security_check.py` before release.
- Keep production secrets in the deployment secret manager, not in `.env`, Kubernetes manifests, or container images.

## Code style

- Follow existing module conventions and type hints.
- Prefer explicit fallback behavior for optional external services.
- Keep structured logs free of sensitive request bodies and credentials.
- Avoid adding comments unless they explain non-obvious behavior.
