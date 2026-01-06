# Contributing to TorchFX

Thank you for your interest in contributing to TorchFX! We welcome contributions from everyone, whether you're fixing a bug, adding a feature, improving documentation, or optimizing performance.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Contribution Guidelines](#contribution-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

This project follows a simple code of conduct: be respectful, collaborative, and constructive. We're all here to build something great together.

## Getting Started

Before you start contributing, please:

1. **Check existing issues** - Someone might already be working on what you have in mind
2. **Read the documentation** - Familiarize yourself with the project structure and goals
3. **Review the [Style Guide](docs/source/guides/developer/style_guide.md)** - Understand our coding standards
4. **Check the [Roadmap](docs/source/guides/developer/roadmap.md)** - See what's planned and where you can help

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Git

### Setup Instructions

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/torchfx.git
cd torchfx
```

2. **Install dependencies**

```bash
uv sync
```

3. **Install pre-commit hooks**

```bash
uv run pre-commit install
```

This ensures code quality checks run automatically before each commit.

4. **Verify your setup**

```bash
uv run pytest
```

All tests should pass before you start making changes.

## How to Contribute

### Reporting Bugs

Found a bug? Please [open an issue](https://github.com/matteospanio/torchfx/issues/new) with:

- A clear, descriptive title
- Steps to reproduce the bug
- Expected vs. actual behavior
- Your environment (OS, Python version, TorchFX version)
- Minimal code example demonstrating the issue

### Suggesting Features

Have an idea for a new feature? We encourage you to:

1. **Open an issue** to discuss the feature before implementing it
2. Explain the use case and why it would benefit TorchFX users
3. Consider if it fits within the project's scope and goals

### Contributing Code

We welcome contributions in these areas (in order of priority):

1. **Performance Improvements** - Optimize existing code, improve GPU utilization
2. **Bug Fixes** - Fix reported issues and edge cases
3. **New Features** - Add filters, effects, or core functionality
4. **Documentation** - Improve docs, add examples, write tutorials

#### When to Open an Issue First

- **Recommended for**: New features, significant refactoring, breaking changes
- **Optional for**: Bug fixes, documentation improvements, small enhancements
- **Not required for**: Typo fixes, formatting, minor doc updates

If you're unsure, feel free to open an issue to discuss your idea before investing time in implementation.

## Contribution Guidelines

### General Principles

1. **Start small** - If you're new to the project, start with a small contribution to get familiar with the workflow
2. **Focus on quality** - Well-tested, documented code is more valuable than rushed features
3. **Be patient** - Reviews take time, especially for larger contributions
4. **Communicate** - If you get stuck or need guidance, don't hesitate to ask

### Coding Standards

Please follow our [Style Guide](docs/source/guides/developer/style_guide.md), which covers:

- **Naming Conventions** - Parameter names, class names, module organization
- **Unit Conventions** - Frequency (Hz), time (seconds), gain (dB/linear)
- **Code Organization** - File structure, import ordering, class templates
- **Documentation Standards** - NumPy-style docstrings with examples

Key points:

- Use type hints for all function signatures
- Follow NumPy docstring format
- Keep functions focused and single-purpose
- Avoid breaking existing APIs (see [API Stability](docs/source/guides/api_stability.md))

## Testing

**Testing is important** - While we don't strictly follow TDD, we do require tests for new features.

### Running Tests

Run all tests:

```bash
uv run pytest
```

Run specific test file:

```bash
uv run pytest tests/test_wave.py
```

Run with coverage:

```bash
uv run pytest --cov=src/torchfx --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_butterworth_lowpass_cutoff_frequency`
- Test edge cases and error conditions
- Include docstrings explaining what the test validates
- Use pytest fixtures for common setups

Example test structure:

```python
def test_wave_from_file_loads_audio():
    """Test that Wave.from_file() correctly loads audio files."""
    wave = Wave.from_file("tests/data/test_audio.wav")
    assert wave.fs == 44100
    assert wave.channels == 2
```

### What to Test

- **New features**: Core functionality, edge cases, error handling
- **Bug fixes**: Add a test that would have caught the bug
- **Performance changes**: Add benchmark tests if applicable

Tests are a great way to ensure nothing breaks when code changes. They also serve as examples for how to use your feature.

## Documentation

**Documentation is really important** - Don't underestimate the value of describing what your code does.

### What to Document

1. **Code-level documentation**
   - Docstrings for all public classes and methods
   - Comments for complex algorithms or non-obvious logic
   - Type hints for function signatures

2. **User-facing documentation**
   - Update relevant sections in `docs/source/`
   - Add examples demonstrating your feature
   - Update API reference if adding new classes

3. **Development documentation**
   - Update CHANGELOG.md with your changes
   - Add migration notes for breaking changes
   - Update README if adding dependencies or changing setup

### Building Documentation Locally

```bash
cd docs
uv run sphinx-build -b html source build/html
```

Then open `docs/build/html/index.html` in your browser.

### Documentation Style

- Use clear, concise language
- Include code examples with expected output
- Explain *why*, not just *what*
- Link to related documentation
- Follow the existing documentation structure

## Code Style

### Formatting and Linting

We use several tools to maintain code quality:

- **ruff** - Fast Python linter and formatter
- **black** - Code formatter
- **mypy** - Static type checker (configured in the project)

**Pre-commit hooks run these automatically**, but you can also run them manually:

```bash
uv run ruff check src/ tests/
uv run black src/ tests/
```

### Pre-commit Hooks

**Please ensure you use pre-commit** before submitting a PR:

```bash
uv run pre-commit install  # One-time setup
uv run pre-commit run --all-files  # Manual run
```

Pre-commit will automatically:
- Format your code
- Check for linting issues
- Run type checks
- Validate commit messages

If pre-commit fails, fix the issues and commit again.

## Pull Request Process

### Before Submitting

1. ✅ **All tests pass** - `uv run pytest`
2. ✅ **Pre-commit hooks pass** - Code is formatted and linted
3. ✅ **Documentation updated** - If you added features or changed behavior
4. ✅ **CHANGELOG.md updated** - Add entry under "Unreleased"
5. ✅ **Branch is up to date** - Rebase on latest `main` if needed

### Submitting Your PR

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, focused commits

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request** on GitHub with:
   - **Clear title** - Summarize the change in one line
   - **Description** - Explain what, why, and how
   - **Link to issue** - If applicable (e.g., "Fixes #123")
   - **Test results** - Mention if all tests pass
   - **Breaking changes** - Clearly note any breaking changes

### PR Review Process

- **Review required** - All PRs need maintainer review
- **CI must pass** - GitHub Actions must be green
- **Exceptions** - In special cases, maintainer review may be sufficient even if CI is pending
- **Be responsive** - Address review comments promptly
- **Be patient** - Reviews may take a few days, especially for large PRs

### After Your PR is Merged

- Delete your feature branch
- Pull latest `main` to stay up to date
- Celebrate! 🎉 Your code is now part of TorchFX

## Community

### Getting Help

- **GitHub Issues** - For bug reports and feature requests
- **GitHub Discussions** - For questions, ideas, and general discussion
- **Documentation** - Check the [guides](docs/source/guides/) for detailed information

### Recognition

All contributors are recognized in our project. Your contributions, whether big or small, are valued and appreciated.

### Contact

- **Project Maintainer**: [Matteo Spanio](https://github.com/matteospanio)
- **GitHub Issues**: [torchfx/issues](https://github.com/matteospanio/torchfx/issues)

---

## Quick Checklist

Before submitting your PR, make sure:

- [ ] Code follows the [Style Guide](docs/source/guides/developer/style_guide.md)
- [ ] All tests pass (`uv run pytest`)
- [ ] Pre-commit hooks pass (`uv run pre-commit run --all-files`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes

---

Thank you for contributing to TorchFX! Your efforts help make audio DSP with PyTorch better for everyone. 🚀
