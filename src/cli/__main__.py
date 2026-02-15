"""Entry point for ``python -m cli`` and the ``torchfx`` console script."""

from cli.app import app


def main() -> None:
    """Launch the TorchFX CLI application."""
    app()


if __name__ == "__main__":
    main()
