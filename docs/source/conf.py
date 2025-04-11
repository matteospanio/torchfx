from sphinx_pyproject import SphinxConfig

config = SphinxConfig("../../pyproject.toml", globalns=globals())
extensions = config["extensions"]
