project = "TorchFX"
copyright = "2026, Matteo Spanio"
author = "Matteo Spanio"
release = "1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "ablog",
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.bibtex",
]
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"
myst_update_mathjax = False
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "posts/*/.ipynb_checkpoints/*",
    ".github/*",
    ".history",
    "github_submodule/*",
    "LICENSE.md",
    "README.md",
]


# -- Options for HTML output -------------------------------------------------
# html_title = "TorchFX"
html_logo = "_static/tfx.svg"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "logo": {
        "text": "TorchFX",
        "image_dark": "_static/tfx.svg",
        "image_light": "_static/tfx_black.svg",
    },
    # "github_url": "https://github.com/matteospanio/torchfx",
    "use_edit_page_button": False,
    "show_toc_level": 2,
    "use_edit_page_button": True,
    "navbar_align": "left",
    # "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    # "navbar_end": ["navbar-icon-links", "theme-switcher"],
    # "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 5,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/matteospanio/torchfx",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/torchfx/",
            "icon": "fa-brands fa-python",
        },
    ],
    "navigation_with_keys": True,
    # "collapse_navigation": False,
    # "show_nav_level": 2,
    # "show_prev_next": True,
    # "search_as_you_type": True,
}


html_sidebars = {
    # Blog sidebars
    # ref: https://ablog.readthedocs.io/manual/ablog-configuration-options/#blog-sidebars
    "blog/*": [
        "ablog/postcard.html",
        "ablog/recentposts.html",
        "ablog/tagcloud.html",
        "ablog/categories.html",
        "ablog/authors.html",
        "ablog/languages.html",
        "ablog/locations.html",
        "ablog/archives.html",
    ],
}

html_context = {
    "github_user": "matteospanio",
    "github_repo": "torchfx",
    "github_version": "main",
    "doc_path": "docs/source",
}


todo_include_todos = True

# -- ABlog configuration -----------------------------------------------------
blog_title = "TorchFX Blog"
blog_baseurl = "https://matteospanio.github.io/torchfx/"
blog_path = "blog/index"
blog_authors = {
    "Matteo": ("Matteo Spanio", "https://github.com/matteospanio"),
}
blog_default_author = "Matteo"
blog_feed_fulltext = True
blog_feed_length = 10
post_auto_excerpt = 1
post_auto_image = 0
fontawesome_included = True

# -- Options for autosummary/autodoc output ------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "groupwise"


# -- linkcheck options ---------------------------------------------------------

linkcheck_anchors_ignore = [
    # match any anchor that starts with a '/' since this is an invalid HTML anchor
    r"\/.*",
]

linkcheck_ignore = [
    # The crawler gets "Anchor not found" for various anchors
    r"https://github.com.+?#.*",
    r"https://www.sphinx-doc.org/en/master/*/.+?#.+?",
    # for whatever reason the Ablog index is treated as broken
    "../examples/blog/index.html",
]
