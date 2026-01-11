# Documentation

Comprehensive guide to building, contributing to, and maintaining the TorchFX documentation system.

## Overview

The TorchFX documentation is built using **Sphinx** with **MyST markdown** support, deployed automatically to GitHub Pages. The documentation system extracts API documentation from Python docstrings, supports Mermaid diagrams, and includes bibliography management via sphinxcontrib.bibtex.

```{seealso}
{doc}`/guides/developer/project-structure` - Project structure
{doc}`/guides/developer/testing` - Testing infrastructure
```

## Documentation System Architecture

```{mermaid}
graph TB
    subgraph "Source Files"
        MD["docs/source/**/*.md<br/>MyST markdown files"]
        RST["docs/source/**/*.rst<br/>reStructuredText files"]
        Python["src/torchfx/*.py<br/>Python docstrings"]
        Conf["docs/source/conf.py<br/>Sphinx configuration"]
        Bib["docs/source/references.bib<br/>Bibliography database"]
    end

    subgraph "Build Process"
        Sphinx["Sphinx build process<br/>sphinx-build"]
        Autodoc["autodoc extension<br/>Extracts docstrings"]
        MyST["myst-parser<br/>Markdown support"]
        Mermaid["sphinxcontrib-mermaid<br/>Diagram rendering"]
        BibTeX["sphinxcontrib-bibtex<br/>Bibliography processing"]
        Make["docs/Makefile<br/>make html command"]
    end

    subgraph "GitHub Actions Workflow"
        Trigger["Push to master branch<br/>or workflow_dispatch"]
        Checkout["actions/checkout@v4<br/>Clone repository"]
        UV["Install uv and sync deps<br/>uv sync --all-groups"]
        Build["Build HTML<br/>cd docs && uv run make html"]
        NoJekyll["Create .nojekyll file<br/>Disable Jekyll processing"]
        Deploy["peaceiris/actions-gh-pages@v4<br/>Deploy to gh-pages branch"]
    end

    subgraph "Output"
        HTML["docs/build/html/*<br/>Generated HTML files"]
        GHPages["gh-pages branch<br/>Deployed documentation"]
        Site["GitHub Pages site<br/>https://matteospanio.github.io/torchfx/"]
    end

    MD --> MyST
    RST --> Sphinx
    Python --> Autodoc
    Conf --> Sphinx
    Bib --> BibTeX
    MyST --> Sphinx
    Autodoc --> Sphinx
    Mermaid --> Sphinx
    BibTeX --> Sphinx
    Make --> Sphinx
    Sphinx --> HTML

    Trigger --> Checkout
    Checkout --> UV
    UV --> Build
    Build --> Sphinx
    Sphinx --> NoJekyll
    NoJekyll --> Deploy
    Deploy --> GHPages
    GHPages --> Site
```

## Documentation Structure

### Directory Layout

```
docs/
├── source/
│   ├── conf.py                    # Sphinx configuration
│   ├── index.md                   # Homepage
│   ├── api.rst                    # API reference
│   ├── references.bib             # Bibliography database
│   ├── guides/
│   │   ├── index.md
│   │   ├── getting-started/       # Getting started guides
│   │   ├── core-concepts/         # Core concepts
│   │   ├── tutorials/             # Step-by-step tutorials
│   │   ├── advanced/              # Advanced topics
│   │   └── developer/             # Developer documentation
│   ├── blog/                      # Blog posts
│   └── _static/                   # Static assets
├── build/                         # Generated HTML (git-ignored)
├── Makefile                       # Build automation (Unix)
└── make.bat                       # Build automation (Windows)
```

### Documentation Types

| Type | Location | Purpose | Format |
|------|----------|---------|--------|
| **API Reference** | `api.rst` | Auto-generated from docstrings | reStructuredText |
| **Getting Started** | `guides/getting-started/` | Installation and quickstart | MyST markdown |
| **Core Concepts** | `guides/core-concepts/` | Fundamental concepts | MyST markdown |
| **Tutorials** | `guides/tutorials/` | Step-by-step guides | MyST markdown |
| **Advanced Topics** | `guides/advanced/` | In-depth technical guides | MyST markdown |
| **Developer Docs** | `guides/developer/` | Contributor documentation | MyST markdown |
| **Blog** | `blog/` | Release notes and articles | MyST markdown |

## Building Documentation Locally

### Prerequisites

Install all dependencies including documentation tools:

```bash
# Sync all dependency groups (includes docs dependencies)
uv sync --all-groups
```

### Build Commands

```bash
# Navigate to docs directory
cd docs

# Build HTML documentation
uv run make html

# Clean build artifacts
uv run make clean

# Rebuild from scratch
uv run make clean html
```

### View Generated Documentation

```bash
# On Linux/macOS
open build/html/index.html

# On Windows
start build/html/index.html

# Or use Python's HTTP server
cd build/html
python -m http.server 8000
# Visit http://localhost:8000
```

### Available Make Targets

| Target | Command | Purpose |
|--------|---------|---------|
| `html` | `make html` | Build HTML documentation |
| `clean` | `make clean` | Remove build artifacts |
| `linkcheck` | `make linkcheck` | Check for broken links |
| `help` | `make help` | Show available targets |

## Docstring Style Guide

TorchFX uses **NumPy-style docstrings** for consistency and compatibility with Sphinx's Napoleon extension.

### Class Docstrings

```python
class MyEffect(FX):
    """Brief one-line description.

    Longer description providing more context about the effect,
    its purpose, and typical use cases.

    Parameters
    ----------
    param1 : float
        Description of param1, including units and valid range.
    param2 : int, optional
        Description of param2. Default is 10.
    strategy : MyStrategy, optional
        Strategy for processing. If None, uses default strategy.

    Attributes
    ----------
    computed_value : torch.Tensor
        Description of computed attribute.

    Examples
    --------
    Basic usage:

    >>> effect = MyEffect(param1=1.5, param2=20)
    >>> wave = Wave.from_file("audio.wav")
    >>> processed = wave | effect

    See Also
    --------
    RelatedEffect : Related effect class
    OtherEffect : Another related effect

    Notes
    -----
    Additional implementation details, mathematical formulas,
    or important considerations.

    References
    ----------
    .. [1] Author, "Title", Journal, Year.
    """
```

### Method Docstrings

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply effect to input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input audio tensor with shape (channels, samples) or
        (batch, channels, samples).

    Returns
    -------
    torch.Tensor
        Processed audio tensor with same shape as input.

    Raises
    ------
    ValueError
        If input tensor has invalid shape or dtype.
    AssertionError
        If required parameters are not configured.

    Examples
    --------
    >>> effect = MyEffect(param=1.5)
    >>> x = torch.randn(2, 44100)
    >>> y = effect(x)
    >>> y.shape
    torch.Size([2, 44100])
    """
```

### Property Docstrings

```python
@property
def delay_samples(self) -> int:
    """Number of delay samples.

    Computed from BPM, delay_time, and sample rate.

    Returns
    -------
    int
        Delay length in samples.

    Raises
    ------
    AssertionError
        If sample rate (fs) is not configured.
    """
```

### Docstring Sections

| Section | Purpose | Required |
|---------|---------|----------|
| **Brief description** | One-line summary | Yes |
| **Extended description** | Detailed explanation | Recommended |
| **Parameters** | Function/method arguments | For functions/methods |
| **Returns** | Return value description | For functions/methods |
| **Yields** | For generators | For generators only |
| **Raises** | Exceptions that may be raised | If applicable |
| **Attributes** | Class attributes | For classes |
| **Examples** | Usage examples | Highly recommended |
| **See Also** | Related functions/classes | Recommended |
| **Notes** | Implementation details | If applicable |
| **References** | Academic citations | If applicable |

## Writing MyST Markdown

### Basic Syntax

TorchFX documentation uses MyST (Markedly Structured Text) markdown, which extends standard markdown with Sphinx capabilities.

#### Headers

```markdown
# Page Title

Brief introduction to the page.

## Section Header

Content for this section.

### Subsection Header

More specific content.
```

#### Code Blocks

```markdown
Standard Python code block:

\`\`\`python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")
processed = wave | fx.effect.Gain(gain=2.0)
\`\`\`

With syntax highlighting and line numbers:

\`\`\`{code-block} python
:linenos:
:emphasize-lines: 3

import torchfx as fx

wave = fx.Wave.from_file("audio.wav")  # Highlighted line
processed = wave | fx.effect.Gain(gain=2.0)
\`\`\`
```

#### Lists

```markdown
Unordered list:
- Item 1
- Item 2
  - Nested item
  - Another nested item
- Item 3

Ordered list:
1. First step
2. Second step
3. Third step

Definition list:
**Term 1**
: Definition of term 1

**Term 2**
: Definition of term 2
```

#### Tables

```markdown
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Row 1    | Data     | More data|
| Row 2    | Data     | More data|

Or with alignment:

| Left | Center | Right |
|:-----|:------:|------:|
| A    | B      | C     |
| D    | E      | F     |
```

### Cross-References

#### Referencing Documentation Pages

```markdown
See {doc}`/guides/core-concepts/wave` for Wave documentation.

Link to tutorial: {doc}`/guides/tutorials/custom-effects`

Relative reference: {doc}`../tutorials/bpm-delay`
```

#### Referencing Classes and Functions

```markdown
The {class}`~torchfx.Wave` class is the main container.

Use {func}`~torchfx.Wave.from_file` to load audio.

Reference with custom text: {class}`Wave class <torchfx.Wave>`

Module reference: {mod}`torchfx.effect`

Method reference: {meth}`torchfx.Wave.to`
```

**Tilde prefix** (`~`): Shows only the last component (e.g., `Wave` instead of `torchfx.Wave`)

#### Referencing Parameters and Attributes

```markdown
The {attr}`~torchfx.Wave.fs` attribute stores sample rate.

Parameter {paramref}`gain` controls amplification.
```

### Admonitions

Use admonitions to highlight important information:

```markdown
\`\`\`{note}
This is a note with general information.
\`\`\`

\`\`\`{warning}
⚠️ This is a warning about potential issues.
\`\`\`

\`\`\`{danger}
🛑 This is a danger notice for critical issues.
\`\`\`

\`\`\`{tip}
💡 This is a helpful tip.
\`\`\`

\`\`\`{seealso}
{doc}`/guides/tutorials/custom-effects` - Creating custom effects
{doc}`/guides/advanced/performance` - Performance optimization
\`\`\`

\`\`\`{versionadded} 0.2.0
This feature was added in version 0.2.0.
\`\`\`

\`\`\`{versionchanged} 0.3.0
Behavior changed in version 0.3.0.
\`\`\`

\`\`\`{deprecated} 0.4.0
This feature is deprecated and will be removed in version 0.5.0.
Use {class}`~torchfx.NewClass` instead.
\`\`\`
```

### Mathematical Formulas

Use LaTeX syntax for mathematical formulas:

```markdown
Inline math: $f(x) = x^2 + 2x + 1$

Display math:

$$
\text{samples} = \frac{n}{d} \times m \times \frac{60}{BPM} \times f_s
$$

Multi-line equations:

$$
\begin{aligned}
y[n] &= x[n] + \alpha \cdot x[n-1] \\
\alpha &= e^{-1/(f_s \tau)}
\end{aligned}
$$
```

### Mermaid Diagrams

TorchFX uses sphinxcontrib-mermaid for diagrams:

```markdown
\`\`\`{mermaid}
graph LR
    Input[Audio Input] --> Effect[Apply Effect]
    Effect --> Output[Audio Output]

    style Input fill:#e1f5ff
    style Output fill:#e1f5ff
    style Effect fill:#e8f5e1
\`\`\`

\`\`\`{mermaid}
sequenceDiagram
    participant User
    participant Wave
    participant Effect

    User->>Wave: Load audio
    User->>Effect: Create effect
    Wave->>Effect: Apply using |
    Effect->>User: Return processed audio
\`\`\`
```

**Common diagram types**:
- `graph TB`: Top-to-bottom flowchart
- `graph LR`: Left-to-right flowchart
- `sequenceDiagram`: Sequence diagram
- `classDiagram`: Class diagram
- `stateDiagram`: State diagram

### Bibliography and Citations

#### Adding Bibliography Entries

Edit `docs/source/references.bib`:

```bibtex
@article{smith2020audio,
  title={Audio Processing with Neural Networks},
  author={Smith, John and Doe, Jane},
  journal={Journal of Audio Engineering},
  volume={68},
  number={4},
  pages={123--145},
  year={2020},
  publisher={Audio Society}
}

@book{oppenheim1999discrete,
  title={Discrete-Time Signal Processing},
  author={Oppenheim, Alan V and Schafer, Ronald W},
  year={1999},
  publisher={Prentice Hall},
  edition={2nd}
}
```

#### Citing References

In markdown files:

```markdown
According to the literature {cite}`smith2020audio`, neural networks
can be effective for audio processing.

Multiple citations: {cite}`smith2020audio,oppenheim1999discrete`

Narrative citation: {cite:t}`smith2020audio` demonstrated that...
```

#### Bibliography Section

At the end of your document:

```markdown
## References

\`\`\`{bibliography}
:filter: docname in docnames
:style: alpha
\`\`\`
```

**Bibliography styles**:
- `alpha`: Author-year style (e.g., [Smi20])
- `plain`: Numbered style (e.g., [1])
- `unsrt`: Unsorted numbered style

## Adding Mermaid Diagrams

### Diagram Best Practices

#### Flowcharts

```markdown
\`\`\`{mermaid}
graph TB
    Start[Start] --> Process1[Process Step]
    Process1 --> Decision{Decision?}
    Decision -->|Yes| Process2[Action A]
    Decision -->|No| Process3[Action B]
    Process2 --> End[End]
    Process3 --> End

    style Start fill:#e1f5ff
    style End fill:#e1f5ff
    style Decision fill:#fff5e1
\`\`\`
```

#### Sequence Diagrams

```markdown
\`\`\`{mermaid}
sequenceDiagram
    participant User
    participant Wave
    participant Filter
    participant GPU

    User->>Wave: Load audio
    User->>Filter: Create filter
    User->>Wave: wave | filter

    Wave->>Filter: Apply filter
    Filter->>GPU: Transfer to GPU
    GPU->>Filter: Process on GPU
    Filter->>GPU: Transfer back
    Filter->>Wave: Return result
    Wave->>User: Processed audio
\`\`\`
```

#### Component Diagrams

```markdown
\`\`\`{mermaid}
graph TB
    subgraph "TorchFX Core"
        Wave[Wave Class]
        FX[FX Base Class]
    end

    subgraph "Effects"
        Gain[Gain]
        Normalize[Normalize]
        Reverb[Reverb]
        Delay[Delay]
    end

    subgraph "Filters"
        IIR[IIR Filters]
        FIR[FIR Filters]
    end

    FX --> Gain
    FX --> Normalize
    FX --> Reverb
    FX --> Delay
    FX --> IIR
    FX --> FIR

    Wave -.->|uses| FX
\`\`\`
```

### Diagram Styling

```markdown
\`\`\`{mermaid}
graph LR
    A[Node A] --> B[Node B]
    B --> C[Node C]

    style A fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    style B fill:#e8f5e1,stroke:#00aa00,stroke-width:2px
    style C fill:#ffe1e1,stroke:#cc0000,stroke-width:2px
\`\`\`
```

**Color scheme**:
- Blue (`#e1f5ff`): Input/Output
- Green (`#e8f5e1`): Processing steps
- Yellow (`#fff5e1`): Decisions
- Red (`#ffe1e1`): Errors/Warnings

## API Documentation

### API Reference Structure

The API reference is defined in `docs/source/api.rst`:

```rst
API Reference
=============

Wave Class
----------

.. autoclass:: torchfx.Wave
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__, __call__

FX Base Class
-------------

.. autoclass:: torchfx.FX
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__, __call__

Filter Module
-------------

.. automodule:: torchfx.filter
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__, __call__

Effect Module
-------------

.. automodule:: torchfx.effect
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__, __call__
```

### Autodoc Directives

| Directive | Purpose | Example |
|-----------|---------|---------|
| `autoclass` | Document a class | `.. autoclass:: torchfx.Wave` |
| `autofunction` | Document a function | `.. autofunction:: torchfx.process_audio` |
| `automodule` | Document entire module | `.. automodule:: torchfx.effect` |
| `automethod` | Document a method | `.. automethod:: Wave.from_file` |
| `autoattribute` | Document an attribute | `.. autoattribute:: Wave.fs` |

### Autodoc Options

| Option | Purpose |
|--------|---------|
| `:members:` | Include all members |
| `:members: method1, method2` | Include specific members |
| `:exclude-members: __init__` | Exclude specific members |
| `:show-inheritance:` | Show base classes |
| `:inherited-members:` | Include inherited members |
| `:undoc-members:` | Include undocumented members |
| `:private-members:` | Include private members (_method) |
| `:special-members:` | Include special members (__method__) |

## Documentation Testing

### Doctest

Test code examples in docstrings:

```python
def my_function(x):
    """Add one to input.

    Examples
    --------
    >>> my_function(5)
    6
    >>> my_function(10)
    11
    """
    return x + 1
```

Run doctests:

```bash
# Test specific module
python -m doctest src/torchfx/effect.py -v

# Test all modules
pytest --doctest-modules src/
```

### Link Checking

Check for broken links:

```bash
cd docs
uv run make linkcheck
```

**Output**: `build/linkcheck/output.txt` contains broken links and their status.

## GitHub Actions Deployment

### Workflow Configuration

The documentation is automatically deployed via `.github/workflows/docs.yml`:

```yaml
name: Deploy Sphinx documentation to GitHub Pages

on:
  push:
    branches: [master]
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Install uv
        run: curl -Ls https://astral.sh/uv/install.sh | sh

      - name: Sync dependencies
        run: uv sync --all-groups

      - name: Build documentation
        run: cd docs && uv run make html

      - name: Create .nojekyll file
        run: touch docs/build/html/.nojekyll

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/build/html
```

### Workflow Execution

```{mermaid}
sequenceDiagram
    participant Trigger as Workflow Trigger
    participant Runner as GitHub Runner
    participant UV as UV Package Manager
    participant Sphinx as Sphinx Build
    participant Deploy as GH Pages Action
    participant Pages as GitHub Pages

    Trigger->>Runner: Push to master
    Runner->>Runner: Checkout repository<br/>actions/checkout@v4

    Runner->>UV: Install uv
    Note over Runner,UV: curl -Ls https://astral.sh/uv/install.sh | sh

    UV->>UV: Sync dependencies
    Note over UV: uv sync --all-groups

    Runner->>Sphinx: Build documentation
    Note over Sphinx: cd docs && uv run make html

    Sphinx->>Sphinx: Generate HTML
    Sphinx->>Runner: Create .nojekyll file
    Note over Runner: touch docs/build/html/.nojekyll

    Runner->>Deploy: Trigger deployment
    Deploy->>Deploy: Publish directory<br/>docs/build/html
    Deploy->>Pages: Push to gh-pages branch
    Pages->>Pages: Update live site
```

### .nojekyll File

The `.nojekyll` file is critical—it prevents GitHub Pages from running Jekyll processing:

```bash
touch docs/build/html/.nojekyll
```

**Why?** Jekyll processing can interfere with Sphinx's generated structure, causing broken links and missing files.

## Contributing to Documentation

### Workflow for Documentation Changes

1. **Create/edit documentation files**:
   ```bash
   # Edit existing file
   vim docs/source/guides/tutorials/my-tutorial.md

   # Or create new file
   touch docs/source/guides/tutorials/new-tutorial.md
   ```

2. **Build locally to verify**:
   ```bash
   cd docs
   uv run make clean html
   ```

3. **Check for errors**:
   - Fix any warnings from Sphinx
   - Verify cross-references work
   - Check Mermaid diagrams render correctly

4. **View in browser**:
   ```bash
   open build/html/index.html
   ```

5. **Commit and push**:
   ```bash
   git add docs/source/guides/tutorials/new-tutorial.md
   git commit -m "docs: Add new tutorial on XYZ"
   git push
   ```

6. **Automatic deployment**: Documentation automatically builds and deploys to GitHub Pages

### Adding a New Guide

1. **Create the markdown file**:
   ```bash
   touch docs/source/guides/tutorials/my-guide.md
   ```

2. **Add frontmatter and content**:
   ```markdown
   # My Guide Title

   Brief introduction explaining what this guide covers.

   ## Overview

   ...
   ```

3. **Add to table of contents**:

   Edit `docs/source/guides/tutorials/index.md`:
   ```markdown
   \`\`\`{toctree}
   :maxdepth: 2

   existing-guide
   my-guide
   \`\`\`
   ```

4. **Build and verify**:
   ```bash
   cd docs && uv run make html
   ```

### Documentation Checklist

Before submitting documentation:

- ✅ All code examples are tested and work
- ✅ Cross-references use correct syntax (`:doc:`, `:class:`, etc.)
- ✅ Mermaid diagrams render correctly
- ✅ No Sphinx warnings or errors
- ✅ Links are not broken (run `make linkcheck`)
- ✅ Mathematical formulas render correctly
- ✅ Bibliography citations are valid
- ✅ File is added to appropriate `toctree`

## Best Practices

### Writing Style

- **Be concise**: Use clear, direct language
- **Be consistent**: Follow existing style and terminology
- **Be complete**: Cover edge cases and gotchas
- **Be helpful**: Provide examples and use cases

### Code Examples

```markdown
\`\`\`python
# ✅ GOOD: Complete, runnable example
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")
processed = wave | fx.effect.Gain(gain=2.0)
processed.save("output.wav")
\`\`\`

\`\`\`python
# ❌ BAD: Incomplete, can't run
wave | Gain(2.0)  # Where does wave come from?
\`\`\`
```

### Cross-References

```markdown
# ✅ GOOD: Use semantic references
The {class}`~torchfx.Wave` class provides audio I/O.
See {doc}`/guides/tutorials/custom-effects` for details.

# ❌ BAD: Hard-coded links
See `tutorials/custom-effects.md` for details.
```

### Diagrams

```markdown
# ✅ GOOD: Clear, focused diagram
\`\`\`{mermaid}
graph LR
    A[Input] --> B[Process]
    B --> C[Output]
\`\`\`

# ❌ BAD: Overly complex diagram
\`\`\`{mermaid}
graph TB
    A[Input] --> B[Step 1]
    B --> C[Step 2]
    C --> D[Step 3]
    D --> E[Step 4]
    E --> F[Step 5]
    F --> G[Step 6]
    G --> H[Step 7]
    H --> I[Output]
\`\`\`
```

## Related Resources

- {doc}`/guides/developer/project-structure` - Project organization
- {doc}`/guides/developer/style_guide` - Coding standards
- {doc}`/guides/developer/testing` - Testing documentation
- [Sphinx Documentation](https://www.sphinx-doc.org/) - Official Sphinx docs
- [MyST Parser](https://myst-parser.readthedocs.io/) - MyST markdown syntax
- [Mermaid Documentation](https://mermaid-js.github.io/) - Mermaid diagram syntax
