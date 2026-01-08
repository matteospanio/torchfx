# TorchFX Documentation Improvement Plan - TODO List

## Overview

Integrate AI-generated wiki documentation into the official Sphinx documentation, creating comprehensive tutorials and improving developer guides while preserving valuable mermaid diagrams and architectural content.

## User Preferences

- **Diagrams**: Keep most mermaid diagrams (recommended for educational value)
- **API Content**: Extract examples to tutorials, keep autodoc for API reference
- **Dev Content**: Merge with existing developer docs (guides/developer/)
- **Tutorial Priority**:
  1. Creating custom effects/filters
  2. Pipeline composition patterns
- **Citations**: Use sphinxcontrib.bibtex for bibliographic references and citations (installed)

## Quick Summary - Task Counts by Phase

- **Phase 0 - Setup Bibliography** (FIRST): ✅ **COMPLETED** - 3 main tasks (with 11 subtasks)
- **Phase 1 - Core Concepts**: ✅ **COMPLETED** - 6 main tasks (with 24 subtasks)
- **Phase 2 - Priority Tutorials** ⭐: 6 main tasks (with 66 subtasks) - **1/6 COMPLETED**
- **Phase 3 - Advanced Usage**: 5 main tasks (with 50 subtasks)
- **Phase 4 - Developer Docs**: 6 main tasks (with 37 subtasks)
- **Phase 5 - Enhance Existing**: 2 main tasks (with 15 subtasks)
- **Phase 6 - Docstrings**: 5 main tasks (with 35 subtasks)
- **Validation Checklist**: 7 validation tasks

**Total: 33 main tasks, 245 subtasks**
**Completed: 10 main tasks (30%), 51 subtasks (21%)**

**Files created**: 7 files (references.bib, glossary.rst, custom-effects.md, core-concepts/index.md, wave.md, fx.md, pipeline-operator.md, type-system.md)
**Files to create**: 19 remaining files (4 tutorials + 5 advanced + 4 developer + 6 indexes)
**Files to update**: 8 existing files (8 docs)

---

## Documentation Structure Analysis

### Wiki Content Mapping

| Wiki Section | Content Type | Target Location | Action |
|--------------|--------------|-----------------|--------|
| 1 Overview | Introduction | index.rst | Already migrated |
| 1.1 Quick Start | Getting started | guides/getting-started/ | Review & enhance |
| 1.2 Installation | Setup guide | guides/getting-started/ | Review & enhance |
| 2 Core Concepts | Architecture | guides/core-concepts/ | **NEW** - Create section |
| 2.1 Wave Class | Core docs | guides/core-concepts/wave.rst | **NEW** |
| 2.2 FX Base Class | Core docs | guides/core-concepts/fx.rst | **NEW** |
| 2.3 Pipeline Operator | Core docs | guides/core-concepts/pipeline-operator.rst | **NEW** |
| 2.4 Type System | Core docs | guides/core-concepts/type-system.rst | **NEW** |
| 3 Effects | User guide | Existing API docs | Extract examples only |
| 3.5 Creating Custom Effects | **PRIORITY** Tutorial | guides/tutorials/custom-effects.rst | **NEW** |
| 4 Filters | User guide | Existing API docs | Extract examples only |
| 4.3 Parallel Filter Combination | **PRIORITY** Tutorial | guides/tutorials/parallel-filters.rst | **NEW** |
| 4.4 Creating Custom Filters | **PRIORITY** Tutorial | guides/tutorials/custom-filters.rst | **NEW** |
| 5 Advanced Usage | User guide | guides/advanced/ | **NEW** section |
| 5.1 Multi-Channel | Advanced guide | guides/advanced/multi-channel.rst | **NEW** |
| 5.2 GPU Acceleration | Advanced guide | guides/advanced/gpu-acceleration.rst | **NEW** |
| 5.3 PyTorch Integration | Advanced guide | guides/advanced/pytorch-integration.rst | **NEW** |
| 6 Examples | Code examples | guides/tutorials/ | Integrate into tutorials |
| 6.1 Series & Parallel | **PRIORITY** Tutorial | guides/tutorials/series-parallel-filters.rst | **NEW** |
| 6.2 Multi-Channel Example | Tutorial | guides/tutorials/multi-channel-effects.rst | **NEW** |
| 6.3 BPM-Synced Delay | Tutorial | guides/tutorials/bpm-delay.rst | **NEW** |
| 7 Development Guide | Dev docs | guides/developer/ | Merge with existing |
| 7.1 Project Structure | Dev guide | guides/developer/project-structure.rst | **NEW** |
| 7.2 Build & Dependencies | Dev guide | CONTRIBUTING.md | Update |
| 7.3 Development Workflow | Dev guide | CONTRIBUTING.md | Already exists |
| 7.4 Testing | Dev guide | guides/developer/testing.rst | **NEW** |
| 7.5 Benchmarking | Dev guide | guides/developer/benchmarking.rst | **NEW** |
| 7.6 Documentation | Dev guide | guides/developer/documentation.rst | **NEW** |
| 8 API Reference | API docs | api/ | Already exists (autodoc) |
| 9 Performance | Performance guide | guides/advanced/performance.rst | **NEW** |

## Implementation Plan - TODO Tasks

### Phase 0: Setup Bibliography (FIRST) ✅ COMPLETED

**Tasks:**

- [x] Create `docs/source/references.bib` for bibliographic entries
  - [x] Add BibTeX entries for design patterns references
  - [x] Add BibTeX entries for DSP/filter design theory
  - [x] Add BibTeX entries for audio processing concepts
  - [x] Add BibTeX entries for GPU computing best practices
  - [x] Ensure proper BibTeX formatting

- [x] Update `docs/source/conf.py` to configure bibtex extension
  - [x] Verify sphinxcontrib.bibtex is in extensions list
  - [x] Configure bibtex_bibfiles to point to references.bib
  - [x] Set bibtex_reference_style (e.g., 'author_year' or 'label')

- [x] Create `docs/source/glossary.rst` with common terms (BONUS)
  - [x] Add glossary entries for DSP terms
  - [x] Add glossary entries for TorchFX-specific terms
  - [x] Link to external resources (Wikipedia, PyTorch docs, etc.)
  - [x] Add to main index.rst toctree

### Phase 1: Core Concepts Section (NEW) ✅ COMPLETED

Create `docs/source/guides/core-concepts/` with architectural documentation.

**Tasks:**

- [x] Create `docs/source/guides/core-concepts/index.md` - Overview with architecture diagram
  - [x] Add architecture overview mermaid diagram
  - [x] Write introduction explaining core concepts scope
  - [x] Add toctree linking to all core concept pages
  - [x] Include cross-references to tutorials

- [x] Create `docs/source/guides/core-concepts/wave.md` - Wave class deep-dive
  - [x] Document Wave tensor structure and shape conventions
  - [x] Explain immutability patterns and why Wave is immutable
  - [x] Add device-aware processing examples
  - [x] Include mermaid diagrams for Wave operations
  - [x] Add executable code examples
  - [x] Link to Wave API reference

- [x] Create `docs/source/guides/core-concepts/fx.md` - FX base class and inheritance
  - [x] Document FX base class architecture
  - [x] Add class hierarchy mermaid diagram
  - [x] Explain sample rate auto-configuration (fs parameter)
  - [x] Document Filter vs Effect patterns
  - [x] Add executable inheritance examples
  - [x] Link to API reference for FX classes

- [x] Create `docs/source/guides/core-concepts/pipeline-operator.md` - Pipe operator mechanics
  - [x] Explain pipe operator (|) syntax and semantics
  - [x] Add sequence diagram for pipeline execution
  - [x] Document automatic configuration flow
  - [x] Show pipeline composition examples
  - [x] Add mermaid diagram for data flow
  - [x] Link to pipeline tutorials

- [x] Create `docs/source/guides/core-concepts/type-system.md` - Type aliases and conventions
  - [x] Document MusicalTime, Second, Millisecond, Decibel types
  - [x] Explain type conventions and conversions
  - [x] Add table of musical time divisions and their meanings
  - [x] Include usage examples for each type
  - [x] Link to typing module API reference

- [x] Update `docs/source/guides/index.md` to include core-concepts in toctree

### Phase 2: Priority Tutorials (NEW) ⭐

Create practical, executable tutorials based on wiki examples.

**Tasks:**

- [x] Create `docs/source/guides/tutorials/custom-effects.md` (PRIORITY) ✅ COMPLETED
  - [x] Read wiki/3.5 Creating Custom Effects.md as source
  - [x] Write basic custom effect template section
  - [x] Document parameter validation patterns
  - [x] Explain sample rate handling (fs auto-configuration)
  - [x] Document strategy pattern for extensibility
  - [x] Add multi-channel effect example (ComplexEffect)
  - [x] Include complete working example: MultiTapDelay
  - [x] Add mermaid diagram: FX inheritance hierarchy
  - [x] Add mermaid diagram: Strategy pattern architecture
  - [x] Add mermaid diagram: Multi-channel architecture
  - [x] Ensure all code examples are executable
  - [x] Add citations for design patterns (gamma1994design)
  - [x] Link to core concepts and API reference
  - [x] Add external links (PyTorch, Wikipedia, DAFX, CCRMA Stanford)
  - [x] Link to glossary terms throughout the document
  - [x] Convert to Markdown (MyST) format for consistency

- [ ] Create `docs/source/guides/tutorials/series-parallel-filters.md` (PRIORITY)
  - [ ] Read wiki/6.1, wiki/4.3, wiki/2.3 as sources
  - [ ] Document sequential chaining with pipe operator (|)
  - [ ] Explain parallel combination with add operator (+)
  - [ ] Show combined series/parallel patterns
  - [ ] Document automatic configuration flow
  - [ ] Include real example from examples/series_and_parallel_filters.py
  - [ ] Add mermaid diagram: Pipeline flow visualization
  - [ ] Add mermaid diagram: Parallel combination architecture
  - [ ] Ensure all code examples are executable
  - [ ] Link to pipeline operator core concepts
  - [ ] Add performance notes

- [ ] Create `docs/source/guides/tutorials/custom-filters.md` (PRIORITY)
  - [ ] Read wiki/4.4 as source
  - [ ] Document AbstractFilter inheritance pattern
  - [ ] Explain coefficient computation pattern
  - [ ] Document IIR vs FIR custom implementations
  - [ ] Show integration with pipeline operator
  - [ ] Add mermaid diagram: Filter class hierarchy
  - [ ] Include executable custom filter examples
  - [ ] Add citations for filter design theory if applicable
  - [ ] Link to filter API reference

- [ ] Create `docs/source/guides/tutorials/multi-channel-effects.md`
  - [ ] Read wiki/6.2 and wiki/5.1 as sources
  - [ ] Document per-channel processing patterns
  - [ ] Explain nn.ModuleList usage for multi-channel
  - [ ] Provide ComplexEffect walkthrough
  - [ ] Show channel-specific processing chains
  - [ ] Include code from examples/multi_channel_effect.py
  - [ ] Add mermaid diagram for multi-channel architecture
  - [ ] Ensure examples are executable
  - [ ] Link to multi-channel advanced guide

- [ ] Create `docs/source/guides/tutorials/bpm-delay.md`
  - [ ] Read wiki/6.3 and delay-related sections as sources
  - [ ] Document MusicalTime type usage
  - [ ] Explain BPM synchronization mechanism
  - [ ] Document delay strategies (Mono, PingPong, etc.)
  - [ ] Add musical time divisions reference table
  - [ ] Include complete delay effect example
  - [ ] Ensure code is executable
  - [ ] Link to type system core concepts

- [ ] Update `docs/source/guides/tutorials/index.md` with new tutorials in toctree

### Phase 3: Advanced Usage Section (NEW)

Create `docs/source/guides/advanced/` for advanced topics.

**Tasks:**

- [ ] Create `docs/source/guides/advanced/index.md` - Advanced topics overview
  - [ ] Write introduction to advanced features
  - [ ] Add toctree linking all advanced topics
  - [ ] Include navigation hints for readers

- [ ] Create `docs/source/guides/advanced/gpu-acceleration.md`
  - [ ] Read wiki/5.2 GPU Acceleration.md as source
  - [ ] Document device management (Wave.to(), device property)
  - [ ] Explain filter/effect device transfer mechanisms
  - [ ] Document automatic device propagation
  - [ ] Add performance considerations comparison table
  - [ ] Include data transfer overhead analysis
  - [ ] Document best practices (conditional device selection, CPU for I/O)
  - [ ] Add mermaid diagram: Device transfer architecture
  - [ ] Add mermaid diagram: Processing pipeline flow
  - [ ] Include complete workflow example with GPU/CPU switching
  - [ ] Ensure examples are executable on both CPU and CUDA
  - [ ] Add citations for GPU computing best practices if applicable

- [ ] Create `docs/source/guides/advanced/pytorch-integration.md`
  - [ ] Read wiki/5.3 PyTorch Integration.md as source
  - [ ] Document nn.Module integration architecture
  - [ ] Explain using nn.Sequential with filters
  - [ ] Show custom module creation patterns
  - [ ] Document gradient computation and differentiability
  - [ ] Explain mixing with torchaudio.transforms
  - [ ] Add mermaid diagram: Integration architecture
  - [ ] Add mermaid diagram: Sequential composition
  - [ ] Include examples from benchmark/api_bench.py
  - [ ] Ensure all code is executable
  - [ ] Link to PyTorch documentation

- [ ] Create `docs/source/guides/advanced/multi-channel.md`
  - [ ] Read wiki/5.1 as source
  - [ ] Document multi-channel tensor shapes and conventions
  - [ ] Explain per-channel vs. cross-channel processing
  - [ ] Cover stereo, surround, and multi-mic scenarios
  - [ ] Add examples for each scenario type
  - [ ] Include visualization diagrams
  - [ ] Link to multi-channel tutorial

- [ ] Create `docs/source/guides/advanced/performance.md`
  - [ ] Read wiki/9 Performance.md as source
  - [ ] Document benchmark results and methodology
  - [ ] Include performance comparison with SciPy (tables/charts)
  - [ ] Analyze GPU vs CPU performance differences
  - [ ] Document optimization strategies
  - [ ] Add performance tips and best practices
  - [ ] Include reproducible benchmark examples
  - [ ] Add citations for benchmarking methodology if applicable
  - [ ] Link to benchmarking developer guide

### Phase 4: Developer Documentation Enhancement

Enhance `docs/source/guides/developer/` with content from wiki sections 7.1-7.6.

**Tasks:**

- [ ] Create `docs/source/guides/developer/project-structure.md`
  - [ ] Read wiki/7.1 Project Structure.md as source
  - [ ] Add package organization diagram
  - [ ] Document module descriptions (wave.py, effect.py, filter/, typing.py)
  - [ ] Explain public API exports and __init__.py structure
  - [ ] Add directory layout explanation with tree diagram
  - [ ] Include mermaid diagram for package dependencies
  - [ ] Link to relevant API reference sections

- [ ] Create `docs/source/guides/developer/testing.md`
  - [ ] Read wiki/7.4 Testing.md as source
  - [ ] Document test structure and organization
  - [ ] Explain running tests (pytest commands and options)
  - [ ] Document writing new tests (patterns and fixtures)
  - [ ] Document coverage requirements and how to check coverage
  - [ ] Add examples of common test patterns
  - [ ] Include continuous integration testing info
  - [ ] Add citations for testing best practices if applicable

- [ ] Create `docs/source/guides/developer/benchmarking.md`
  - [ ] Read wiki/7.5 Benchmarking.md as source
  - [ ] Document benchmark suite structure
  - [ ] Explain running benchmarks (commands and options)
  - [ ] Document performance metrics collection
  - [ ] Explain comparing with baseline (SciPy)
  - [ ] Add example benchmark outputs
  - [ ] Link to performance advanced guide
  - [ ] Add citations for benchmarking methodology if applicable

- [ ] Create `docs/source/guides/developer/documentation.md`
  - [ ] Read wiki/7.6 Documentation.md as source
  - [ ] Document documentation build process (Sphinx commands)
  - [ ] Explain docstring style guide (NumPy format with examples)
  - [ ] Document adding mermaid diagrams to docs
  - [ ] Explain documentation testing and validation
  - [ ] Document using sphinxcontrib.bibtex for citations
  - [ ] Add bibliography file setup instructions
  - [ ] Include cross-reference syntax guide
  - [ ] Add best practices for writing tutorials

- [ ] Update `docs/source/guides/developer/index.md`
  - [ ] Add new sections to toctree (project-structure, testing, benchmarking, documentation)
  - [ ] Update introduction if needed

- [ ] Update `CONTRIBUTING.md`
  - [ ] Read wiki/7.2 Build & Dependencies as source
  - [ ] Add build & dependencies information
  - [ ] Update development setup instructions
  - [ ] Ensure consistency with developer guide documentation

### Phase 5: Improve Existing Documentation

**Tasks:**

- [ ] Enhance `docs/source/guides/getting-started/installation.rst`
  - [ ] Read current installation.rst to understand existing content
  - [ ] Read wiki/1.2 Installation.md as source
  - [ ] Read wiki/1 Overview for dependency details
  - [ ] Add platform-specific PyTorch configuration instructions
  - [ ] Document dependency details (PyTorch version, optional deps)
  - [ ] Add troubleshooting section for common installation issues
  - [ ] Ensure installation commands are up-to-date
  - [ ] Add verification steps (import test)

- [ ] Enhance `docs/source/guides/getting-started/getting_started.rst`
  - [ ] Read current getting_started.rst to understand existing content
  - [ ] Read wiki/1.1 Quick Start.md as source
  - [ ] Read wiki/1 Overview for architecture diagram
  - [ ] Add/improve architecture overview diagram
  - [ ] Add pipeline operator introduction with examples
  - [ ] Ensure first examples are clear and executable
  - [ ] Link to relevant tutorials and core concepts
  - [ ] Add "what's next" navigation section

### Phase 6: Docstring Improvements

Based on wiki API documentation, improve docstrings in source code.

**Tasks:**

- [ ] Enhance docstrings in `src/torchfx/wave.py`
  - [ ] Read wiki/2.1 and wiki/8.1 as sources for examples
  - [ ] Read current wave.py docstrings
  - [ ] Add comprehensive examples to Wave class docstring
  - [ ] Document parameter constraints with examples
  - [ ] Include common usage patterns (creation, manipulation, device transfer)
  - [ ] Add references to related classes/methods
  - [ ] Add citations for audio processing concepts if applicable
  - [ ] Ensure examples are executable

- [ ] Enhance docstrings in `src/torchfx/effect.py`
  - [ ] Read wiki/3.x sections as sources
  - [ ] Read current effect.py docstrings
  - [ ] Add strategy pattern documentation to Effect class
  - [ ] Document AbstractEffect inheritance pattern
  - [ ] Add examples for creating custom effects
  - [ ] Include common usage patterns
  - [ ] Add references to custom effects tutorial
  - [ ] Ensure examples are executable

- [ ] Enhance docstrings in `src/torchfx/filter/__base.py`
  - [ ] Read wiki/4.3 as source for parallel combination examples
  - [ ] Read current filter/__base.py docstrings
  - [ ] Add parallel combination examples (+ operator)
  - [ ] Document filter composition patterns
  - [ ] Add examples for AbstractFilter inheritance
  - [ ] Include references to filter tutorials
  - [ ] Ensure examples are executable

- [ ] Enhance docstrings in `src/torchfx/filter/iir.py`
  - [ ] Read wiki/4.1 as source for filter design examples
  - [ ] Read current filter/iir.py docstrings
  - [ ] Add filter design examples for each IIR filter type
  - [ ] Document filter parameter effects with examples
  - [ ] Include frequency response considerations
  - [ ] Add citations for filter design theory if applicable
  - [ ] Ensure examples are executable

- [ ] Enhance docstrings in `src/torchfx/filter/fir.py`
  - [ ] Read wiki/4.2 as source for FIR filter examples
  - [ ] Read current filter/fir.py docstrings
  - [ ] Add FIR filter design examples
  - [ ] Document windowing methods and their effects
  - [ ] Include usage examples for each FIR filter type
  - [ ] Add citations for FIR filter design if applicable
  - [ ] Ensure examples are executable

## File Modification Summary

### New Files (25 total)

**Core Concepts (5):**
- docs/source/guides/core-concepts/index.md
- docs/source/guides/core-concepts/wave.md
- docs/source/guides/core-concepts/fx.md
- docs/source/guides/core-concepts/pipeline-operator.md
- docs/source/guides/core-concepts/type-system.md
**Tutorials (5):**
- docs/source/guides/tutorials/custom-effects.md
- docs/source/guides/tutorials/series-parallel-filters.md
- docs/source/guides/tutorials/custom-filters.md
- docs/source/guides/tutorials/multi-channel-effects.md
- docs/source/guides/tutorials/bpm-delay.md
**Advanced (5):**
- docs/source/guides/advanced/index.md
- docs/source/guides/advanced/gpu-acceleration.md
- docs/source/guides/advanced/pytorch-integration.md
- docs/source/guides/advanced/multi-channel.md
- docs/source/guides/advanced/performance.md
**Developer (4):**
- docs/source/guides/developer/project-structure.md
- docs/source/guides/developer/testing.md
- docs/source/guides/developer/benchmarking.md
- docs/source/guides/developer/documentation.md

**Plan file (1):**
- DOCS_PLAN.md (copy of this file to repository root)

### Files to Update (8)

- docs/source/guides/index.md
- docs/source/guides/tutorials/index.md
- docs/source/guides/developer/index.rst
- docs/source/guides/getting-started/installation.rst
- docs/source/guides/getting-started/getting_started.rst
- CONTRIBUTING.md
- src/torchfx/wave.py (docstrings)
- src/torchfx/effect.py (docstrings)
- src/torchfx/filter/__base.py (docstrings)

## Implementation Notes

### Mermaid Diagram Handling

- sphinxcontrib.mermaid already installed in conf.py
- Preserve diagrams from wiki:
  - Architecture overviews
  - Sequence diagrams for workflows
  - Class hierarchies
  - Data flow diagrams
- Simplify only where overly complex
- Test all diagrams render correctly in HTML build

### Citations and Bibliography (sphinxcontrib.bibtex)

- sphinxcontrib.bibtex is installed and available
- Create `docs/source/references.bib` for bibliographic entries
- Use for citing:
  - Design patterns (Strategy pattern, etc.)
  - DSP/Filter design theory and algorithms
  - GPU computing best practices
  - Testing and benchmarking methodologies
  - Audio processing concepts
- Citation syntax: `:cite:p:`key`` for parenthetical, `:cite:t:`key`` for textual
- Add `.. bibliography::` directive at end of documents with citations
- Keep citations relevant and academically appropriate

### Code Example Standards

All code examples must:
- Be executable without modification
- Include necessary imports
- Show expected output where relevant
- Reference actual files in examples/ directory
- Work on both CPU and CUDA (with torch.cuda.is_available() checks)

### Cross-References

Use Sphinx cross-references extensively:
- `:doc:` for document links
- `:py:class:` for class references
- `:py:meth:` for method references
- Link tutorials ↔ API reference
- Link core concepts ↔ tutorials

### Validation Checklist

Before completion:
- [ ] All mermaid diagrams render correctly
- [ ] All code examples execute without errors
- [ ] Cross-references resolve correctly
- [ ] Bibliography file exists and citations resolve correctly
- [ ] No duplicate content between API autodoc and tutorials
- [ ] Navigation flows logically (core concepts → tutorials → advanced)
- [ ] Build succeeds with 0 errors (warnings acceptable if minor)
- [ ] All new files added to appropriate toctrees

## Execution Priority

1. **Phase 0** (Setup Bibliography) - Required first for citations throughout docs
2. **Phase 2** (Priority Tutorials) ⭐ - Most valuable for users learning TorchFX
3. **Phase 1** (Core Concepts) - Foundation for understanding
4. **Phase 3** (Advanced Usage) - For experienced users
5. **Phase 4** (Developer Docs) - For contributors
6. **Phase 5** (Enhance Existing) - Polish and completeness
7. **Phase 6** (Docstrings) - Long-term maintenance improvement

## Wiki Files Reference

Key wiki files to process:
- wiki/2 Core Concepts.md → Core Concepts section
- wiki/2.3 Pipeline Operator.md → Priority for tutorials
- wiki/3.5 Creating Custom Effects.md → Priority tutorial
- wiki/4.3 Parallel Filter Combination.md → Priority tutorial
- wiki/5.2 GPU Acceleration.md → Advanced section
- wiki/5.3 PyTorch Integration.md → Advanced section
- wiki/6 Examples.md → Extract to tutorials
- wiki/7.x Development Guide.md → Developer section

## Success Metrics

Documentation improvement will be successful when:
1. New users can create custom effects/filters after reading tutorials
2. Pipeline composition (series/parallel) is clear with visual diagrams
3. GPU acceleration setup is straightforward with working examples
4. Developers can contribute using enhanced developer guides
5. All code examples are executable and tested
6. Navigation between concepts → tutorials → API is intuitive
