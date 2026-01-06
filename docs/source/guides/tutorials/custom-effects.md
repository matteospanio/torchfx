# Custom Effects

Build your own custom audio effects by extending TorchFX's FX base class.

## Overview

This tutorial covers:
- Understanding the FX base class
- Implementing custom effect logic
- Making effects differentiable
- Integrating custom effects into pipelines

## Prerequisites

- [Getting Started](../getting-started/getting_started.rst)
- [Effects Design](effects-design.md)
- Strong Python and PyTorch knowledge

## Tutorial Content

*Coming soon! This tutorial is currently under development.*

## What You'll Learn

- How to extend the FX base class
- Implementing forward pass for custom effects
- Working with torch.nn.Module features
- Testing and validating custom effects

## FX Base Class

All effects in TorchFX inherit from the `FX` base class, which is a subclass of `torch.nn.Module`:

```python
from torchfx import FX
import torch

class MyCustomEffect(FX):
    def __init__(self, param1, param2, fs=None):
        super().__init__(fs=fs)
        self.param1 = param1
        self.param2 = param2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your effect logic here
        return x
```

## Example: Simple Gain Effect

```python
# Example code will be added here
```

## Advanced Topics

- Parameter validation
- State management
- Multichannel processing
- GPU optimization

## Next Steps

- [Filters Design](filters-design.md)
- [ML Batch Processing](ml-batch-processing.md)
- Check out the [API Reference](../../api.rst) for detailed documentation
