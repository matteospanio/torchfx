API
---

TorchFX provides two main classes: :py:class:`torchfx.Wave` and :py:class:`torchfx.FX`.
The :py:class:`torchfx.Wave` class is used to handle audio data, while the :py:class:`torchfx.FX` class is used to apply various audio effects and transformations. The library also provides a set of built-in effects and filters that can be easily applied to audio data.

.. autoclass:: torchfx.Wave
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__, __call__

.. autoclass:: torchfx.FX
    :members:
    :show-inheritance:
    :exclude-members: __init__, __str__, __repr__, __call__

The already implemented filters are disponible under the `filter` module:

.. automodule:: torchfx.filter
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__, __call__

The already implemented effects are disponible under the `effects` module:

.. automodule:: torchfx.effects
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__
