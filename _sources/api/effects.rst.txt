Effects
=======

TorchFX provides built-in audio effects for processing audio signals.

.. currentmodule:: torchfx.effect

Gain and Normalization
-----------------------

Gain
~~~~

.. autoclass:: Gain
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

Normalize
~~~~~~~~~

.. autoclass:: Normalize
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

Normalization Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: NormalizationStrategy
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: PeakNormalizationStrategy
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: RMSNormalizationStrategy
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: PercentileNormalizationStrategy
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: PerChannelNormalizationStrategy
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: CustomNormalizationStrategy
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

Time-based Effects
------------------

Reverb
~~~~~~

.. autoclass:: Reverb
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

Delay
~~~~~

.. autoclass:: Delay
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

Delay Strategies
^^^^^^^^^^^^^^^^

.. autoclass:: DelayStrategy
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: MonoDelayStrategy
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: PingPongDelayStrategy
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__
