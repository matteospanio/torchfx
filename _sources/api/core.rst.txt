Core Classes
============

The core module provides the fundamental classes for working with audio in TorchFX.

Wave
----

The :py:class:`~torchfx.Wave` class handles audio data representation and I/O operations.

.. autoclass:: torchfx.Wave
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__, __call__

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~Wave.from_file
      ~Wave.save
      ~Wave.to
      ~Wave.transform
      ~Wave.get_channel
      ~Wave.merge

FX
--

The :py:class:`~torchfx.FX` base class is the foundation for all effects and filters.

.. autoclass:: torchfx.FX
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__, __call__

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~FX.forward
