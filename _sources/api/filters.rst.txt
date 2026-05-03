Filters
=======

TorchFX provides a comprehensive collection of IIR and FIR filters for audio processing.

.. currentmodule:: torchfx.filter

Base Classes
------------

IIR
~~~

.. autoclass:: IIR
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

FIR Filters
-----------

FIR
~~~

.. autoclass:: FIR
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

DesignableFIR
~~~~~~~~~~~~~

.. autoclass:: DesignableFIR
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

IIR Filters
-----------

Standard Filters
~~~~~~~~~~~~~~~~

Butterworth
^^^^^^^^^^^

.. autoclass:: Butterworth
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: HiButterworth
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: LoButterworth
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

Chebyshev Type I
^^^^^^^^^^^^^^^^

.. autoclass:: Chebyshev1
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: HiChebyshev1
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: LoChebyshev1
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

Chebyshev Type II
^^^^^^^^^^^^^^^^^

.. autoclass:: Chebyshev2
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: HiChebyshev2
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: LoChebyshev2
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

Elliptic
^^^^^^^^

.. autoclass:: Elliptic
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: HiElliptic
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: LoElliptic
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

Linkwitz-Riley
^^^^^^^^^^^^^^

.. autoclass:: LinkwitzRiley
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: HiLinkwitzRiley
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: LoLinkwitzRiley
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

Equalizers and Specialty Filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shelving Filters
^^^^^^^^^^^^^^^^

.. autoclass:: HiShelving
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: LoShelving
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

Parametric EQ
^^^^^^^^^^^^^

.. autoclass:: ParametricEQ
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: Peaking
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

Other Filters
^^^^^^^^^^^^^

.. autoclass:: Notch
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: AllPass
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

Biquad Filters
~~~~~~~~~~~~~~

Direct biquad implementations from the Audio EQ Cookbook. The base ``Biquad``
class plus convenience subclasses for common topologies.

.. autoclass:: Biquad
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: BiquadLPF
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: BiquadHPF
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: BiquadBPF
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: BiquadNotch
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: BiquadAllPass
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

Filterbanks and Fused Cascades
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LogFilterBank
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__

.. autoclass:: FusedSOSCascade
   :members:
   :show-inheritance:
   :exclude-members: __init__, __str__, __repr__
