.. _glossary:

========
Glossary
========

This glossary defines common terms used throughout the TorchFX documentation.

.. glossary::
   :sorted:

   Audio Effect
      A signal processing algorithm that modifies an audio signal to produce a desired sonic characteristic. Examples include reverb, delay, distortion, and modulation effects. See also :term:`DSP`.

   BPM
      Beats Per Minute. A unit of measurement for musical tempo, indicating the number of beats that occur in one minute. Used in :term:`Musical Time` calculations for tempo-synchronized effects.

   CUDA
      Compute Unified Device Architecture. NVIDIA's parallel computing platform and programming model that enables :term:`GPU` acceleration for general-purpose computing. See `CUDA documentation <https://docs.nvidia.com/cuda/>`_.

   Cutoff Frequency
      The frequency at which a :term:`Filter` begins to attenuate the signal, typically defined as the point where the magnitude response is reduced by 3 dB. Fundamental parameter in filter design.

   Device
      In PyTorch, refers to the hardware where tensor computations are performed, either CPU or GPU (CUDA). TorchFX supports automatic device management for audio processing pipelines.

   Digital Filter
      A computational algorithm that processes digital signals to modify their frequency content. Can be :term:`FIR` or :term:`IIR`. Fundamental building block in :term:`DSP`.

   DSP
      Digital Signal Processing. The use of digital computation to perform signal processing operations. Core to audio manipulation in TorchFX. See `DSP on Wikipedia <https://en.wikipedia.org/wiki/Digital_signal_processing>`_.

   Filter
      Short for :term:`Digital Filter`. In TorchFX, filters are PyTorch modules that implement frequency-selective signal processing operations.

   FIR
      Finite Impulse Response. A type of :term:`Digital Filter` whose output depends only on current and past input samples. FIR filters are always stable and have linear phase response. See `FIR filters on Wikipedia <https://en.wikipedia.org/wiki/Finite_impulse_response>`_.

   Frequency Response
      The measure of a system's output spectrum in response to an input signal, typically showing magnitude and phase as functions of frequency. Essential for understanding filter behavior.

   GPU
      Graphics Processing Unit. A specialized processor designed for parallel computation, increasingly used for general-purpose computing tasks including audio DSP. See :term:`CUDA`.

   Hz
      Hertz. The SI unit of frequency, defined as one cycle per second. Used to specify frequencies in audio processing (e.g., cutoff frequency, sample rate).

   IIR
      Infinite Impulse Response. A type of :term:`Digital Filter` whose output depends on both input samples and previous output samples (feedback). IIR filters can be more efficient than FIR but require stability considerations. See `IIR filters on Wikipedia <https://en.wikipedia.org/wiki/Infinite_impulse_response>`_.

   Immutability
      A design pattern where objects cannot be modified after creation. TorchFX's :class:`~torchfx.Wave` class follows immutability principles, returning new instances rather than modifying existing ones.

   Musical Time
      Time specified in musical units (e.g., quarter notes, eighth notes) relative to a :term:`BPM` tempo, rather than absolute time in seconds. Used for tempo-synchronized audio effects.

   Pipeline
      A sequence of connected audio processing operations where the output of one stage becomes the input to the next. In TorchFX, pipelines are created using the pipe operator (``|``).

   Pipeline Operator
      The pipe operator (``|``) in TorchFX, used to chain effects and filters into processing pipelines. Automatically handles sample rate configuration and signal routing.

   Sample Rate
      The number of audio samples captured or played per second, measured in :term:`Hz`. Common sample rates include 44.1 kHz (CD quality) and 48 kHz (professional audio). Abbreviated as ``fs`` in TorchFX.

   Strategy Pattern
      A software design pattern that enables selecting an algorithm's behavior at runtime. Used in TorchFX effects (e.g., :class:`~torchfx.Delay`) to support different processing strategies.

   Tensor
      PyTorch's fundamental data structure, an n-dimensional array similar to NumPy arrays but with GPU acceleration support. Audio signals in TorchFX are represented as tensors. See `PyTorch tensor documentation <https://pytorch.org/docs/stable/tensors.html>`_.

   Transfer Function
      A mathematical representation of the relationship between a system's input and output in the frequency domain. Used to characterize :term:`Digital Filter` behavior.

   Wave
      TorchFX's primary data structure for representing audio signals, implemented as an immutable wrapper around PyTorch tensors. Provides audio-specific operations and device management. See :class:`~torchfx.Wave`.

   Z-Transform
      A mathematical transform used to analyze and design :term:`Digital Filter` in the discrete-time domain. The discrete-time equivalent of the Laplace transform. See `Z-transform on Wikipedia <https://en.wikipedia.org/wiki/Z-transform>`_.

.. seealso::

   External Resources
      - `Digital Signal Processing (Wikipedia) <https://en.wikipedia.org/wiki/Digital_signal_processing>`_
      - `PyTorch Documentation <https://pytorch.org/docs/stable/index.html>`_
      - `Audio Signal Processing (Wikipedia) <https://en.wikipedia.org/wiki/Audio_signal_processing>`_
      - `CCRMA Stanford - DSP Resources <https://ccrma.stanford.edu/>`_
