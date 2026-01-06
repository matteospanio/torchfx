Getting Started
===============

This guide will help you get started with **TorchFX**, a GPU-accelerated audio DSP library built on top of PyTorch.

We will cover:

- Loading audio using the `Wave` class
- Applying built-in filters
- Creating your own custom audio effect using the `FX` base class

Basic Concepts
--------------

TorchFX uses an object-oriented interface where audio signals are wrapped in a `Wave` object that holds both the audio samples (`y`) and the sampling rate (`fs`).

You can build audio processing pipelines by chaining operations using the **pipe operator (`|`)**, thanks to Python operator overloading.

Wave Class
----------

To begin, import the library and load a waveform from file:

.. code-block:: python

   import torchfx as fx

   # Load an audio file
   wave = fx.Wave.from_file("path_to_audio.wav")

   # Access the raw audio data and sampling rate
   print(wave.y.shape)   # e.g., torch.Size([2, 44100])
   print(wave.fs)        # e.g., 44100

The `Wave` object automatically handles stereo or multichannel data and ensures that filters retain sample rate context.

Applying Built-in Filters
-------------------------

TorchFX provides a collection of IIR and FIR filters under the `torchfx.filter` module. All filters are implemented as subclasses of `torch.nn.Module`.

Here's an example of chaining filters with the pipe operator:

.. code-block:: python

   from torchfx import filter as fx_filter

   # Apply a low-pass Butterworth filter at 8 kHz and a high-shelving filter at 2 kHz
   filtered = (
       fx.Wave.from_file("example.wav")
       | fx_filter.LoButterworth(8000)
       | fx_filter.HiShelving(2000)
   )

   # Save the processed signal
   filtered.to_file("filtered_output.wav")

You can also build pipelines using `torch.nn.Sequential` or define custom modules as in PyTorch.

Creating Your Own Effect
------------------------

To create your own audio effect, subclass the `FX` class (a utility base class derived from `torch.nn.Module`):

.. code-block:: python

   from torchfx.core import FX

   class Invert(FX):
       def forward(self, wave):
           return wave.new(-wave.y)

This custom `Invert` effect simply negates the audio signal. You can now use it like any other TorchFX module:

.. code-block:: python

   inverted = wave | Invert()

   # Listen or save the output
   inverted.to_file("inverted.wav")

The `FX` base class ensures that your custom effect works seamlessly with the `Wave` class and supports the pipe operator.
