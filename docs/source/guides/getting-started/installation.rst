Installation
================

TorchFX can be installed either from the Python Package Index (PyPI) or by cloning the repository from GitHub. Below are the recommended methods for installing the library.

Installing from PyPI
--------------------

The easiest way to install TorchFX is via pip:

.. code-block:: bash

   pip install torchfx

This will install the latest stable release published on PyPI, along with all required dependencies.

Installing from Source
----------------------

To install the latest development version of TorchFX directly from GitHub, follow these steps:

.. code-block:: bash

   git clone https://github.com/matteospanio/torchfx
   cd torchfx
   pip install -e .

This approach is useful if you plan to contribute to the project or want access to the latest features and updates that may not yet be available on PyPI.

GPU Support
-----------

TorchFX is built on top of **PyTorch**, which means GPU support depends on your local PyTorch installation. To enable GPU acceleration:

1. Make sure you have a compatible NVIDIA GPU.
2. Install PyTorch with CUDA support. You can find the correct installation command on the official PyTorch website: https://pytorch.org/get-started/locally/

Example with pip:

.. code-block:: bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Replace `cu121` with the correct CUDA version for your system.

Developers
----------------
If you are a developer and want to contribute to the TorchFX project, you can set up a development environment by following these steps:
1. Clone the repository:

   .. code-block:: bash

      git clone

2. Navigate to the project directory:

   .. code-block:: bash

      cd torchfx

3. Create a virtual environment (optional but recommended), the project is built using `uv`, hence we suggest to use it:

    .. code-block:: bash

      uv sync --all-groups # the flag --all-groups will install also dev dependencies

Dependencies
++++++++++++

TorchFX requires the following Python packages:

- `torch >= 2.6`
- `torchaudio`
- `scipy`
- `numpy`

These dependencies will be installed automatically if you use `pip install torchfx`.

Checking Installation
---------------------

To verify that the package has been correctly installed, run the following command in Python:

.. code-block:: python

   import torchfx
   print(torchfx.__version__)
