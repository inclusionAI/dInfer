.. _installation:

============
Installation
============

Last updated: 2025-11-13

This guide shows the minimal steps to set up **dInfer**. All dependency
metadata is defined in ``setup.py``, so you can install directly in editable mode.

Prerequisites
=============

- Git
- Python 3.10 or higher
- CUDA 12.0 or higher (optional, for GPU acceleration) 

Clone the Repository
====================

.. code-block:: bash

   git clone https://github.com/inclusionAI/dInfer.git
   cd dInfer

Install (Editable Mode)
=======================

Install dInfer in editable mode so that your local source changes take effect
immediately without re-installation:

.. code-block:: bash

   pip install -e .

.. note::

   If you prefer isolated environments, create and activate one (e.g., ``python -m venv .venv && source .venv/bin/activate``)
   **before** running the install command above.

GPU Support (Optional)
======================

If your workflow uses GPU acceleration, ensure your CUDA drivers/toolkit and the
relevant deep-learning libraries are properly installed for your platform.

Next Steps
==========

You’re ready to use **dInfer**. Continue to the quickstart for basic usage.
