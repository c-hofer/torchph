.. chofer_torchex documentation master file, created by
   sphinx-quickstart on Mon Feb  4 13:39:08 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to chofer_torchex's documentation!
==========================================

This package contains the backend methods for multiple works using persistent homology
in machine learning problems. In particular::

    @inproceedings{Hofer17a,
        author    = {C.~Hofer, R.~Kwitt, M.~Niethammer and A.~Uhl},
        title     = {Deep Learning with Topological Signatures},
        booktitle = {NIPS},
        year      = {2017}}

    @inproceedings{Hofer19a,
        author    = {C.~Hofer, R.~Kwitt, M.~Dixit and M.~Niethammer},
        title     = {Connectivity-Optimized Representation Learning via Persistent Homology},
        booktitle = {ICML},
        year      = {2019}}

    @inproceedings{Hofer19b,
        author    = {C.~Hofer, R.~Kwitt and M.~Niethammer},
        title     = {Graph Filtration Learning},
        booktitle = {arXiv},
        year      = {2019}}

Functionality
=============
* Vietoris-Rips persistent homology
* Vertex-based filtrations (e.g., usable for graphs)
* Learnable vectorizations of persistence barcodes

All of this functionality is available for GPU computations and can easily be
used within the PyTorch environment.


``chofer_torchex`` Package
==========================

.. automodule:: chofer_torchex
    :members:

**Sub-Modules:**

.. toctree::
    nn
    pershom
    utils
    

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
