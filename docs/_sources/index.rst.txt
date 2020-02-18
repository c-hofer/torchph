.. chofer_torchex documentation master file, created by
   sphinx-quickstart on Mon Feb  4 13:39:08 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyTorch extensions for persistent homology
==========================================

This package contains the backend methods (to be used within the PyTorch environment)
for multiple works using persistent homology in machine learning problems. In particular,
the following publications are most relevant::

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

.. note::
  Note that not all of the available functionality is exposed in the
  documentation yet.

Get started
===========

.. toctree::
   :maxdepth: 1
   :caption: Get Started
   :hidden:
   :glob:

   install/index

Follow the :doc:`instructions<install/index>` to install ``chofer_torchex``.

Functionality
=============
* Vietoris-Rips (VR) persistent homology (from point clouds and distance matrices)
* Vertex-based filtrations (e.g., usable for graphs)
* Learnable vectorizations of persistence barcodes

All of this functionality is available for **GPU** computations and can easily be
used within the PyTorch environment.

The following **simple example** is a teaser showing how to compute 0-dim. persistent
homology of a (1) Vietoris-Rips filtration which uses the Manhatten distance between
samples and (2) doing the same using a pre-computed distance matrix.

.. code-block:: python

    device = "cuda:0"

    # import numpy
    import numpy as np

    # import VR persistence computation functionality
    from chofer_torchex.pershom import vr_persistence_l1, vr_persistence

    # import scipy methods to compute pairwise distance matrices
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform

    # create 10-dim. point cloud with 100 samples
    x = np.random.randn(100, 10)

    # compute VR persistent homology (using l1 metric)
    X = torch.Tensor(x).to(device)
    l_a, _ = vr_persistence_l1(X.contiguous(),0, 0);

    # compute the same using a pre-computed distance matrix
    D = torch.tensor(
            squareform(
                pdist(x, metric='cityblock')
            )
        ).to("cuda:0")
    l_b, _ = vr_persistence(D, 0, 0)
    print("Diff: ",
        (l_a[0].float()-l_b[0].float()).abs().sum().item())

.. toctree::
    :caption: Modules 
    :maxdepth: 1
    :hidden:
    :glob:

    nn
    pershom
    utils

.. toctree::
    :caption: Notebooks
    :maxdepth: 0
    :hidden:
    :glob:

    tutorials/SLayer.ipynb
    tutorials/ToyDiffVR.ipynb
    tutorials/ComparisonSOTA.ipynb
    tutorials/InputOptim.ipynb

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
