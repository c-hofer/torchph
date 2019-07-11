Installation
============

The following setup was tested with the following system configuration:

* Ubuntu 18.04.2 LTS
* CUDA 10 (driver version 410.48)
* Anaconda (Python 3.7)
* PyTorch 1.1

In the following, we assume that we work in ``/tmp`` (obviously, you have to
change this to reflect your choice and using ``/tmp`` is, of course, not
the best choice :).

First, get the Anaconda installer and install Anaconda (in ``/tmp/anaconda3``)
using

.. code-block:: bash

    cd /tmp/
    wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
    bash Anaconda3-2019.03-Linux-x86_64.sh
    # specify /tmp/anconda3 as your installation path
    source /tmp/anaconda3/bin/activate

Second, we install PyTorch (v1.1) using

.. code-block:: bash

    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch


Third, we clone the ``chofer_torchex`` repository from GitHub and make
it available within Anaconda.

.. code-block:: bash

    cd /tmp/
    git clone https://github.com/c-hofer/chofer_torchex.git
    git fetch --all --tags --prune     
    git checkout tags/icml2019_code_release -b icml2019_code_release
    conda develop /tmp/chofer_torchex

A quick check if everything works can be done with

.. code-block:: python

    >>> import chofer_torchex

.. note::

    At the moment, we only have GPU support available. CPU support
    is not planned yet, as many other packages exist which support 
    PH computation on the CPU.

