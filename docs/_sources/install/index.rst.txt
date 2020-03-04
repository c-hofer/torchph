Installation
============

The following setup was tested with the following system configuration:

* Ubuntu 18.04.2 LTS
* CUDA 10.1 (driver version 418.87.00)
* Anaconda (Python 3.7.6)
* PyTorch 1.4

In the following, we assume that we work in ``/tmp`` (obviously, you have to
change this to reflect your choice and using ``/tmp`` is, of course, not
the best choice :).

First, get the Anaconda installer and install Anaconda (in ``/tmp/anaconda3``)
using

.. code-block:: bash

    cd /tmp/
    wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
    bash Anaconda3-2019.10-Linux-x86_64.sh
    # specify /tmp/anaconda3 as your installation path
    source /tmp/anaconda3/bin/activate

Second, we install PyTorch (v1.4) using

.. code-block:: bash

    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch


Third, we clone the ``chofer_torchex`` repository from GitHub and make
it available within Anaconda.

.. code-block:: bash

    cd /tmp/
    git clone https://github.com/c-hofer/chofer_torchex.git
    conda develop /tmp/chofer_torchex

A quick check if everything works can be done with

.. code-block:: python

    >>> import chofer_torchex

.. note::

    At the moment, we only have GPU support available. CPU support
    is not planned yet, as many other packages exist which support 
    PH computation on the CPU.

