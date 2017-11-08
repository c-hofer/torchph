This repository is still **under construction** ... you're more than invited to try stuff, but strange things may happen ;) 

# chofer_torchex

In this repository I gather my extensions to [PyTorch](http://pytorch.org). 
The packaging structure tries to reproduce PyTorch's structure in order 
to facilitate usage for people familiar with PyTorch. In the following, you
can find list of its main features (admittedly short ... yet ;) ).

1. `nn.SLayer`:
This is an input layer which can operate on multisets of points in some 
cartesian product of the real numbers. Its primary intention is to train 
networks on the output of a topological data analysis pipeline, but can 
be used on arbitrary (real vector) multiset input. [Tutorial](https://github.com/c-hofer/chofer_torchex/blob/master/tutorials/SLayer.ipynb)

See [Deep Learning with Topological Signatures](https://arxiv.org/abs/1707.04041) for 
further reading. 

```bash
@inproceedings{Hofer17c,
  author    = {C.~Hofer and R.~Kwitt and M.~Niethammer and A.~Uhl},
  title     = {Deep Learning with Topological Signatures},
  booktitle = {NIPS},
  year      = 2017,
  note      = {accepted}
}
```

The folder *tutorials* contains minimalistic examples in form of Jupyter notebooks
to demonstrate how to use the `PyTorch` extensions. 
