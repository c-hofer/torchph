# torchph

This repository contains [PyTorch](http://pytorch.org) extensions to **compute**
persistent homology and to **differentiate** through the persistent homology computation.
The packaging structure is similar to PyTorch's structure to facilitate usage for people 
familiar with PyTorch. 

## Documentation

[Read the docs!](https://c-hofer.github.io/torchph/)

The folder *tutorials* (within `docs`) contains some (more or less) minimalistic examples in form of Jupyter notebooks
to demonstrate how to use the `PyTorch` extensions. 

## Associated publications

If you use any of these extensions, please cite the following works (depending on which functionality you use, obviously :)

```bash
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

@article{Hofer19b,
  author    = {C.~Hofer, R.~Kwitt, and M.~Niethammer},
  title     = {Learning Representations of Persistence Barcodes},
  booktitle = {JMLR},
  year      = {2019}}
  
@inproceedings{Hofer20a},
  author    = {C.~Hofer, F.~Graf, R.~Kwitt, B.~Rieck and M.~Niethammer},
  title     = {Graph Filtration Learning},
  booktitle = {arXiv},
  year      = {2020}}
  
@inproceedings{Hofer20a,     
  author    = {C.~Hofer, F.~Graf, M.~Niethammer and R.~Kwitt},     
  title     = {Topologically Densified Distributions},     
  booktitle = {arXiv},    
  year      = {2020}} 
```
