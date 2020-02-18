This branch contains most up-to-date implementations, improvements and documentation. We are in the process of merging this branch into master as soon as it is stable. 

We are also moving all documentation away from the `README.md` file into a sphinx-generated documentation. To build the documentation and obtain all installation instructions, tutorials, etc., follow the instructions below:

```bash
git clone https://github.com/c-hofer/chofer_torchex.git
git checkout persom_devel
git pull
pip install sphinx
pip install nbsphinx
cd chofer_torchex
bash update_docs.sh
```

Then open `docs/index.html`.
