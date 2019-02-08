rm -r docs/*

cd docs_src

sphinx-build -b html source ../docs 

cd ../docs

rm -r .doctrees 

touch .nojekyll