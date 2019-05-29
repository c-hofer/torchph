find tests/ | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
pytest tests/  
