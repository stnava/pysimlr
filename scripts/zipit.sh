
zip -r ~/Downloads/pysimlrclean.zip . \
  -x "pysimlr.egg-info/*" \
     "**/pysimlr.egg-info/*" \
     "**/node_modules/*" "**/.git/*" "**/__pycache__/*" \
     "__pycache__/*" \
     "**/__pycache__/*" \
     "*.pyc" \
     "*.pyo" \
     ".pytest_cache/*" \
     "**/.pytest_cache/*" \
     ".coverage" \
     ".coverage.*" \
     "htmlcov/*" \
     ".mypy_cache/*" \
     "**/.mypy_cache/*" \
     ".ruff_cache/*" \
     "**/.ruff_cache/*" \
     ".tox/*" \
     ".venv/*" \
     "venv/*" \
     "env/*" \
     "smalldata/*" \
     "bigdata/*" \
     ".env" \
     ".DS_Store" \
     "**/.DS_Store" \
     ".git/*"
