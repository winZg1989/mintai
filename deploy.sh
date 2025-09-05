#!/bin/bash

# Create deployment directory
mkdir -p .python_packages/lib/python3.13/site-packages

# Install dependencies to target directory
pip install -t .python_packages/lib/python3.13/site-packages -r requirements.txt

# Create startup command file
echo "python -m gunicorn app:app -b 0.0.0.0:\$PORT --workers 2 --threads 4 --timeout 60" > startup.txt