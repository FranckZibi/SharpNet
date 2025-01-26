import os, sys
# Define the path to the directory
directory = os.path.abspath('../../Prod/Python/')
# Add the directory to sys.path if not already there
if directory not in sys.path:
    sys.path.append(directory)
