import os
import sys
from pathlib import Path
import importlib.util

directory = Path(__file__).parent.absolute()
name = os.path.split(directory)[-1] 

module_ext = ".so"
if sys.platform == "win32":
    module_ext = ".pyd"
    os.add_dll_directory(directory)

for file in os.listdir(directory):
    filename, ext = os.path.splitext(file)
    if ext == module_ext:
        target = os.path.join(directory, file)
        spec = importlib.util.spec_from_file_location(name, target)
        sys.modules[name] = importlib.util.module_from_spec(spec)
