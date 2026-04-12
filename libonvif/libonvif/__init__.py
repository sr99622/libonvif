from importlib.metadata import PackageNotFoundError, version

from .libonvif import *
from .libonvif import __doc__

try:
    __version__ = version("libonvif")
except PackageNotFoundError:
    __version__ = "unknown"