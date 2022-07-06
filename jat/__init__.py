"""Pure JAX implementation of a graph attention network."""

# Add imports here
from .jat import *
from . import graph

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
