"""Base class for database engines

Copyright 2022-2024, Terence Lim

MIT License
"""

_VERBOSE = 1

class Database:
    def __init__(self, verbose: int = _VERBOSE):
        self._verbose = verbose

    def _print(self, *args, verbose: int = _VERBOSE, level: int = 0, **kwargs):
        """helper to print verbose messages"""
        if max(verbose, self._verbose) > 0:
            print(*args, **kwargs)

