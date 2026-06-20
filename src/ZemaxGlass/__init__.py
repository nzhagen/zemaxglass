#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from importlib.metadata import version

try:
    __version__ = version(__name__)
except:
    __version__ = 'unknown'
finally:
    del version
