#!/usr/bin/env python
# coding=utf-8
"""
Python utility package for common computer vision tasks.
"""

__all__ = ['cam_projections', 'colormaps', 'detection2d',
           'flowutils', 'imutils', 'imvis', 'pyutils']
__author__ = 'snototter'

# Load version
import os
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'version.py')) as vf:
    exec(vf.read())
