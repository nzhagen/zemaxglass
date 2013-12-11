# chardet's setup.py
from distutils.core import setup

setup(
    name = "ZemaxGlass",
    py_modules = ["ZemaxGlass"],
    version = "1.0",
    description = "Zemax glass (.agf) file reader",
    author = "Nathan Hagen",
    author_email = "nhagen@optics.arizona.edu",
    url = "https://github.com/nzhagen/zemaxglass",
    download_url = "https://github.com/nzhagen/zemaxglass/blob/master/ZemaxGlass.py",
    keywords = ["zemax", "glass", "spectra", "dispersion"],
    classifiers = [
        "Programming Language :: Python :: 2.7",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        ],
    long_description = """\
A reader for Zemax format optical glass catalog files.
-----------------------------------------------------------

"ZemaxGlass" reads in a directory of Zemax ".agf" format files and builds a library
of glasses from them. A complete library is included with the repository. This allows
users to plot the dispersion curves of glasses, and to visually compare the properties of all
glasses in the library against one another. (See the user manual for details.)

This version requires Python 2.7 or later.
"""
)
