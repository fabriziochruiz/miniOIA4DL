from setuptools import setup
from Cython.Build import cythonize
import numpy
from pathlib import Path

here = Path(__file__).parent
pyx_files = [str(p) for p in here.glob("*.pyx")]

if pyx_files:
    ext_modules = cythonize(pyx_files, language_level=3)
else:
    print("No hay archivos .pyx en cython_modules. Se omite la compilacion Cython.")
    ext_modules = []

setup(ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
)
