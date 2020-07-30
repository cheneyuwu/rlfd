from distutils.core import setup
from setuptools import find_packages

setup(
    name='gym-rlfd',
    version='0.0.0',
    install_requires=['gym', 
                      'numpy', 
                      'mujoco_py'],
    packages=find_packages(),
)