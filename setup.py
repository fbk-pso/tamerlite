#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(name='tamerlite',
      version='0.1.0',
      description='Lite version of Tamer',
      author='FBK PSO Unit',
      author_email='tamer@fbk.eu',
      url='tamer.fbk.eu',
      packages=find_packages(),
      install_requires=['rustamer~=0.1.0'],
      python_requires='>=3.10',
      license='Free For Educational Use',
      classifiers=[
          'License :: Free For Educational Use'
      ]
     )
