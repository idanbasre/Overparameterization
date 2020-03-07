from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['argparse']

setup(
  name='my-package',
  version='0.1',
  author = 'Idan Basre and Adam Shafir',
  author_email = 'idanbasre@gmail.com and adamshafir@gmail.com',
  install_requires=REQUIRED_PACKAGES,
  packages=find_packages(),
  description='Pytorch implementartion and further experiments of the article: On the Optimization of Deep Networks: Implicit Acceleration by Overparameterization\\Sanjeev Arora, Nadav Cohen, Elad Hazan'