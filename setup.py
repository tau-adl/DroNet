from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['argparse']

setup(
  name='my-package',
  version='0.1',
  author = 'Idan Basre and Alon Kurtzwile',
  author_email = 'idanbasre@gmail.com',
  install_requires=REQUIRED_PACKAGES,
  packages=find_packages(),
  description='Deep Drone Racing training for DroNet network.')