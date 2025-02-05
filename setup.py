from setuptools import setup, find_packages

setup(
   name='carpepy',
   version='0.0.2',
   description='Module for visualising polarised genomes',
   author='Nina Daley & Stuart J.E. Baird',
   author_email='ninahaladova@gmail.com',
   packages=find_packages(),
   install_requires=['matplotlib', 'numpy', 'pandas', 'scipy'],
   download_url='https://github.com/Studenecivb/carpepy/archive/refs/tags/v0.0.1.tar.gz',
)
