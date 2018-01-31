from setuptools import setup,find_packages

setup(
   name='pytensor',
   version='0.101',
   description='A deep learning framework using pure numpy',
   author='Xinjian Li',
   author_email='lixinjian1217@gmail.com',
   packages=find_packages(),
   install_requires=['scikit-learn', 'numpy'], #external packages as dependencies
)
