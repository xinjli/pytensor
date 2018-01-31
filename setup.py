from setuptools import setup

setup(
   name='pytensor',
   version='0.1',
   description='A deep learning framework using pure numpy',
   author='Xinjian Li',
   author_email='lixinjian1217@gmail.com',
   packages=['pytensor'],  #same as name
   install_requires=['scikit-learn', 'numpy'], #external packages as dependencies
)
