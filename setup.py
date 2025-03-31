from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='robotdatapy-minus-ros',
    version='1.0.0',    
    description='Pure Python package for interfacing with robot data (robotdatapy without ROS dependencies)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/AparajitoSaha/robotdatapy-minus-ros',
    author='Aparajito Saha',
    author_email='aparajito.saha@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'pykitti',
                        'evo',
                        'opencv-python',
                      ],
)