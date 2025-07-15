from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str)-> List[str]:
    """
    This function accepts path of the requirements file and returns them as a list of strings
    """
    with open(file_path) as file_obj:
        requirements= file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith(('#', '-'))]

    return requirements

setup(
    name= 'machine_learning_project',
    version='0.0.1',
    packages= find_packages(),
    install_requires= get_requirements('requirements.txt'),
    author= 'Yogesh Barthwal',
    author_email= 'barthwal.yogesh@gmail.com',
    description= 'An end to end Machine Learning Project',
    long_description=open('README.md').read(),
    python_requires= '>=3.7',
) 