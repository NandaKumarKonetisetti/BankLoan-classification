from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(filepath:str) ->List[str]:
    requirements = []
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements  = [req.replace("\n","") for req in requirements]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name="Loan Classification Project",
    version=1.0,
    author="nanda",
    author_email="nandakumaroye@gmail.com",
    install_requires = get_requirements('requirements.txt'),
    packages= find_packages()
    )
