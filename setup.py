from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    """
    This function will return list of requirements
    """
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            ## Read lines from the file
            lines=file.readlines()
            ## process each line
            for line in lines:
                requirement=line.strip()
                ## ignore empty lines and -e.
                if requirement and requirement!='-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("Requirements.txt file is not found")

    return requirement_lst


setup(
    name="Network-Security",
    version="0.0.1",
    author="Akash Agalave",
    author_email="akashagalaveaaa1@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)