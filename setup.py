from setuptools import find_packages,setup

HYPHEN_E_DOT = '-e .'

def get_requirments(file_path):

    requirements=[]

    with open(file_path) as file_obj:

        requirements=file_obj.readlines()

        requirements=[req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:

            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name="ai_vs_human_face_detection",
    version="0.0.1",
    author="devansh",
    packages=find_packages(),
    install_requires=get_requirments('requirements.txt')

)