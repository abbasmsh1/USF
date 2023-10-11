from setuptools import setup, find_packages

# Read the contents of requirements.txt as dependencies
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='fastapi-app',
    version='1.0.0',
    description='Your FastAPI App Description',
    author='Abbas',
    author_email='abbasmsh1@gmail.com',
    packages=find_packages(),
    install_requires=requirements,
)
