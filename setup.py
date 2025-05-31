from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()
    
    
setup(
    author = "channu",
    version = "0.1",
    name = "Anime-Recommender",
    packages= find_packages(),
    install_requires = requirements
)