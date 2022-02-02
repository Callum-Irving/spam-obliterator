# TODO: Make this compatible with Pypi
from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()


setup(
    name="Spam Obliterator",
    version="0.1.0",
    description="Discord bot that deletes spam messages",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Callum Irving",
    author_email="callum.irving04@gmail.com",
    url="https://github.com/Callum-Irving/spam_obliterator",
    license=license,
    packages=find_packages(),
)
