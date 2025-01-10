from setuptools import setup, find_packages

setup(
    name="Extractive-Text-Analyzer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'openai',
        'nltk',
        'pytest',
        'pytest-asyncio',
        'pytest-cov',
        'hypothesis',
    ],
) 