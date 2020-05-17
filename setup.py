from setuptools import setup, find_packages

setup(
    name='snli',
    version='1.0.0',
    author='John Koutsikakis',
    author_email="jkoutsikakis@gmail.com",
    packages=find_packages(),
    install_requires=[
        'click',
        'pytorch-wrapper',
        'torch',
        'tqdm>=4.36.1,<5',
        'spacy',
        'nlp',
        'numpy'
    ],
    entry_points={
        'console_scripts': ['snli=snli.__main__:cli']
    }
)
