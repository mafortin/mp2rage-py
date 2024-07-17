from setuptools import setup, find_packages

# Read the content of README.md for the long description
with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='mp2rage-py',
    version='0.1.0',
    author='Marc-Antoine Fortin',
    author_email='marc.a.fortin@ntnu.no',
    description='A small python package for processing and *trying) to understand MP2RAGE.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mafortin/mp2rage-py',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        line.strip() for line in open('requirements.txt')
    ],
    scripts=[
        'get_T1map_from_UNI.py',
    ],
    include_package_data=True,
)
