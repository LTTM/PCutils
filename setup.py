import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='PCutils',
    version='0.3.0',
    author='Daniele Mari',
    author_email='daniele.mari@phd.unipd.it',
    description='Package with useful processing functions for PCs with focus on PC coding',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/LTTM/PCutils',
    project_urls = {
    },
    license='MIT',
    packages=['PCutils'],
    install_requires=['numpy', 'matplotlib', "plyfile", "pandas"],
)
