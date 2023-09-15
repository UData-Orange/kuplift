from setuptools import setup

try:
    import pypandoc

    long_description = pypandoc.convert_file("README.md", "rst")
except (IOError, ImportError):
    long_description = open("README.md").read()

setup(
    name="kuplift",
    version="0.0.5",
    packages=["kuplift"],
    description="A User Parameter-free Bayesian Framework for Uplift Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "pandas",
        "sortedcontainers",
        "scikit-learn",
        "pytest",
    ],
)
