from setuptools import setup

setup(
    name="kuplift",
    version="0.0.2",
    packages=["kuplift"],
    description="A User Parameter-free Bayesian Framework for Uplift Modeling",
    long_description="Refer to the documentation at https://udata-orange.github.io/kuplift/",
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "pandas",
        "sortedcontainers",
    ],
)
