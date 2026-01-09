from setuptools import find_packages, setup

setup(
    name="HD-utils",
    version="0.0.1",
    description="Custom functions used by Siyuan Mei in HD cell project",
    packages=find_packages(exclude=["code_from_Ruben", "data_from_Ruben","__Turner-Evans et al eLife 2017 code"]),
    include_package_data=True,
    zip_safe=False,
)