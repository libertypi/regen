from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="regen",
    version="0.1",
    description="A regular expressions generator",
    long_description=long_description,
    url="https://github.com/libertypi/regen",
    author="David Pi",
    author_email="libertypi@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="regular-expression, optimization",
    packages=find_packages(exclude=["test"]),
    install_requires=["ortools"],
    python_requires=">=3.6",
)
