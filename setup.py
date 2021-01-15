import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="riznets", # Replace with your own username
    version="0.0.3",
    author="Richard Hu",
    author_email="r.hu@berkeley.edu",
    description="Make and train neural networks on MNIST and CIFAR-10",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rizhu/riznets",
    packages=setuptools.find_packages(exclude=['*.nn', 'saved-neural-nets/']),
    entry_points = {
        'console_scripts': ['riznets=riznets.cli:main'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)