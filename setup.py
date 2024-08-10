from setuptools import setup, find_packages

setup(
    name='Fine Tuning AI',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "pillow~=10.4.0",
        "torchvision~=0.19.0",
        "diffusers~=0.30.0",
        "transformers~=4.44.0",
        "tqdm~=4.66.4",
        "datasets~=2.20.0",
        "bitsandbytes"
    ],
    author='Alex',
    long_description=open('README.md').read(),
    long_description_content='text/markdown',
    url='https://github.com/skillfi/fine-tuning',
    classifiers=[
        "Programing Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12'
)
