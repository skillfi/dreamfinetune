import sys

from setuptools import setup, find_packages

version_range_max = max(sys.version_info[1], 10) + 1

setup(
    name='dreamfinetune',
    version="1.5",
    packages=find_packages(where='src'),  # Вказуємо директорію, де шукати пакети
    package_dir={'': 'src'},  # Вказуємо, що пакети знаходяться в директорії `src`
    license="Apache 2.0 License",
    install_requires=[
        "pillow>=9.4.0",
        "diffusers~=0.30.0",
        "transformers>=4.42.4,<4.45.0",
        "tqdm~=4.66.4",
        "datasets~=2.20.0",
        "bitsandbytes",
        "ftfy",
        "gradio",
        "tensorboard"
    ],
    extras_require={
        'torch': [
            "torch>=2.3.1",
            "torchvision>=0.18.0",
            "torchaudio>=2.3.1"
        ]
    },
    dependency_links=[
        "https://download.pytorch.org/whl/cu121"
    ],
    include_package_data=True,
    author='Alex',
    long_description=open('README.rst', "r", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/skillfi/fine-tuning',
    classifiers=[
                    "Development Status :: 5 - Production/Stable",
                    "Intended Audience :: Developers",
                    "Intended Audience :: Education",
                    "Intended Audience :: Science/Research",
                    "License :: OSI Approved :: Apache Software License",
                    "Operating System :: OS Independent",
                    "Topic :: Scientific/Engineering :: Artificial Intelligence",
                    "Programming Language :: Python :: 3",
                ] + [f"Programming Language :: Python :: 3.{i}" for i in range(8, version_range_max)],
    python_requires='>=3.8'
)
