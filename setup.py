from setuptools import setup, find_packages

setup(
    name="mlops-classification",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "joblib>=1.0.1",
        "pyyaml>=5.4.1",
        "dvc>=2.9.3",
        "boto3>=1.18.0",
        "jupyter>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "black>=21.7b0",
            "flake8>=3.9.2",
            "mypy>=0.910",
        ]
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="MLOps Classification Project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mlops-classification",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
) 