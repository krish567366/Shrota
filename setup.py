from setuptools import setup, find_packages

setup(
    name="predictive-maintenance-ml",
    version="0.1.0",
    description="Advanced Predictive Maintenance ML System with A100 GPU Optimization",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.25.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "wandb>=0.15.0",
        "pytorch-forecasting>=1.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
        ],
        "deployment": [
            "tensorrt>=8.6.0",
            "onnx>=1.14.0",
            "onnxruntime-gpu>=1.15.0",
            "triton-client[all]>=2.35.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pm-train=training.train:main",
            "pm-predict=inference.predict:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)