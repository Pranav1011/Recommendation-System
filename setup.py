from setuptools import setup, find_packages

setup(
    name="recommendation-engine",
    version="0.1.0",
    description="Scalable Recommendation Engine with Vector Search",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "pyspark>=3.5.0",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "torch>=2.1.0",
        "scikit-learn>=1.3.0",
        "mlflow>=2.9.0",
        "qdrant-client>=1.7.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "redis>=5.0.0",
        "prometheus-client>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "isort>=5.13.0",
        ]
    },
)
