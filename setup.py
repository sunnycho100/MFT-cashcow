from setuptools import setup, find_packages

setup(
    name="crypto-trader",
    version="0.1.0",
    description="Crypto Quantitative Trading System",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "polars>=0.20.0",
        "duckdb>=0.9.0",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "xgboost>=2.0.0",
        "scikit-learn>=1.3.0",
        "arch>=6.2.0",
        "statsmodels>=0.14.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "crypto-backtest=scripts.run_backtest:main",
            "crypto-paper=scripts.run_paper:main",
            "crypto-retrain=scripts.retrain:main",
        ],
    },
)
