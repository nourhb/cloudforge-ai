# scripts/install_dataset_dependencies.py - Install Required Packages for Dataset Scripts
import subprocess
import sys
import os

def install_package(package):
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install all required packages for dataset scripts"""
    print("🔧 Installing CloudForge AI Dataset Dependencies")
    print("="*50)
    
    # Required packages for dataset operations
    packages = [
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "requests>=2.28.0",
        "psycopg2-binary>=2.9.0",  # PostgreSQL adapter
        "minio>=7.1.0",  # MinIO client
        "boto3>=1.26.0",  # AWS SDK
        "sqlalchemy>=1.4.0",  # Database ORM
        "python-dotenv>=0.19.0",  # Environment variables
        "pyyaml>=6.0",  # YAML parsing
        "openpyxl>=3.0.0",  # Excel file support
        "matplotlib>=3.5.0",  # Plotting
        "seaborn>=0.11.0",  # Statistical visualization
        "scikit-learn>=1.1.0",  # Machine learning
        "scipy>=1.9.0",  # Scientific computing
        "jupyterlab>=3.4.0",  # Jupyter notebook environment
        "plotly>=5.10.0",  # Interactive plots
        "dash>=2.6.0",  # Web dashboard framework
        "streamlit>=1.12.0",  # Alternative dashboard
        "fastapi>=0.85.0",  # Fast API framework
        "uvicorn>=0.18.0",  # ASGI server
        "python-multipart>=0.0.5",  # File upload support
        "aiofiles>=0.8.0",  # Async file operations
        "httpx>=0.23.0",  # Async HTTP client
        "pytest>=7.1.0",  # Testing framework
        "pytest-asyncio>=0.19.0",  # Async testing
        "black>=22.6.0",  # Code formatter
        "flake8>=5.0.0",  # Linting
        "mypy>=0.971",  # Type checking
    ]
    
    # Optional packages (won't fail if installation fails)
    optional_packages = [
        "tensorflow>=2.10.0",  # Deep learning
        "torch>=1.12.0",  # PyTorch
        "transformers>=4.21.0",  # Hugging Face transformers
        "xgboost>=1.6.0",  # Gradient boosting
        "lightgbm>=3.3.0",  # Light gradient boosting
        "catboost>=1.0.6",  # CatBoost
        "dask>=2022.8.0",  # Parallel computing
        "polars>=0.14.0",  # Fast dataframes
        "pyarrow>=9.0.0",  # Arrow format
        "redis>=4.3.0",  # Redis client
        "celery>=5.2.0",  # Task queue
        "kafka-python>=2.0.0",  # Kafka client
    ]
    
    print(f"📦 Installing {len(packages)} core packages...")
    
    success_count = 0
    failed_packages = []
    
    # Install core packages
    for package in packages:
        if install_package(package):
            success_count += 1
        else:
            failed_packages.append(package)
    
    print(f"\n📦 Installing {len(optional_packages)} optional packages...")
    
    optional_success = 0
    optional_failed = []
    
    # Install optional packages
    for package in optional_packages:
        if install_package(package):
            optional_success += 1
        else:
            optional_failed.append(package)
            print(f"⚠️  Optional package {package} failed - continuing...")
    
    # Summary
    print("\n" + "="*50)
    print("📊 INSTALLATION SUMMARY")
    print("="*50)
    print(f"✅ Core packages installed: {success_count}/{len(packages)}")
    print(f"✅ Optional packages installed: {optional_success}/{len(optional_packages)}")
    
    if failed_packages:
        print(f"\n❌ Failed core packages:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
    
    if optional_failed:
        print(f"\n⚠️  Failed optional packages:")
        for pkg in optional_failed:
            print(f"   - {pkg}")
    
    # Installation verification
    print(f"\n🔍 VERIFYING INSTALLATIONS")
    print("-" * 30)
    
    critical_imports = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('requests', 'requests'),
        ('minio', 'Minio'),
        ('sqlalchemy', 'create_engine'),
        ('yaml', 'yaml'),
        ('sklearn', 'sklearn'),
        ('matplotlib', 'plt'),
        ('seaborn', 'sns')
    ]
    
    for module, alias in critical_imports:
        try:
            if module == 'matplotlib':
                import matplotlib.pyplot as plt
            elif module == 'seaborn':
                import seaborn as sns
            elif module == 'sklearn':
                import sklearn
            elif module == 'yaml':
                import yaml
            elif module == 'minio':
                from minio import Minio
            elif module == 'sqlalchemy':
                from sqlalchemy import create_engine
            else:
                __import__(module)
            print(f"✅ {module}: OK")
        except ImportError as e:
            print(f"❌ {module}: Failed - {e}")
    
    # Create environment info file
    env_info = {
        'python_version': sys.version,
        'platform': sys.platform,
        'packages_installed': success_count + optional_success,
        'core_failures': failed_packages,
        'optional_failures': optional_failed,
        'installation_date': str(__import__('datetime').datetime.now())
    }
    
    try:
        import json
        with open('dataset_env_info.json', 'w') as f:
            json.dump(env_info, f, indent=2)
        print(f"\n💾 Environment info saved to: dataset_env_info.json")
    except Exception as e:
        print(f"⚠️  Could not save environment info: {e}")
    
    # Final recommendations
    print(f"\n🎯 NEXT STEPS")
    print("-" * 20)
    print("1. Run: python scripts/validate_datasets.py")
    print("2. Run: python scripts/sprint_validator.py") 
    print("3. Execute: bash scripts/setup_datasets.sh")
    print("4. Test AI services: python scripts/load_uci_dataset.py")
    print("5. Validate marketplace: python scripts/load_kaggle_ecommerce.py")
    
    if len(failed_packages) == 0:
        print(f"\n🎉 All core dependencies installed successfully!")
        print(f"📈 CloudForge AI is ready for dataset operations!")
        return True
    else:
        print(f"\n⚠️  Some core packages failed. Please install manually:")
        for pkg in failed_packages:
            print(f"   pip install {pkg}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)