#!/usr/bin/env python3
"""
Setup script for Query by Humming project.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        command: Command to run.
        description: Description of what the command does.
        
    Returns:
        True if command succeeded, False otherwise.
    """
    print(f"Running: {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e}")
        if e.stdout:
            print(f"  stdout: {e.stdout}")
        if e.stderr:
            print(f"  stderr: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print(f"✗ Python 3.10+ required, found {sys.version}")
        return False
    print(f"✓ Python version {sys.version.split()[0]} is compatible")
    return True


def install_dependencies() -> bool:
    """Install project dependencies."""
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -e .", "Installing project in development mode"),
        ("pip install -e \".[dev]\"", "Installing development dependencies"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True


def setup_pre_commit() -> bool:
    """Setup pre-commit hooks."""
    commands = [
        ("pre-commit install", "Installing pre-commit hooks"),
        ("pre-commit run --all-files", "Running pre-commit on all files"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True


def run_tests() -> bool:
    """Run test suite."""
    commands = [
        ("pytest tests/ -v", "Running test suite"),
        ("pytest tests/ --cov=src --cov-report=html", "Running tests with coverage"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True


def generate_documentation() -> bool:
    """Generate documentation."""
    commands = [
        ("python example.py", "Running example script"),
        ("python scripts/generate_leaderboard.py --variants", "Generating comprehensive leaderboard"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True


def create_directories() -> bool:
    """Create necessary directories."""
    directories = [
        "data/synthetic",
        "assets",
        "checkpoints",
        "logs",
        "notebooks",
        "tests",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True


def main():
    """Main setup function."""
    print("Query by Humming - Project Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("✗ Failed to create directories")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("✗ Failed to install dependencies")
        sys.exit(1)
    
    # Setup pre-commit (optional)
    setup_pre_commit()
    
    # Run tests
    if not run_tests():
        print("✗ Tests failed")
        sys.exit(1)
    
    # Generate documentation and examples
    if not generate_documentation():
        print("✗ Failed to generate documentation")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✓ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the Streamlit demo: streamlit run demo/streamlit_app.py")
    print("2. Start the FastAPI server: python demo/fastapi_server.py")
    print("3. Explore the Jupyter notebook: jupyter notebook notebooks/")
    print("4. Run training: python scripts/train.py")
    print("5. Run evaluation: python scripts/eval.py")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
