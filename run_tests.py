#!/usr/bin/env python3
"""
Test runner for the predictive maintenance ML system.

This script provides different test execution modes for various scenarios.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run tests for predictive maintenance ML system")
    parser.add_argument("--mode", choices=["unit", "integration", "all", "fast", "gpu", "slow"], 
                       default="fast", help="Test mode to run")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-n", type=int, help="Number of parallel workers")
    parser.add_argument("--gpu", action="store_true", help="Include GPU tests")
    parser.add_argument("--install-deps", action="store_true", help="Install test dependencies first")
    
    args = parser.parse_args()
    
    # Change to project directory
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Install dependencies if requested
    if args.install_deps:
        deps_cmd = [sys.executable, "-m", "pip", "install", "-e", ".", "pytest", "pytest-cov", "pytest-xdist"]
        if not run_command(deps_cmd, "Installing test dependencies"):
            return 1
    
    # Build base pytest command
    base_cmd = [sys.executable, "-m", "pytest"]
    
    if args.verbose:
        base_cmd.append("-v")
    
    if args.parallel:
        base_cmd.extend(["-n", str(args.parallel)])
    
    if args.coverage:
        base_cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])
    
    # Add mode-specific arguments
    if args.mode == "unit":
        base_cmd.extend(["tests/unit/", "-m", "unit"])
        description = "Unit Tests"
        
    elif args.mode == "integration":
        base_cmd.extend(["tests/integration/", "-m", "integration"])
        description = "Integration Tests"
        
    elif args.mode == "fast":
        base_cmd.extend(["tests/", "-m", "not slow and not gpu"])
        description = "Fast Tests (no slow or GPU tests)"
        
    elif args.mode == "gpu":
        if not args.gpu:
            print("WARNING: GPU mode requested but --gpu flag not set. Adding --gpu flag.")
            args.gpu = True
        base_cmd.extend(["tests/", "-m", "gpu", "--gpu"])
        description = "GPU Tests"
        
    elif args.mode == "slow":
        base_cmd.extend(["tests/", "-m", "slow"])
        description = "Slow Tests"
        
    elif args.mode == "all":
        base_cmd.append("tests/")
        if args.gpu:
            base_cmd.append("--gpu")
        description = "All Tests"
    
    # Run tests
    success = run_command(base_cmd, description)
    
    if success:
        print(f"\n✅ {description} completed successfully!")
        if args.coverage:
            print("\n📊 Coverage report generated in htmlcov/index.html")
        return 0
    else:
        print(f"\n❌ {description} failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())