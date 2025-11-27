#!/usr/bin/env python3
"""
SAM3 Conversational Agent - Development Startup Script

This script initializes the complete development environment and starts the agent.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command with error handling"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            # Extract GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce RTX' in line or 'Tesla' in line:
                    print(f"  GPU: {line.strip()}")
                    break
            return True
    except FileNotFoundError:
        pass
    
    print("⚠️ No NVIDIA GPU detected or CUDA not available")
    return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Check if requirements file exists
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    cmd = f"{sys.executable} -m pip install -r requirements.txt"
    return run_command(cmd, "Installing Python dependencies")

def setup_environment():
    """Setup development environment"""
    print("🛠️ Setting up environment...")
    
    # Create directories
    dirs = ["models", "cache", "data", "logs", "tests"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"  📁 Created directory: {dir_name}")
    
    # Run infrastructure setup
    cmd = f"{sys.executable} setup_infrastructure.py"
    return run_command(cmd, "Setting up infrastructure")

def run_tests():
    """Run basic tests"""
    print("🧪 Running tests...")
    
    # Test imports
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
    except ImportError:
        print("  ❌ PyTorch not available")
        return False
    
    try:
        import jax
        import jax.numpy as jnp
        print(f"  ✅ JAX {jax.__version__}")
        
        # Test basic JAX operation
        a = jnp.array([1, 2, 3])
        b = jnp.array([4, 5, 6])
        c = a + b
        print(f"  ✅ JAX operations working: {c}")
        
    except ImportError:
        print("  ❌ JAX not available")
        return False
    
    try:
        from transformers import Sam3Processor
        print("  ✅ Transformers available")
    except ImportError:
        print("  ❌ Transformers not available")
        return False
    
    return True

def start_agent():
    """Start the SAM3 agent"""
    print("🚀 Starting SAM3 Conversational Agent...")
    
    # Start the web server
    cmd = f"{sys.executable} main.py"
    print("🌐 Starting web server on http://localhost:8000")
    print("🔧 API endpoints available at:")
    print("  GET  /           - API information")
    print("  GET  /health     - Health check")
    print("  POST /query      - Process query")
    print("  WS   /ws         - WebSocket endpoint")
    print("  GET  /performance - Performance metrics")
    print("")
    print("💡 Run 'python cli.py --mode interactive' for CLI mode")
    print("Press Ctrl+C to stop")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Agent stopped")

def main():
    """Main setup and startup function"""
    print("=" * 60)
    print("🚀 SAM3 CONVERSATIONAL AGENT - DEVELOPMENT SETUP")
    print("=" * 60)
    
    # Check system requirements
    if not check_python_version():
        return False
    
    gpu_available = check_gpu()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        return False
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed")
        return False
    
    # Run tests
    if not run_tests():
        print("⚠️ Some tests failed, but continuing...")
    
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print("Your SAM3 Conversational Agent is ready!")
    print("")
    print("Available commands:")
    print("  python main.py           # Start web server")
    print("  python cli.py            # Start CLI interface")
    print("  python cli.py --query 'segment cars'  # Single query")
    print("  python cli.py --benchmark              # Run benchmarks")
    print("")
    
    if gpu_available:
        print("🎯 GPU acceleration enabled - optimal performance!")
    else:
        print("⚠️ CPU mode only - consider installing CUDA for better performance")
    
    print("")
    print("Ready to build! 🚀")
    
    # Ask if user wants to start the agent
    try:
        response = input("Start the agent now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            start_agent()
    except KeyboardInterrupt:
        print("\n👋 Setup complete. Run 'python main.py' to start the agent.")

if __name__ == "__main__":
    main()