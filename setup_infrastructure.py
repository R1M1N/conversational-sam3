#!/usr/bin/env python3
"""
SAM3 Conversational Agent - Core Infrastructure Setup

This script sets up the core infrastructure for high-performance SAM3 processing
with JAX acceleration and optimized query processing.
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAM3Infrastructure:
    """Core infrastructure setup for SAM3 conversational agent"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.models_dir = self.project_root / "models"
        self.cache_dir = self.project_root / "cache"
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"
        
        # Create directories
        for dir_path in [self.models_dir, self.cache_dir, self.data_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def check_system_requirements(self):
        """Check system compatibility"""
        logger.info("Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8+ required")
        
        # Check CUDA availability
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {cuda_available}")
            if cuda_available:
                logger.info(f"CUDA version: {torch.version.cuda}")
                logger.info(f"GPU count: {torch.cuda.device_count()}")
                logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        except ImportError:
            logger.warning("PyTorch not installed yet")
        
        # Check JAX installation
        try:
            import jax
            import jax.numpy as jnp
            logger.info(f"JAX version: {jax.__version__}")
            logger.info(f"JAX devices: {jax.devices()}")
        except ImportError:
            logger.warning("JAX not installed yet")
    
    def download_sam3_model(self):
        """Download and setup SAM3 model"""
        logger.info("Setting up SAM3 model...")
        
        model_config = {
            "model_name": "facebook/sam3",
            "cache_dir": str(self.models_dir / "sam3"),
            "local_files_only": False
        }
        
        # Save model configuration
        config_path = self.models_dir / "sam3_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        logger.info(f"SAM3 configuration saved to: {config_path}")
        return model_config
    
    def setup_environment(self):
        """Setup environment variables"""
        logger.info("Setting up environment...")
        
        env_config = {
            "CUDA_VISIBLE_DEVICES": "0",
            "JAX_ENABLE_X64": "false",
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
            "JAX_NUM_CPU_DEVICES": "1",
            "JAX_NUM_GPU_DEVICES": "1",
            "CUDA_CACHE_MAXSIZE": "2147483648",  # 2GB
            "MODEL_CACHE_DIR": str(self.models_dir),
            "DATA_CACHE_DIR": str(self.cache_dir),
            "LOG_DIR": str(self.logs_dir)
        }
        
        # Save environment config
        env_path = self.project_root / ".env"
        with open(env_path, 'w') as f:
            for key, value in env_config.items():
                f.write(f"{key}={value}\n")
        
        logger.info(f"Environment config saved to: {env_path}")
        return env_config
    
    def create_startup_script(self):
        """Create startup script for the development environment"""
        logger.info("Creating startup script...")
        
        startup_script = self.project_root / "start_dev.sh"
        script_content = """#!/bin/bash
# SAM3 Conversational Agent Development Startup Script

echo "🚀 Starting SAM3 Conversational Agent Development Environment..."

# Set environment variables
source .env

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi
fi

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "✨ Development environment ready!"
echo "📁 Models directory: $MODEL_CACHE_DIR"
echo "📁 Cache directory: $DATA_CACHE_DIR"
echo "📁 Logs directory: $LOG_DIR"
echo ""
echo "Available commands:"
echo "  python main.py --dev          # Start development server"
echo "  python -m pytest tests/       # Run tests"
echo "  python scripts/benchmark.py   # Run benchmarks"
echo ""
echo "Ready to build! 🎯"
"""
        
        with open(startup_script, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(startup_script, 0o755)
        logger.info(f"Startup script created: {startup_script}")
    
    def run_setup(self):
        """Run complete infrastructure setup"""
        logger.info("=" * 60)
        logger.info("🚀 SAM3 CONVERSATIONAL AGENT - INFRASTRUCTURE SETUP")
        logger.info("=" * 60)
        
        try:
            self.check_system_requirements()
            self.download_sam3_model()
            self.setup_environment()
            self.create_startup_script()
            
            logger.info("=" * 60)
            logger.info("✅ INFRASTRUCTURE SETUP COMPLETE!")
            logger.info("=" * 60)
            logger.info("Next steps:")
            logger.info("1. Run: pip install -r requirements.txt")
            logger.info("2. Run: ./start_dev.sh")
            logger.info("3. Run: python main.py --dev")
            logger.info("")
            logger.info("You're ready to start building! 🚀")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise

if __name__ == "__main__":
    setup = SAM3Infrastructure()
    setup.run_setup()