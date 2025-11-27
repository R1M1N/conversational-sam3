#!/bin/bash
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
