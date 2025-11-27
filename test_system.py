#!/usr/bin/env python3
"""
SAM3 Agent System Test

Quick test to verify all components are working correctly.
"""

import sys
import time
from pathlib import Path

def test_imports():
    """Test all imports"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import jax
        import jax.numpy as jnp
        print(f"✅ JAX {jax.__version__}")
        print(f"   Devices: {jax.devices()}")
        
        # Test JAX operation
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0, 6.0])
        c = a + b
        print(f"   Test operation: {a.tolist()} + {b.tolist()} = {c.tolist()}")
        
    except ImportError as e:
        print(f"❌ JAX import failed: {e}")
        return False
    
    try:
        from transformers import Sam3Processor
        print("✅ Transformers (SAM3 support)")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import flax
        print("✅ Flax")
    except ImportError as e:
        print(f"❌ Flax import failed: {e}")
        return False
    
    return True

def test_agent_imports():
    """Test our agent components"""
    print("\n🤖 Testing agent components...")
    
    try:
        sys.path.append(str(Path(__file__).parent / "src"))
        
        # Test query parser
        from core.query_parser import SAM3QueryParser
        parser = SAM3QueryParser()
        print("✅ Query Parser initialized")
        
        # Test query parsing
        test_query = "segment all red cars"
        parsed = parser.parse_query(test_query)
        print(f"   Test query: '{test_query}'")
        print(f"   Parsed entities: {[e.text for e in parsed.entities]}")
        print(f"   Query type: {parsed.query_type.value}")
        
    except Exception as e:
        print(f"❌ Query parser test failed: {e}")
        return False
    
    try:
        # Test SAM3 engine (without actual model)
        from core.sam3_engine import SAM3Config
        config = SAM3Config()
        print("✅ SAM3 Engine config initialized")
        
    except Exception as e:
        print(f"❌ SAM3 engine test failed: {e}")
        return False
    
    try:
        # Test conversational agent
        from agent.conversational_agent import AgentConfig
        config = AgentConfig()
        print("✅ Conversational Agent config initialized")
        
    except Exception as e:
        print(f"❌ Conversational agent test failed: {e}")
        return False
    
    return True

def test_jax_performance():
    """Test JAX performance with simple operations"""
    print("\n⚡ Testing JAX performance...")
    
    try:
        import jax
        import jax.numpy as jnp
        from jax import random
        
        # Simple matrix operations
        key = random.PRNGKey(42)
        size = 1000
        
        # Test 1: Matrix multiplication
        start_time = time.time()
        A = random.normal(key, (size, size))
        B = random.normal(key, (size, size))
        C = A @ B
        matmul_time = time.time() - start_time
        print(f"✅ Matrix multiplication ({size}x{size}): {matmul_time:.3f}s")
        
        # Test 2: Batch operations
        start_time = time.time()
        batch_size = 100
        matrices = random.normal(key, (batch_size, 100, 100))
        results = jax.vmap(lambda m: m @ m.T)(matrices)
        batch_time = time.time() - start_time
        print(f"✅ Batch operations ({batch_size}x{100}x{100}): {batch_time:.3f}s")
        
        # Test 3: GPU memory info
        if jax.devices('gpu'):
            print(f"✅ GPU devices available: {len(jax.devices('gpu'))}")
        else:
            print("⚠️ No GPU devices found")
        
        return True
        
    except Exception as e:
        print(f"❌ JAX performance test failed: {e}")
        return False

def test_system_info():
    """Display system information"""
    print("\n💻 System Information:")
    
    import platform
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {platform.python_version()}")
    
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA: {'✅' if torch.cuda.is_available() else '❌'}")
        if torch.cuda.is_available():
            print(f"   GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    except:
        pass
    
    try:
        import jax
        print(f"   JAX: {jax.__version__}")
        print(f"   JAX Devices: {jax.devices()}")
    except:
        pass

def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 SAM3 CONVERSATIONAL AGENT - SYSTEM TEST")
    print("=" * 60)
    
    # Test system info
    test_system_info()
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        return False
    
    # Test agent components
    if not test_agent_imports():
        print("\n❌ Agent component tests failed!")
        return False
    
    # Test JAX performance
    if not test_jax_performance():
        print("\n❌ JAX performance tests failed!")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("✅ Your SAM3 Conversational Agent is ready!")
    print("\nNext steps:")
    print("  python cli.py --mode interactive")
    print("  python main.py")
    print("  python cli.py --benchmark")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)