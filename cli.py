#!/usr/bin/env python3
"""
CLI Interface for SAM3 Conversational Agent

Command-line interface for interacting with the SAM3 conversational agent
with interactive mode, batch processing, and benchmarking capabilities.
"""

import asyncio
import time
import argparse
import json
import os
import sys
from pathlib import Path
import glob
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.agent.conversational_agent import SAM3ConversationalAgent, AgentConfig, SAM3Config
from src.core.sam3_engine import SAM3Config as EngineConfig

class SAM3CLI:
    """Command-line interface for SAM3 conversational agent"""
    
    def __init__(self):
        self.agent = None
        self.current_image = None  # Added state to track loaded image
        self.colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'reset': '\033[0m',
            'bold': '\033[1m'
        }
    
    def colorize(self, text: str, color: str = 'white', bold: bool = False) -> str:
        """Add color to text"""
        base = f"{self.colors.get(color, '')}"
        if bold:
            base += self.colors['bold']
        return f"{base}{text}{self.colors['reset']}"
    
    def print_header(self):
        """Print CLI header"""
        header = f"""
{self.colorize('=' * 60, 'cyan')}
{self.colorize('🚀 SAM3 CONVERSATIONAL AGENT CLI', 'bold', True)}
{self.colorize('=' * 60, 'cyan')}
{self.colorize('High-Performance SAM3 with Natural Language Understanding', 'white')}
{self.colorize('Type "help" for commands or "quit" to exit', 'yellow')}
{self.colorize('=' * 60, 'cyan')}
        """
        print(header)
    
    def print_help(self):
        """Print help information"""
        help_text = f"""
{self.colorize('Available Commands:', 'green', True)}

{self.colorize('Basic Commands:', 'blue', True)}
  help                    - Show this help message
  quit/exit               - Exit the CLI
  clear                   - Clear conversation history
  cache                   - Clear cache
  
{self.colorize('Information Commands:', 'blue', True)}
  status                  - Show agent status and performance
  history                 - Show conversation history
  performance             - Show detailed performance metrics
  
{self.colorize('Processing Commands:', 'blue', True)}
  load <path>             - Load an image for processing
  query <text>            - Process a natural language query
  batch <file>            - Process queries from a file
  benchmark               - Run performance benchmarks
  
{self.colorize('Examples:', 'yellow', True)}
  {self.colorize('load Grapes.jpg', 'white')}
  {self.colorize('query "segment all red cars"', 'white')}
  {self.colorize('query "segment grapes in my_fruits_folder"', 'white')}
        """
        print(help_text)
    
    async def initialize_agent(self):
        """Initialize the SAM3 agent"""
        print(self.colorize("🔄 Initializing SAM3 agent...", 'yellow'))
        
        try:
            # Configure agent
            agent_config = AgentConfig(
                max_history_turns=50,
                enable_learning=True,
                confidence_threshold=0.7,
                cache_responses=True,
                enable_monitoring=True
            )
            
            sam3_config = SAM3Config(
                device="cuda" if self.check_cuda() else "cpu",
                batch_size=4,
                enable_jit=True,
                precompile_models=False,  # Disabled for faster startup
                memory_efficient=True
            )
            
            # Initialize agent
            self.agent = SAM3ConversationalAgent(agent_config, sam3_config)
            
            # Health check
            health = await self.agent.health_check()
            if health["status"] == "healthy":
                print(self.colorize("✅ Agent initialized successfully", 'green'))
                return True
            else:
                print(self.colorize(f"⚠️ Agent initialized with warnings: {health}", 'yellow'))
                return True
                
        except Exception as e:
            print(self.colorize(f"❌ Failed to initialize agent: {e}", 'red'))
            return False
    
    def check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    async def process_query(self, query: str):
        """Process a single query (with folder support)"""
        if not self.agent:
            print(self.colorize("❌ Agent not initialized", 'red'))
            return
        
        # Check if query refers to a folder (Batch Processing)
        parts = query.split()
        target_folder = None
        for word in parts:
            if os.path.isdir(word):
                target_folder = word
                break
        
        if target_folder:
            print(self.colorize(f"📂 Detected folder: {target_folder}", 'cyan', True))
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.avif', '*.webp']
            image_files = []
            for ext in extensions:
                image_files.extend(glob.glob(os.path.join(target_folder, ext)))
            
            if not image_files:
                print(self.colorize("❌ No images found in folder.", 'red'))
                return

            print(self.colorize(f"🚀 Batch processing {len(image_files)} images...", 'yellow'))
            
            for i, img_file in enumerate(image_files, 1):
                print(self.colorize(f"\n[{i}/{len(image_files)}] Processing {img_file}...", 'blue'))
                # Pass the file explicitly via image_path
                await self.agent.process_query(query, image_path=img_file)
            
            print(self.colorize("\n✅ Batch processing complete!", 'green', True))
            return

        # Single Image Logic
        if not self.current_image and "image:" not in query and not any(x in query for x in ['.jpg', '.png', '.avif']):
             print(self.colorize("⚠️ No image loaded. Use 'load <file>' or mention a filename.", 'yellow'))

        print(self.colorize(f"🔍 Processing: {query}", 'blue'))
        
        try:
            start_time = time.time()
            # Pass current_image if available
            result = await self.agent.process_query(
                query, 
                image=self.current_image if self.current_image else None
            )

            processing_time = time.time() - start_time
            
            # Print response
            print(f"\n{self.colorize('🤖 Agent Response:', 'green', True)}")
            print(result['response'])
            
            # Print additional info if available
            if result.get('sam3_results'):
                sam3_data = result['sam3_results']
                num_objects = sam3_data.get('num_objects_found', 0)
                if num_objects > 0:
                    print(f"\n{self.colorize(f'🎯 Objects found: {num_objects}', 'cyan')}")
                    labels = sam3_data.get("labels", [])
                    print(self.colorize(f"Labels: {labels}", "white"))
            
            print(f"\n{self.colorize(f'⏱️ Processing time: {processing_time:.3f}s', 'yellow')}")
            
            if result.get('from_cache'):
                print(self.colorize('📦 From cache', 'magenta'))
            
        except Exception as e:
            print(self.colorize(f"❌ Error: {e}", 'red'))
    
    async def show_status(self):
        """Show agent status"""
        if not self.agent:
            print(self.colorize("❌ Agent not initialized", 'red'))
            return
        
        try:
            health = await self.agent.health_check()
            print(self.colorize("📊 Agent Status:", 'green', True))
            print(f"Status: {health['status']}")
            print(f"Model loaded: {health['model_loaded']}")
            print(f"Loaded Image: {self.current_image if self.current_image else 'None'}")
            print(f"GPU memory available: {health.get('gpu_memory_available', 'Unknown')}")
            
        except Exception as e:
            print(self.colorize(f"❌ Error getting status: {e}", 'red'))
    
    async def show_performance(self):
        """Show detailed performance metrics"""
        if not self.agent:
            print(self.colorize("❌ Agent not initialized", 'red'))
            return
        
        try:
            report = self.agent.get_performance_report()
            
            print(self.colorize("📈 Performance Report:", 'green', True))
            
            # Performance summary
            perf = report['performance_summary']
            print(f"\n{self.colorize('Performance Summary:', 'blue', True)}")
            print(f"Total queries: {perf['total_queries']}")
            print(f"Success rate: {perf['success_rate']}")
            print(f"Average processing time: {perf['average_processing_time']}")
            print(f"Cache hit rate: {perf['cache_hit_rate']}")
            
            # System resources
            resources = report['system_resources']
            print(f"\n{self.colorize('System Resources:', 'blue', True)}")
            print(f"GPU memory used: {resources.get('gpu_memory_used', 'Unknown')}")
            print(f"Cache size: {resources['cache_size']}")
            print(f"Conversation turns: {resources['conversation_history']}")
            
            # Optimization status
            opt = report['optimization_status']
            print(f"\n{self.colorize('Optimization Status:', 'blue', True)}")
            print(f"JAX enabled: {opt['jax_enabled']}")
            print(f"Batch processing: {opt['batch_processing']}")
            print(f"Caching enabled: {opt['caching_enabled']}")
            
        except Exception as e:
            print(self.colorize(f"❌ Error getting performance: {e}", 'red'))
    
    async def show_history(self):
        """Show conversation history"""
        if not self.agent:
            print(self.colorize("❌ Agent not initialized", 'red'))
            return
        
        try:
            history = self.agent.get_conversation_history()
            
            print(self.colorize(f"📚 Conversation History ({len(history)} turns):", 'green', True))
            
            for i, turn in enumerate(history[-10:], 1):  # Show last 10
                print(f"\n{self.colorize(f'Turn {i}:', 'blue', True)}")
                print(f"Query: {turn['user_query']}")
                print(f"Response: {turn['agent_response']}")
                print(f"Time: {turn['processing_time']:.3f}s")
                
        except Exception as e:
            print(self.colorize(f"❌ Error getting history: {e}", 'red'))
    
    async def clear_history(self):
        """Clear conversation history"""
        if not self.agent:
            print(self.colorize("❌ Agent not initialized", 'red'))
            return
        
        try:
            self.agent.clear_history()
            print(self.colorize("✅ History cleared", 'green'))
        except Exception as e:
            print(self.colorize(f"❌ Error clearing history: {e}", 'red'))
    
    async def clear_cache(self):
        """Clear cache"""
        if not self.agent:
            print(self.colorize("❌ Agent not initialized", 'red'))
            return
        
        try:
            self.agent.clear_cache()
            print(self.colorize("✅ Cache cleared", 'green'))
        except Exception as e:
            print(self.colorize(f"❌ Error clearing cache: {e}", 'red'))
    
    async def run_interactive(self):
        """Run interactive mode"""
        if not await self.initialize_agent():
            return
        
        self.print_header()
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n{self.colorize('SAM3> ', 'cyan', True)}").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(' ', 1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                # Handle commands
                if command in ['quit', 'exit']:
                    print(self.colorize("👋 Goodbye!", 'yellow'))
                    break
                    
                elif command == 'help':
                    self.print_help()
                    
                elif command == 'clear':
                    await self.clear_history()
                    
                elif command == 'cache':
                    await self.clear_cache()
                
                elif command == 'load':
                    if os.path.exists(args):
                        self.current_image = args
                        print(self.colorize(f"✅ Loaded image: {args}", 'green'))
                    else:
                        print(self.colorize(f"❌ File not found: {args}", 'red'))
                    
                elif command == 'status':
                    await self.show_status()
                    
                elif command == 'performance':
                    await self.show_performance()
                    
                elif command == 'history':
                    await self.show_history()
                    
                elif command == 'query':
                    if args:
                        await self.process_query(args)
                    else:
                        print(self.colorize("❌ Please provide a query", 'red'))
                        
                elif command == 'benchmark':
                    await self.run_benchmark()
                    
                else:
                    # Treat as direct query
                    await self.process_query(user_input)
                    
            except KeyboardInterrupt:
                print(self.colorize("\n👋 Goodbye!", 'yellow'))
                break
            except Exception as e:
                print(self.colorize(f"❌ Error: {e}", 'red'))
    
    async def run_benchmark(self):
        """Run performance benchmarks"""
        if not self.agent:
            print(self.colorize("❌ Agent not initialized", 'red'))
            return
        
        print(self.colorize("🏃 Running benchmarks...", 'yellow'))
        
        # Benchmark queries
        test_queries = [
            "segment all cars",
            "find people wearing glasses",
            "detect red objects",
            "segment buildings and trees",
            "find dogs that are sitting"
        ]
        
        results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"{self.colorize(f'Benchmark {i}/{len(test_queries)}:', 'blue')} {query}")
            
            try:
                start_time = time.time()
                result = await self.agent.process_query(query)
                end_time = time.time()
                
                results.append({
                    'query': query,
                    'processing_time': end_time - start_time,
                    'success': result.get('error') is None
                })
                
                print(f"  ✅ {end_time - start_time:.3f}s")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    'query': query,
                    'processing_time': 0,
                    'success': False,
                    'error': str(e)
                })
        
        # Summary
        successful = [r for r in results if r['success']]
        avg_time = sum(r['processing_time'] for r in successful) / max(1, len(successful))
        
        print(f"\n{self.colorize('📊 Benchmark Summary:', 'green', True)}")
        print(f"Successful: {len(successful)}/{len(results)}")
        print(f"Average time: {avg_time:.3f}s")
        if successful:
            min_time = min(r['processing_time'] for r in successful)
            max_time = max(r['processing_time'] for r in successful)
        else:
            min_time = max_time = 0
        print(f"Min time: {min_time:.3f}s")
        print(f"Max time: {max_time:.3f}s")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="SAM3 Conversational Agent CLI")
    parser.add_argument('--mode', choices=['interactive', 'query', 'benchmark'], 
                        default='interactive', help='Run mode')
    parser.add_argument('--query', help='Process a single query')
    parser.add_argument('--batch', help='Process queries from file')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    
    args = parser.parse_args()
    
    cli = SAM3CLI()
    
    # Run based on mode
    if args.mode == 'interactive':
        asyncio.run(cli.run_interactive())
        
    elif args.mode == 'query':
        if not args.query:
            print("❌ Please provide --query")
            return
        
        async def single_query():
            if not await cli.initialize_agent():
                return
            await cli.process_query(args.query)
        
        asyncio.run(single_query())
        
    elif args.mode == 'benchmark':
        async def run_benchmark():
            if not await cli.initialize_agent():
                return
            await cli.run_benchmark()
        
        asyncio.run(run_benchmark())

if __name__ == "__main__":
    main()
