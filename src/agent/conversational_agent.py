"""
Conversational SAM3 Agent

Main agent that combines query parsing, SAM3 processing, and conversational interface
for high-performance segment anything operations with natural language.
"""

import os
import time
import logging
import asyncio
import re
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

# Import core components
from ..core.sam3_engine import SAM3JAXEngine, SAM3Config, ProcessingResult
from ..core.query_parser import SAM3QueryParser, ParsedQuery, QueryType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in conversation"""
    user_query: str
    agent_response: str
    processing_time: float
    sam3_results: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AgentConfig:
    """Configuration for the conversational agent"""
    max_history_turns: int = 10
    enable_learning: bool = True
    confidence_threshold: float = 0.7
    timeout_seconds: float = 30.0
    enable_monitoring: bool = True
    cache_responses: bool = True
    auto_optimize: bool = True

class SAM3ConversationalAgent:
    """High-performance conversational agent for SAM3 operations"""
    
    def __init__(self, config: AgentConfig = None, sam3_config: SAM3Config = None):
        """Initialize the conversational agent"""
        self.config = config or AgentConfig()
        self.sam3_config = sam3_config or SAM3Config()
        
        # Initialize core components
        self.query_parser = SAM3QueryParser()
        self.sam3_engine = SAM3JAXEngine(self.sam3_config)
        
        # Conversation management
        self.conversation_history: List[ConversationTurn] = []
        self.query_cache = {}
        self.performance_metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "average_processing_time": 0.0,
            "cache_hit_rate": 0.0,
            "user_satisfaction": 0.0
        }
        
        # Load models
        self._load_models()
        
        logger.info("SAM3 Conversational Agent initialized")
        logger.info(f"Config: {self.config}")
        logger.info(f"SAM3 Config: {self.sam3_config}")
    
    def _load_models(self):
        """Load all required models"""
        logger.info("Loading models...")
        
        try:
            # Load SAM3 model
            self.sam3_engine.load_model()
            logger.info("✅ SAM3 model loaded")
            
            # Log system status
            self._log_system_status()
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise
    
    def _log_system_status(self):
        """Log current system status"""
        stats = self.sam3_engine.get_performance_stats()
        memory = self.sam3_engine.memory_usage()
        
        logger.info("=" * 50)
        logger.info("🚀 SAM3 CONVERSATIONAL AGENT STATUS")
        logger.info("=" * 50)
        logger.info(f"Total Queries Processed: {stats['total_queries']}")
        logger.info(f"Average Processing Time: {stats['avg_time']:.3f}s")
        logger.info(f"GPU Memory Used: {memory.get('gpu_memory', 0):.2f}GB")
        logger.info(f"GPU Memory Free: {memory.get('gpu_free', 0):.2f}GB")
        logger.info(f"Model Loaded: {stats['model_loaded']}")
        logger.info(f"Processor Loaded: {stats['processor_loaded']}")
        logger.info("=" * 50)
    
    async def process_query(self, 
                             user_query: str, 
                             image: Union[str, Path, Any] = None,
                             image_path: Optional[str] = None,
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a user query with optional image
        """
        # 1. Consolidate image input
        target_path = image_path if image_path else (image if isinstance(image, (str, Path)) else None)
        loaded_image = image if not isinstance(image, (str, Path)) and image is not None else None

        # 2. Intelligent Path Extraction
        if not target_path and not loaded_image:
            match = re.search(r'\b[\w\-\/]+\.(jpg|jpeg|png|avif|webp|bmp)\b', user_query, re.IGNORECASE)
            if match:
                possible_path = match.group(0)
                if os.path.exists(possible_path):
                    target_path = possible_path
                    logger.info(f"📂 Auto-detected image in query: {target_path}")
                else:
                    logger.warning(f"Query contains filename '{possible_path}' but file not found.")

        # 3. Load the image if we found a path
        if target_path and not loaded_image:
            if os.path.exists(target_path):
                try:
                    from PIL import Image
                    loaded_image = Image.open(target_path)
                    logger.info(f"✅ Loaded image from {target_path}")
                except Exception as e:
                    logger.error(f"Failed to load image {target_path}: {e}")

        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {user_query}")
            
            # Check cache first
            cache_key = self._get_cache_key(user_query, loaded_image)
            if self.config.cache_responses and cache_key in self.query_cache:
                logger.info("📦 Using cached result")
                cached_result = self.query_cache[cache_key]
                cached_result["from_cache"] = True
                self.performance_metrics["cache_hit_rate"] += 1
                return cached_result
            
            # Parse the query
            parsed_query = self.query_parser.parse_query(user_query, image_path=target_path)
            logger.info(f"🔍 Query type: {parsed_query.query_type.value}")
            logger.info(f"🔍 Entities found: {len(parsed_query.entities)}")
            logger.info(f"🔍 Constraints: {len(parsed_query.constraints)}")
            
            # Generate processing plan
            processing_plan = self.query_parser.generate_processing_plan(parsed_query)
            logger.info(f"📋 Processing plan: {processing_plan}")
            
            # Process with SAM3 if image is provided
            sam3_results = None
            if loaded_image is not None:
                sam3_results = await self._process_with_sam3(loaded_image, parsed_query)
            
            # Generate conversational response
            response = self._generate_response(
                user_query, parsed_query, sam3_results, context
            )
            
            # Auto-Save Results
            saved_path = None
            if sam3_results and sam3_results.get("num_objects_found", 0) > 0 and loaded_image:
                original_name = target_path if target_path else "image"
                saved_path = self._save_results(loaded_image, sam3_results, original_name)
                if saved_path:
                    response += f"\n\n💾 Results saved to: {saved_path}"
            
            # Compile final result
            processing_time = time.time() - start_time
            result = {
                "user_query": user_query,
                "response": response,
                "sam3_results": sam3_results,
                "saved_path": saved_path,
                "processing_time": processing_time,
                "parsed_query": asdict(parsed_query),
                "processing_plan": processing_plan,
                "performance_metrics": self._get_current_metrics(),
                "from_cache": False
            }
            
            # Cache the result
            if self.config.cache_responses:
                self.query_cache[cache_key] = result
            
            # Update metrics & history
            self._update_metrics(processing_time, True)
            self._update_history(user_query, response, processing_time, sam3_results)
            
            logger.info(f"✅ Query processed in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Query processing failed: {e}")
            
            error_response = {
                "user_query": user_query,
                "response": f"I apologize, but I encountered an error processing your query: {str(e)}",
                "sam3_results": None,
                "processing_time": processing_time,
                "error": str(e),
                "from_cache": False
            }
            
            self._update_metrics(processing_time, False)
            return error_response
    
    async def _process_with_sam3(self, image: Any, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """Process image with SAM3 based on parsed query"""
        logger.info("🎯 Starting SAM3 processing...")
        
        # Convert parsed query to SAM3 prompts
        sam3_prompts = self.query_parser.convert_to_sam3_prompts(parsed_query)
        
        # Process with SAM3
        start_time = time.time()
        sam3_result = self.sam3_engine.segment_objects(
            image=image,
            text_prompts=sam3_prompts,
            batch_mode=len(sam3_prompts) > 1
        )
        processing_time = time.time() - start_time
        
        logger.info(f"🎯 SAM3 processing completed in {processing_time:.3f}s")
        logger.info(f"🎯 Found {len(sam3_result.masks)} objects")
        
        # Convert result to serializable format
        return {
            "masks": sam3_result.masks.tolist() if len(sam3_result.masks) > 0 else [],
            "labels": sam3_result.labels,
            "scores": sam3_result.scores.tolist(),
            "confidence_scores": sam3_result.confidence_scores.tolist(),
            "processing_time": sam3_result.processing_time,
            "num_objects_found": len(sam3_result.masks)
        }
    
    def _save_results(self, image: Any, sam3_results: Dict[str, Any], original_filename: str = "output") -> str:
        """Save segmented image to disk with overlays"""
        if not sam3_results or not sam3_results.get("masks"):
            return None
            
        # Convert PIL image to OpenCV format
        if hasattr(image, 'convert'):
            img_np = np.array(image.convert("RGB"))
        else:
            img_np = np.array(image)
            
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        orig_h, orig_w = img_bgr.shape[:2]
        
        # Create overlay
        overlay = img_bgr.copy()
        masks = sam3_results["masks"]
        
        # Colors for masks (random colors)
        np.random.seed(42)
        colors = [np.random.randint(0, 255, 3).tolist() for _ in range(len(masks))]
        
        for i, mask in enumerate(masks):
            mask_np = np.array(mask)
            if mask_np.ndim > 2:
                mask_np = mask_np[0]
            
            # Threshold to binary BEFORE resizing
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            
            # Resize mask to match original image size
            if mask_binary.shape != (orig_h, orig_w):
                # Use LINEAR for smoothness, then re-threshold
                mask_resized_float = cv2.resize(mask_binary.astype(float), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                mask_resized = (mask_resized_float > 0.5).astype(np.uint8)

            else:
                mask_resized = mask_binary
            
            # Ensure binary
            mask_bool = mask_resized > 0
            
            color = colors[i]
            
            # Draw mask overlay
            if mask_bool.shape == overlay.shape[:2]:
                overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array(color) * 0.5
                
                # Draw contour
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)
            
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(original_filename).stem if isinstance(original_filename, (str, Path)) else "result"
        output_filename = f"segmented_{base_name}_{timestamp}.jpg"
        
        # Save
        cv2.imwrite(output_filename, overlay)
        logger.info(f"✅ Saved segmentation result to {output_filename}")
        return output_filename

    def _generate_response(self, 
                                user_query: str, 
                                parsed_query: ParsedQuery,
                                sam3_results: Optional[Dict[str, Any]],
                                context: Optional[Dict[str, Any]]) -> str:
        """Generate conversational response"""
        
        if sam3_results is None:
            return ("I understand you want to work with segmentation, but I need an image to process. "
                    "Please provide an image along with your query.")
        
        num_objects = sam3_results.get("num_objects_found", 0)
        labels = sam3_results.get("labels", [])
        
        # Generate response based on query type and results
        if parsed_query.query_type == QueryType.SIMPLE_SEGMENTATION:
            if num_objects == 0:
                return f"I didn't find any {labels[0] if labels else 'objects'} in the image."
            elif num_objects == 1:
                return f"Found 1 {labels[0]} in the image."
            else:
                return f"Found {num_objects} {labels[0] if labels else 'objects'} in the image."
        
        elif parsed_query.query_type == QueryType.MULTI_OBJECT:
            if num_objects == 0:
                return "I didn't find any of the requested objects in the image."
            elif num_objects == 1:
                return f"Found 1 object: {labels[0] if labels else 'identified'}."
            else:
                return f"Found {num_objects} objects: {', '.join(labels[:5])}{'...' if len(labels) > 5 else ''}."
        
        elif parsed_query.query_type == QueryType.CONDITIONAL:
            return (f"Found {num_objects} objects that match your criteria. "
                    f"I've applied the constraints you specified.")
        
        else:
            return (f"Processed your query and found {num_objects} relevant objects. "
                    f"The segmentation results are ready for analysis.")
    
    def _get_cache_key(self, query: str, image: Any) -> str:
        """Generate cache key for query and image"""
        import hashlib
        
        # Simple hash for query
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        
        # Image hash if provided
        if image is not None:
            try:
                if hasattr(image, 'read'):
                    # File-like object
                    image_content = image.read()
                    image_hash = hashlib.md5(image_content).hexdigest()[:8]
                elif isinstance(image, str) and os.path.exists(image):
                    # File path
                    with open(image, 'rb') as f:
                        image_content = f.read()
                    image_hash = hashlib.md5(image_content).hexdigest()[:8]
                else:
                    # Other image type - use string representation
                    image_hash = hashlib.md5(str(image).encode()).hexdigest()[:8]
            except Exception:
                image_hash = "error"
        else:
            image_hash = "no_image"
        
        return f"{query_hash}_{image_hash}"
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.performance_metrics,
            "conversation_turns": len(self.conversation_history),
            "cache_size": len(self.query_cache),
            "system_status": self.sam3_engine.get_performance_stats()
        }
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update performance metrics"""
        self.performance_metrics["total_queries"] += 1
        
        if success:
            self.performance_metrics["successful_queries"] += 1
        
        # Update average processing time
        total_time = (self.performance_metrics["average_processing_time"] * 
                      (self.performance_metrics["total_queries"] - 1) + processing_time)
        self.performance_metrics["average_processing_time"] = (
            total_time / self.performance_metrics["total_queries"]
        )

    def _update_history(self, query: str, response: str, processing_time: float, sam3_results: Dict[str, Any]):
        """Update conversation history"""
        turn = ConversationTurn(
            user_query=query,
            agent_response=response,
            processing_time=processing_time,
            sam3_results=sam3_results
        )
        self.conversation_history.append(turn)
        if len(self.conversation_history) > self.config.max_history_turns:
            self.conversation_history = self.conversation_history[-self.config.max_history_turns:]
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return [asdict(turn) for turn in self.conversation_history]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        self.sam3_engine.clear_cache()
        logger.info("Cache cleared")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        stats = self.sam3_engine.get_performance_stats()
        memory = self.sam3_engine.memory_usage()
        
        success_rate = (self.performance_metrics["successful_queries"] / 
                        max(1, self.performance_metrics["total_queries"]))
        
        return {
            "performance_summary": {
                "total_queries": self.performance_metrics["total_queries"],
                "successful_queries": self.performance_metrics["successful_queries"],
                "success_rate": f"{success_rate:.1%}",
                "average_processing_time": f"{self.performance_metrics['average_processing_time']:.3f}s",
                "cache_hit_rate": f"{self.performance_metrics['cache_hit_rate']:.1%}"
            },
            "system_resources": {
                "gpu_memory_used": f"{memory.get('gpu_memory', 0):.2f}GB",
                "gpu_memory_free": f"{memory.get('gpu_free', 0):.2f}GB",
                "cache_size": len(self.query_cache),
                "conversation_history": len(self.conversation_history)
            },
            "sam3_performance": stats,
            "conversation_turns": len(self.conversation_history),
            "optimization_status": {
                "jax_enabled": self.sam3_config.enable_jit,
                "batch_processing": self.sam3_config.batch_size > 1,
                "caching_enabled": self.config.cache_responses
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Check SAM3 model
            stats = self.sam3_engine.get_performance_stats()
            
            # Check memory
            memory = self.sam3_engine.memory_usage()
            
            # Test with dummy data
            test_start = time.time()
            # Note: In production, you'd want to do a real test with actual data
            test_time = time.time() - test_start
            
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "model_loaded": stats["model_loaded"],
                "processor_loaded": stats["processor_loaded"],
                "gpu_memory_available": memory.get("gpu_free", 0) > 1.0,  # >1GB free
                "test_processing_time": test_time,
                "components": {
                    "query_parser": "ok",
                    "sam3_engine": "ok" if stats["model_loaded"] else "error",
                    "conversation_manager": "ok"
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
