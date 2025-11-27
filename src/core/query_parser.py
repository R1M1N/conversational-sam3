"""
Natural Language Query Parser for SAM3 (VLM-Powered)

This module uses a Vision-Language Model (Qwen3-VL) to convert natural language 
queries into SAM3-optimized prompts, handling complex queries, relationships, 
and image context for high-performance segmentation.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

logger = logging.getLogger(__name__)

class QueryType(Enum):
    SIMPLE_SEGMENTATION = "simple_segmentation"
    MULTI_OBJECT = "multi_object"
    CONDITIONAL = "conditional"
    TRACKING = "tracking"
    COMPARATIVE = "comparative"
    DESCRIPTIVE = "descriptive"
    VLM_PARSED = "vlm_parsed"

class ConstraintType(Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    FILTER = "filter"
    TEMPORAL = "temporal"

@dataclass
class QueryEntity:
    text: str
    aliases: List[str]
    confidence: float = 1.0
    entity_type: str = "object"

@dataclass
class QueryConstraint:
    type: ConstraintType
    condition: str
    verification_prompt: str
    entities_affected: Optional[List[str]] = None

@dataclass
class ParsedQuery:
    query_type: QueryType
    entities: List[QueryEntity]
    constraints: List[QueryConstraint]
    optimization_hints: Dict[str, Any]
    original_query: str
    confidence: float

class SAM3QueryParser:
    """
    Advanced query parser for SAM3 conversational agent, powered by Qwen3-VL.
    """
    
    def __init__(self):
        # --- VLM-powered parser setup ---
        try:
            logger.info("Initializing VLM-powered query parser (Qwen3-VL-2B)...")
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

            self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen3-VL-2B-Instruct",
                quantization_config=quantization_config,
                device_map="auto",
            )
            
            self.vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
            self.vlm_enabled = True
            logger.info("✅ Qwen3-VL model loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load Qwen3-VL model: {e}. Falling back to regex parser.")
            self.vlm_enabled = False

        # --- Regex-based fallback patterns ---
        self.object_patterns = {
            "person": ["person", "people"], "car": ["car", "vehicle"],
            "dog": ["dog", "puppy"], "cat": ["cat", "kitten"],
            "building": ["building", "house"], "tree": ["tree", "plant"],
        }
        
        logger.info("SAM3QueryParser initialized.")
    
    def _parse_with_vlm(self, query: str, image_path: Optional[str] = None) -> Optional[Dict]:
        """Use Qwen3-VL to convert a natural query into structured JSON."""
        if not self.vlm_enabled:
            return None

        messages = [{
            "role": "user",
            "content": [
                *([{"type": "image"}] if image_path else []),
                {
                    "type": "text",
                    "text": (
                        "You are a vision-language parser for a segmentation agent.\n"
                        "Given the user request below, extract:\n"
                        "1) entities: list of objects to segment (e.g., [\"grapes\", \"leaf\"])\n"
                        "2) constraints: list of natural-language constraints (e.g., [\"only the purple ones\"])\n"
                        "Return ONLY valid JSON with keys: entities, constraints.\n\n"
                        f"User request: {query}"
                    ),
                },
            ],
        }]

        text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [image_path] if image_path else None
        inputs = self.vlm_processor(text=[text], images=images, return_tensors="pt").to(self.vlm_model.device)

        with torch.no_grad():
            generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=256)

        out_ids = generated_ids[0, inputs.input_ids.shape[1]:]
        raw = self.vlm_processor.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            parsed_json = json.loads(raw[start:end])
            # Validate essential key
            if "entities" in parsed_json and isinstance(parsed_json["entities"], list):
                return parsed_json
            return None
        except (ValueError, json.JSONDecodeError):
            logger.warning("VLM did not return valid JSON, falling back to regex.")
            return None

    def parse_query(self, natural_query: str, image_path: Optional[str] = None) -> ParsedQuery:
        """
        Parse natural language query into SAM3-optimized structure.
        Tries VLM first, then falls back to regex.
        """
        logger.info(f"Parsing query: {natural_query}")
        
        normalized_query = self._normalize_query(natural_query)
        vlm_struct = self._parse_with_vlm(normalized_query, image_path)

        if vlm_struct and vlm_struct.get("entities"):
            # VLM succeeded
            logger.info(f"VLM parsed entities: {vlm_struct['entities']}")
            entities = [
                QueryEntity(text=e, aliases=[e], confidence=0.9, entity_type="vlm_object")
                for e in vlm_struct["entities"]
            ]
            # Simple constraint mapping
            constraints = [
                QueryConstraint(
                    type=ConstraintType.FILTER,
                    condition=c,
                    verification_prompt=f"Does this object satisfy the condition: {c}?",
                )
                for c in vlm_struct.get("constraints", [])
            ]
            query_type = QueryType.VLM_PARSED
        else:
            # Fallback to regex
            logger.info("Falling back to regex-based parser.")
            query_type = self._detect_query_type(normalized_query)
            entities = self._extract_entities_regex(normalized_query)
            constraints = self._extract_constraints_regex(normalized_query, entities)

        optimization_hints = self._generate_optimization_hints(normalized_query, entities, constraints)
        confidence = self._calculate_confidence(entities, constraints)
        
        parsed_query = ParsedQuery(
            query_type=query_type,
            entities=entities,
            constraints=constraints,
            optimization_hints=optimization_hints,
            original_query=natural_query,
            confidence=confidence
        )
        
        logger.info(f"Parsed query with {len(entities)} entities, {len(constraints)} constraints")
        return parsed_query

    def _normalize_query(self, query: str) -> str:
        return query.lower().strip()

    def _detect_query_type(self, query: str) -> QueryType:
        if " and " in query or "," in query: return QueryType.MULTI_OBJECT
        if any(w in query for w in ["but not", "except", "without"]): return QueryType.CONDITIONAL
        if any(w in query for w in ["track", "follow"]): return QueryType.TRACKING
        return QueryType.SIMPLE_SEGMENTATION

    def _extract_entities_regex(self, query: str) -> List[QueryEntity]:
        """Fallback regex entity extractor."""
        entities = []
        for obj_type, obj_words in self.object_patterns.items():
            for obj_word in obj_words:
                if obj_word in query:
                    entities.append(QueryEntity(text=obj_type, aliases=obj_words, confidence=0.7))
                    break
        
        # Generic fallback for unknown nouns
        if not entities:
            match = re.search(r'(?:segment|find|detect|locate)\s+(?:the\s+)?([\w\s]+)', query)
            if match:
                entity_text = match.group(1).split(" in ")[0].strip()
                if entity_text:
                    entities.append(QueryEntity(text=entity_text, aliases=[entity_text], confidence=0.6, entity_type="fallback_object"))

        return self._merge_similar_entities(entities)

    def _merge_similar_entities(self, entities: List[QueryEntity]) -> List[QueryEntity]:
        merged = {}
        for entity in entities:
            if entity.text not in merged:
                merged[entity.text] = entity
        return list(merged.values())

    def _extract_constraints_regex(self, query: str, entities: List[QueryEntity]) -> List[QueryConstraint]:
        # Simplified for brevity
        return []
    
    def _generate_optimization_hints(self, query: str, entities: List, constraints: List) -> Dict:
        return {"batch_mode": len(entities) > 1}
    
    def _calculate_confidence(self, entities: List, constraints: List) -> float:
        if not entities: return 0.0
        return sum(e.confidence for e in entities) / len(entities)

    def convert_to_sam3_prompts(self, parsed_query: ParsedQuery) -> List[str]:
        prompts = [entity.text for entity in parsed_query.entities]
        logger.info(f"Generated {len(prompts)} SAM3 prompts: {prompts}")
        return prompts

    def generate_processing_plan(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        return {"steps": [{"action": "segment_with_sam3"}]}
