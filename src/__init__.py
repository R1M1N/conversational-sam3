"""
SAM3 Conversational Agent - Source Package

High-performance conversational agent for SAM3 segmentation operations
with JAX acceleration and natural language understanding.
"""

__version__ = "1.0.0"
__author__ = "MiniMax Agent"
__description__ = "High-performance SAM3 Conversational Agent"

# Core exports
from .core.sam3_engine import SAM3JAXEngine, SAM3Config, ProcessingResult
from .core.query_parser import SAM3QueryParser, ParsedQuery, QueryType, QueryEntity, QueryConstraint
from .agent.conversational_agent import SAM3ConversationalAgent, AgentConfig, ConversationTurn

__all__ = [
    # Core engine
    "SAM3JAXEngine",
    "SAM3Config", 
    "ProcessingResult",
    
    # Query parser
    "SAM3QueryParser",
    "ParsedQuery",
    "QueryType",
    "QueryEntity", 
    "QueryConstraint",
    
    # Conversational agent
    "SAM3ConversationalAgent",
    "AgentConfig",
    "ConversationTurn"
]