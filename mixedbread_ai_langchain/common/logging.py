"""Centralized logging utilities for mixedbread-ai-langchain components."""

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for mixedbread-ai-langchain components.
    
    Args:
        name: Logger name. If None, uses the calling module's __name__.
        
    Returns:
        Configured logger instance.
    """
    if name is None:
        # Get the calling module's name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'mixedbread_ai_langchain')
        else:
            name = 'mixedbread_ai_langchain'
    
    return logging.getLogger(name)