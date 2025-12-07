"""UI Package - Gradio interface components."""

# Import from refactored modules
from ui.search_helpers import search_for_autocomplete, select_from_dropdown, compose_final_prompt
from ui.validation import validate_parameters, parse_resolution_string
from ui.widgets import create_clear_handler, create_status_updater, create_search_ui, connect_search_events
from ui.engine_manager import (
    is_engine_ready, get_engine_safely, initialize_engine,
    auto_initialize, get_dora_ui_state, find_model_path
)
from ui.generation import (
    start_generation, generate_image_with_progress,
    finish_generation, interrupt_generation
)

# Import main interface from original ui.py (until it's split)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui import create_interface

__all__ = [
    'create_interface',
    'search_for_autocomplete',
    'select_from_dropdown',
    'compose_final_prompt',
    'validate_parameters',
    'parse_resolution_string',
    'create_clear_handler',
    'create_status_updater',
    'create_search_ui',
    'connect_search_events',
    'is_engine_ready',
    'get_engine_safely',
    'initialize_engine',
    'auto_initialize',
    'get_dora_ui_state',
    'find_model_path',
    'start_generation',
    'generate_image_with_progress',
    'finish_generation',
    'interrupt_generation'
]
