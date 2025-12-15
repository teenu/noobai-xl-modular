"""Engine management for UI."""

import os
import gc
from threading import Lock
from typing import Optional, Tuple, List
from config import logger, OPTIMAL_SETTINGS, MODEL_SEARCH_PATHS
from state import state_manager, GenerationState, resource_pool
from utils import (
    validate_model_path, validate_dora_path, discover_dora_adapters,
    get_dora_adapter_by_name, detect_adapter_precision, find_dora_path,
    format_file_size, get_user_friendly_error
)
from engine.embedding_manager import discover_embeddings, get_embedding_ui_choices, get_embedding_path_from_selection
from engine import NoobAIEngine

engine: Optional[NoobAIEngine] = None
_engine_lock = Lock()


def is_engine_ready() -> bool:
    """Check if engine is initialized and ready."""
    with _engine_lock:
        return engine is not None and engine.is_initialized


def get_engine_safely() -> Optional[NoobAIEngine]:
    """Get engine instance safely."""
    with _engine_lock:
        return engine


def find_model_path() -> Optional[str]:
    """Search for the model file or directory in common locations."""
    from config import MODEL_CONFIG
    for path in MODEL_SEARCH_PATHS:
        if os.path.exists(path):
            try:
                if os.path.isdir(path):
                    unet_path = os.path.join(path, "unet")
                    vae_path = os.path.join(path, "vae")
                    if os.path.isdir(unet_path) and os.path.isdir(vae_path):
                        return path
                else:
                    file_size = os.path.getsize(path)
                    if file_size > MODEL_CONFIG.MIN_FILE_SIZE_MB * 1024 * 1024:
                        return path
            except Exception:
                continue
    return None


def get_adapter_choices() -> List[str]:
    """Get list of adapter choices."""
    adapters = discover_dora_adapters()
    if adapters:
        return [adapter['name'] for adapter in adapters]
    else:
        return ["None"]


def get_default_adapter_selection() -> str:
    """Get default adapter selection based on availability."""
    adapters = discover_dora_adapters()
    if adapters:
        return adapters[0]['name']
    return "None"


def get_dora_ui_state() -> dict:
    """Get DoRA UI state based on adapter availability."""
    adapters = discover_dora_adapters()
    has_adapters = len(adapters) > 0

    return {
        'enable_dora_interactive': has_adapters,
        'enable_dora_value': has_adapters,
        'dropdown_choices': get_adapter_choices(),
        'dropdown_value': get_default_adapter_selection(),
        'dropdown_interactive': has_adapters,
        'info_message': "Select DoRA adapter from /dora directory" if has_adapters else "No adapters found in /dora directory",
        'checkbox_info': "Load DoRA adapter for enhanced generation" if has_adapters else "No adapters available - install adapters in /dora directory"
    }


def get_embedding_ui_state() -> dict:
    """Get embedding UI state based on embedding availability."""
    embeddings = discover_embeddings()
    has_embeddings = len(embeddings) > 0

    choices, _ = get_embedding_ui_choices()

    return {
        'enable_embeddings_interactive': has_embeddings,
        'enable_embeddings_value': has_embeddings,
        'dropdown_choices': choices,
        'dropdown_value': "None (no embedding)",
        'dropdown_interactive': has_embeddings,
        'has_embeddings': has_embeddings,
        'info_message': "Select embeddings from /embeddings directory" if has_embeddings else "No embeddings found in /embeddings directory",
        'checkbox_info': "Load textual inversion embeddings" if has_embeddings else "No embeddings available - install .safetensors embeddings in /embeddings directory"
    }


def auto_initialize(preferred_model_path: str = None) -> Tuple[str, str, bool, str, str]:
    """Auto-initialize with DoRA defaults."""
    model_path = preferred_model_path if preferred_model_path else find_model_path()

    dora_ui_state = get_dora_ui_state()
    enable_dora = dora_ui_state['enable_dora_value']
    default_adapter = dora_ui_state['dropdown_value']

    if model_path:
        status = initialize_engine(model_path, enable_dora=enable_dora, dora_selection=default_adapter)
        return status, model_path, enable_dora, "", default_adapter

    return ("⚠️ No model found. Please specify path manually.",
            os.path.join(os.getcwd(), "NoobAI-XL-Vpred-v1.0.safetensors"),
            enable_dora, "", default_adapter)


def initialize_engine(model_path: str, enable_dora: bool = False, dora_path: str = "", dora_selection: str = "", embedding_selections: List[str] = None) -> str:
    """Initialize the engine.

    Args:
        model_path: Path to the model file
        enable_dora: Whether to enable DoRA adapter
        dora_path: Manual DoRA path override
        dora_selection: DoRA selection from dropdown
        embedding_selections: List of embedding selections to load
    """
    global engine

    with _engine_lock:
        try:
            current_state = state_manager.get_state()
            if current_state in (GenerationState.GENERATING, GenerationState.INTERRUPTED):
                state_name = current_state.value
                return f"❌ Cannot reinitialize: System is {state_name}. Please wait for current operation to complete."

            if engine is not None:
                logger.info("Performing teardown of previous engine instance")

                try:
                    engine.teardown_engine()
                    del engine
                    engine = None
                    resource_pool.clear()
                    gc.collect()
                    logger.info("Previous engine instance torn down")

                except Exception as e:
                    logger.error(f"Error during engine teardown: {e}")
                    engine = None
                    resource_pool.clear()
                    gc.collect()

            is_valid, validated_model_path = validate_model_path(model_path)
            if not is_valid:
                return f"❌ {validated_model_path}"

            dora_path_to_use = None
            dora_status = ""

            if enable_dora:
                if dora_selection and dora_selection != "None":
                    adapter_info = get_dora_adapter_by_name(dora_selection)
                    if adapter_info:
                        dora_path_to_use = adapter_info['path']
                        dora_status = f"\n🎯 DoRA: {adapter_info['display_name']}"
                    else:
                        dora_status = f"\n⚠️ DoRA: Selected adapter '{dora_selection}' not found"
                elif dora_path.strip():
                    dora_valid, dora_result = validate_dora_path(dora_path)
                    if dora_valid:
                        dora_path_to_use = dora_result
                        precision = detect_adapter_precision(dora_result)
                        dora_status = f"\n🎯 DoRA: {os.path.basename(dora_result)} ({precision})"
                    else:
                        dora_status = f"\n⚠️ DoRA Error: {dora_result}"
                else:
                    auto_dora_path = find_dora_path()
                    if auto_dora_path:
                        dora_path_to_use = auto_dora_path
                        precision = detect_adapter_precision(auto_dora_path)
                        dora_status = f"\n🎯 DoRA: {os.path.basename(auto_dora_path)} ({precision}, auto-detected)"
                    else:
                        dora_status = "\n⚠️ DoRA: Enabled but no valid DoRA file found"

            engine = NoobAIEngine(
                model_path=validated_model_path,
                enable_dora=enable_dora,
                dora_path=dora_path_to_use,
                dora_start_step=OPTIMAL_SETTINGS['dora_start_step']
            )

            # Load embeddings if specified
            embedding_status = ""
            if embedding_selections:
                loaded_embeddings = []
                for selection in embedding_selections:
                    if selection and selection not in ["None (no embedding)", "No embeddings found"]:
                        embedding_path = get_embedding_path_from_selection(selection)
                        if embedding_path and engine.load_embedding(embedding_path):
                            # Get just the name from the selection
                            name = selection.split(" (")[0] if " (" in selection else selection
                            loaded_embeddings.append(name)
                if loaded_embeddings:
                    embedding_status = f"\n📎 Embeddings: {', '.join(loaded_embeddings)}"

            if os.path.isdir(validated_model_path):
                model_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(validated_model_path)
                    for filename in filenames
                )
            else:
                model_size = os.path.getsize(validated_model_path)

            status_msg = f"✅ Engine initialized!\n📊 Model: {format_file_size(model_size)}{dora_status}{embedding_status}"

            return status_msg

        except (IOError, OSError) as e:
            engine = None
            error_msg = get_user_friendly_error(e)
            logger.error(f"Engine initialization failed (file error): {e}")
            return f"❌ Initialization failed: {error_msg}"
        except (RuntimeError, ValueError) as e:
            engine = None
            error_msg = get_user_friendly_error(e)
            logger.error(f"Engine initialization failed (runtime/validation error): {e}")
            return f"❌ Initialization failed: {error_msg}"
        except Exception as e:
            engine = None
            error_msg = get_user_friendly_error(e)
            logger.error(f"Unexpected error during engine initialization: {e}")
            return f"❌ Initialization failed: {error_msg}"
