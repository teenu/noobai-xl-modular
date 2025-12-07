"""Search and autocomplete functionality for UI."""

import gradio as gr
from config import logger, SEARCH_CONFIG
from prompt_formatter import get_prompt_data


def search_for_autocomplete(query: str, data_type: str) -> dict:
    """Handle autocomplete search."""
    try:
        if not query or len(query.strip()) < SEARCH_CONFIG.MIN_QUERY_LENGTH:
            return gr.update(choices=[], value=None)

        results = get_prompt_data().search(query, data_type, limit=SEARCH_CONFIG.MAX_RESULTS)
        choices = [f"{'🔴' if r['source'] == 'danbooru' else '🔵'} {r['display']}" for r in results]
        return gr.update(choices=choices, value=choices[0] if choices else None)

    except (AttributeError, KeyError, ValueError) as e:
        logger.error(f"Error in {data_type} search (data error): {e}")
        return gr.update(choices=[], value=None)
    except Exception as e:
        logger.error(f"Unexpected error in {data_type} search: {e}")
        return gr.update(choices=[], value=None)


def select_from_dropdown(search_query: str, selected_choice: str, data_type: str) -> str:
    """Handle dropdown selection."""
    try:
        if not selected_choice or not selected_choice.strip():
            return ""

        clean_trigger = selected_choice[2:].strip()

        data = get_prompt_data()
        if not data.is_loaded or not search_query:
            return clean_trigger

        results = data.search(search_query, data_type, limit=SEARCH_CONFIG.MAX_RESULTS_PER_SOURCE)
        for result in results:
            if result['display'] == clean_trigger:
                return result['value']

        return clean_trigger

    except (AttributeError, KeyError, IndexError) as e:
        logger.error(f"Error in {data_type} selection (data error): {e}")
        return selected_choice or ""
    except Exception as e:
        logger.error(f"Unexpected error in {data_type} selection: {e}")
        return selected_choice or ""


def compose_final_prompt(prefix: str, character: str, artist: str, custom: str) -> str:
    """Compose final prompt from components."""
    from utils import normalize_text
    return ", ".join(filter(None, map(normalize_text, [prefix, character, artist, custom])))
