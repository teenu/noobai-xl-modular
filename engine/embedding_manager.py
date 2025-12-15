"""NoobAI XL V-Pred 1.0 - Embedding Manager for Textual Inversions.

This module handles discovery, loading, and management of textual inversion
embeddings (CLIP embeddings stored in safetensors format).
"""

import os
from typing import List, Dict, Optional, Any
from config import (
    logger, EMBEDDING_CONFIG, EMBEDDING_SEARCH_DIRECTORIES,
    QUALITY_EMBEDDING_TRIGGERS, NEGATIVE_EMBEDDING_TRIGGERS
)


class EmbeddingManager:
    """Manages textual inversion embeddings for the diffusion pipeline."""

    def __init__(self, pipe):
        """Initialize the embedding manager.

        Args:
            pipe: The diffusion pipeline (StableDiffusionXLPipeline)
        """
        self.pipe = pipe
        self.loaded_embeddings: Dict[str, str] = {}  # token -> path

    def load_embedding(self, embedding_path: str, token: Optional[str] = None) -> bool:
        """Load a textual inversion embedding into the pipeline.

        Args:
            embedding_path: Path to the .safetensors embedding file
            token: Optional custom trigger token. If None, derived from filename.

        Returns:
            True if loading succeeded, False otherwise
        """
        if not os.path.exists(embedding_path):
            logger.error(f"Embedding file not found: {embedding_path}")
            return False

        if token is None:
            # Derive token from filename (without extension)
            token = os.path.splitext(os.path.basename(embedding_path))[0]
            # Handle -neg suffix convention - keep the full name as trigger
            # but also register a version without -neg for convenience
            if token.endswith("-neg"):
                base_token = token[:-4]  # Version without -neg suffix
            else:
                base_token = None

        try:
            self.pipe.load_textual_inversion(embedding_path, token=token)
            self.loaded_embeddings[token] = embedding_path
            logger.info(f"Loaded embedding '{token}' from {embedding_path}")

            # If this is a -neg file, also register the base name as an alias
            if base_token and base_token not in self.loaded_embeddings:
                try:
                    self.pipe.load_textual_inversion(embedding_path, token=base_token)
                    self.loaded_embeddings[base_token] = embedding_path
                    logger.info(f"Also registered alias '{base_token}' for embedding")
                except Exception as e:
                    logger.debug(f"Could not register alias '{base_token}': {e}")

            return True

        except Exception as e:
            logger.error(f"Failed to load embedding '{token}' from {embedding_path}: {e}")
            return False

    def load_embeddings(self, embedding_paths: List[str]) -> int:
        """Load multiple embeddings.

        Args:
            embedding_paths: List of paths to embedding files

        Returns:
            Number of successfully loaded embeddings
        """
        loaded_count = 0
        for path in embedding_paths:
            if self.load_embedding(path):
                loaded_count += 1
        return loaded_count

    def is_loaded(self, token: str) -> bool:
        """Check if an embedding with the given token is loaded."""
        return token in self.loaded_embeddings

    def get_loaded_tokens(self) -> List[str]:
        """Get list of all loaded embedding tokens."""
        return list(self.loaded_embeddings.keys())

    def has_quality_embedding(self, prompt: str) -> bool:
        """Check if prompt contains a loaded quality embedding trigger.

        Args:
            prompt: The positive prompt to check

        Returns:
            True if a quality embedding trigger is found and loaded
        """
        for trigger in QUALITY_EMBEDDING_TRIGGERS:
            if trigger in prompt and self.is_loaded(trigger):
                return True
        return False

    def has_negative_embedding(self, prompt: str) -> bool:
        """Check if prompt contains a loaded negative embedding trigger.

        Args:
            prompt: The negative prompt to check

        Returns:
            True if a negative embedding trigger is found and loaded
        """
        for trigger in NEGATIVE_EMBEDDING_TRIGGERS:
            if trigger in prompt and self.is_loaded(trigger):
                return True
        return False

    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about loaded embeddings."""
        return {
            'loaded_count': len(self.loaded_embeddings),
            'tokens': list(self.loaded_embeddings.keys()),
            'paths': list(self.loaded_embeddings.values())
        }


def discover_embeddings() -> List[Dict[str, Any]]:
    """Discover available embedding files in search directories.

    Returns:
        List of dicts with embedding info: name, path, size, is_negative
    """
    embeddings = []
    seen_paths = set()

    for directory in EMBEDDING_SEARCH_DIRECTORIES:
        if not os.path.isdir(directory):
            continue

        try:
            for file in os.listdir(directory):
                if not file.endswith('.safetensors'):
                    continue

                path = os.path.join(directory, file)

                # Skip duplicates
                if path in seen_paths:
                    continue
                seen_paths.add(path)

                try:
                    size_bytes = os.path.getsize(path)
                    size_kb = size_bytes / 1024

                    # Check size bounds (embeddings are small, typically < 1MB)
                    if not (EMBEDDING_CONFIG.MIN_FILE_SIZE_KB <= size_kb <= EMBEDDING_CONFIG.MAX_FILE_SIZE_MB * 1024):
                        continue

                    name = os.path.splitext(file)[0]
                    is_negative = name.endswith("-neg") or "negative" in name.lower()

                    embeddings.append({
                        'name': name,
                        'path': path,
                        'size': f"{size_kb:.1f} KB",
                        'size_bytes': size_bytes,
                        'is_negative': is_negative,
                        'display_name': f"{name} ({size_kb:.1f} KB)"
                    })

                except (OSError, IOError) as e:
                    logger.debug(f"Could not read embedding file {path}: {e}")

        except (OSError, IOError) as e:
            logger.debug(f"Could not list directory {directory}: {e}")

    # Sort by name
    embeddings.sort(key=lambda x: x['name'].lower())
    return embeddings


def get_embedding_ui_choices() -> tuple:
    """Get embedding choices formatted for Gradio dropdown.

    Returns:
        Tuple of (choices_list, has_embeddings)
    """
    embeddings = discover_embeddings()

    if not embeddings:
        return (["No embeddings found"], False)

    choices = ["None (no embedding)"]
    for emb in embeddings:
        choices.append(emb['display_name'])

    return (choices, True)


def get_embedding_path_from_selection(selection: str) -> Optional[str]:
    """Get the file path for a selected embedding from dropdown.

    Args:
        selection: The dropdown selection string

    Returns:
        Path to the embedding file, or None if not found/applicable
    """
    if not selection or selection == "None (no embedding)" or selection == "No embeddings found":
        return None

    embeddings = discover_embeddings()

    for emb in embeddings:
        if emb['display_name'] == selection or emb['name'] == selection:
            return emb['path']

    return None
