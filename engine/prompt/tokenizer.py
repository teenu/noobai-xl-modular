"""Token counting and prompt validation for SDXL dual text encoder."""

import re
import logging
from typing import Tuple, Optional, Any
from config import logger

# Get HuggingFace tokenizer logger to suppress long sequence warnings
_hf_tokenizer_logger = logging.getLogger("transformers.tokenization_utils_base")


class TokenManager:
    """Manages token counting for SDXL dual text encoder.

    SDXL uses two CLIP text encoders:
    - CLIP-L (tokenizer): Standard CLIP tokenizer
    - OpenCLIP-G (tokenizer_2): Larger OpenCLIP tokenizer

    Both have a 77-token context limit. This class provides accurate
    token counting for UI feedback and prompt analysis.
    """

    CLIP_TOKEN_LIMIT = 77
    CHUNK_SIZE = 75  # 77 - 2 for BOS/EOS tokens per chunk

    def __init__(self, tokenizer_1: Any, tokenizer_2: Any):
        """Initialize with both SDXL tokenizers.

        Args:
            tokenizer_1: CLIP-L tokenizer from pipeline.tokenizer
            tokenizer_2: OpenCLIP-G tokenizer from pipeline.tokenizer_2
        """
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2

    def count_tokens(self, text: str) -> Tuple[int, int]:
        """Count tokens for both encoders.

        Removes A1111 weight syntax before counting for accuracy,
        since weight modifiers like (text:1.2) are not actual tokens.

        Args:
            text: The prompt text to count tokens for

        Returns:
            Tuple of (clip_l_tokens, openclip_g_tokens)
        """
        if not text or not text.strip():
            return (0, 0)

        # Remove weight syntax for accurate counting
        clean_text = self._strip_weights(text)

        try:
            # Temporarily suppress HuggingFace tokenizer warning about sequence length
            # "Token indices sequence length is longer than the specified maximum sequence length"
            # This warning is expected when counting tokens for long prompts with truncation=False
            # Since sd_embed handles chunking, the warning is not relevant for our use case
            original_level = _hf_tokenizer_logger.level
            _hf_tokenizer_logger.setLevel(logging.ERROR)

            try:
                # Get token count from CLIP-L tokenizer
                tokens_1 = self.tokenizer_1(
                    clean_text,
                    truncation=False,
                    add_special_tokens=True,
                    return_tensors=None
                )
                count_1 = len(tokens_1['input_ids'])

                # Get token count from OpenCLIP-G tokenizer
                tokens_2 = self.tokenizer_2(
                    clean_text,
                    truncation=False,
                    add_special_tokens=True,
                    return_tensors=None
                )
                count_2 = len(tokens_2['input_ids'])
            finally:
                # Restore original logger level
                _hf_tokenizer_logger.setLevel(original_level)

            return (count_1, count_2)

        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            return (0, 0)

    def get_chunk_count(self, text: str) -> int:
        """Calculate number of 75-token chunks needed.

        Each chunk can hold 75 tokens of actual content,
        with 2 tokens reserved for BOS/EOS markers.

        Args:
            text: The prompt text

        Returns:
            Number of chunks needed (minimum 1)
        """
        clip_l, openclip_g = self.count_tokens(text)
        # Use the larger token count to determine chunks
        max_tokens = max(clip_l, openclip_g)
        # Account for BOS/EOS tokens already included in count
        effective_tokens = max(0, max_tokens - 2)  # Subtract BOS/EOS
        return max(1, (effective_tokens + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE)

    def is_long_prompt(self, text: str) -> bool:
        """Check if prompt exceeds single-chunk limit (77 tokens).

        Args:
            text: The prompt text

        Returns:
            True if prompt requires multiple chunks
        """
        clip_l, openclip_g = self.count_tokens(text)
        return max(clip_l, openclip_g) > self.CLIP_TOKEN_LIMIT

    def get_status_info(self, text: str) -> dict:
        """Get detailed token status for UI display.

        Args:
            text: The prompt text

        Returns:
            Dictionary with token information:
            - clip_l_tokens: Token count for CLIP-L encoder
            - openclip_g_tokens: Token count for OpenCLIP-G encoder
            - max_tokens: Maximum of both counts
            - chunks: Number of 75-token chunks needed
            - is_long: Whether prompt exceeds 77 tokens
            - warning: Warning message if very long prompt, else None
        """
        clip_l, openclip_g = self.count_tokens(text)
        max_tokens = max(clip_l, openclip_g)
        chunks = self.get_chunk_count(text)

        warning = None
        if chunks > 8:
            warning = f"Very long prompt ({chunks} chunks, ~{max_tokens} tokens) may affect quality"
        elif chunks > 4:
            warning = f"Long prompt ({chunks} chunks) - quality may vary"

        return {
            'clip_l_tokens': clip_l,
            'openclip_g_tokens': openclip_g,
            'max_tokens': max_tokens,
            'chunks': chunks,
            'is_long': max_tokens > self.CLIP_TOKEN_LIMIT,
            'warning': warning
        }

    @staticmethod
    def _strip_weights(text: str) -> str:
        """Remove A1111 weight syntax for accurate token counting.

        Handles:
        - (text:1.2) -> text (explicit weight)
        - ((text)) -> text (nested emphasis)
        - [text] -> text (de-emphasis)
        - Nested weights like (outer (inner:1.2):0.9) -> outer inner

        Args:
            text: Text with potential weight syntax

        Returns:
            Text with weight syntax removed
        """
        if not text:
            return ""

        # Iteratively remove explicit weights until no more matches
        # This handles nested weights like (outer (inner:1.2):0.9)
        pattern = r'\(([^():]+):[\d.]+\)'
        while re.search(pattern, text):
            prev = text
            text = re.sub(pattern, r'\1', text)
            if text == prev:  # Prevent infinite loop on malformed input
                break

        # Remove nested emphasis: ((text)) -> text, (((text))) -> text
        while '((' in text or '))' in text:
            prev = text
            text = re.sub(r'\(([^()]+)\)', r'\1', text)
            if text == prev:  # Prevent infinite loop on malformed input
                break

        # Remove remaining single parentheses (shouldn't normally happen)
        text = re.sub(r'\(([^()]+)\)', r'\1', text)

        # Remove de-emphasis brackets: [text] -> text
        text = re.sub(r'\[([^\[\]]+)\]', r'\1', text)

        return text.strip()
