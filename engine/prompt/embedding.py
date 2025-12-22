"""Embedding generation with long prompt support via sd_embed.

This module provides a wrapper around the sd_embed library for generating
text embeddings that support prompts longer than the 77-token CLIP limit.

The sd_embed library handles:
- Automatic chunking of long prompts into 75-token segments
- A1111-compatible weight syntax: (tag:1.2), [tag], ((emphasis))
- Proper handling of both SDXL text encoders (CLIP-L and OpenCLIP-G)
- Correct generation of pooled embeddings required by SDXL

If sd_embed is not available, a fallback mode uses the pipeline's native
encoding which truncates at 77 tokens.
"""

from typing import Tuple, Any, Optional
import torch
from config import logger

# Try to import sd_embed, set availability flag
try:
    from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl
    SD_EMBED_AVAILABLE = True
except ImportError:
    SD_EMBED_AVAILABLE = False
    logger.info("sd_embed not installed - long prompt support will use fallback mode")


class EmbeddingGenerator:
    """Generate embeddings for SDXL with long prompt support.

    This class wraps sd_embed's get_weighted_text_embeddings_sdxl function
    to provide:
    - Unlimited prompt length via automatic chunking
    - A1111 weight syntax support: (tag:1.2), [tag], ((emphasis))
    - Graceful fallback to standard encoding if sd_embed fails

    Usage:
        generator = EmbeddingGenerator(pipe)
        embeds = generator.generate(prompt, negative_prompt)
        # Pass embeds to pipeline instead of raw prompt strings
    """

    def __init__(self, pipe: Any):
        """Initialize with SDXL pipeline.

        Args:
            pipe: StableDiffusionXLPipeline instance with text encoders
        """
        self.pipe = pipe
        self._fallback_mode = not SD_EMBED_AVAILABLE

        if self._fallback_mode:
            logger.warning(
                "Long prompt support disabled - sd_embed not available. "
                "Install with: pip install sd-embed"
            )

    def generate(
        self,
        prompt: str,
        negative_prompt: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate embeddings for prompt and negative prompt.

        Handles long prompts by chunking and concatenating embeddings.
        Supports A1111 weight syntax: (tag:1.2), [tag], ((emphasis))

        Args:
            prompt: The positive prompt text
            negative_prompt: The negative prompt text

        Returns:
            Tuple of 4 tensors:
            - prompt_embeds: Encoded positive prompt
            - negative_prompt_embeds: Encoded negative prompt
            - pooled_prompt_embeds: Pooled positive embedding
            - negative_pooled_prompt_embeds: Pooled negative embedding
        """
        if self._fallback_mode:
            return self._fallback_encode(prompt, negative_prompt)

        try:
            # Use sd_embed for long prompt handling
            (prompt_embeds,
             negative_prompt_embeds,
             pooled_prompt_embeds,
             negative_pooled_prompt_embeds) = get_weighted_text_embeddings_sdxl(
                self.pipe,
                prompt=prompt,
                neg_prompt=negative_prompt
            )

            return (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds
            )

        except Exception as e:
            logger.warning(f"sd_embed encoding failed, using fallback: {e}")
            return self._fallback_encode(prompt, negative_prompt)

    def _fallback_encode(
        self,
        prompt: str,
        negative_prompt: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fallback to standard pipeline encoding.

        This method uses the pipeline's built-in encode_prompt method,
        which will truncate prompts at 77 tokens per encoder.

        Args:
            prompt: The positive prompt text
            negative_prompt: The negative prompt text

        Returns:
            Tuple of 4 tensors (same as generate method)
        """
        try:
            # Determine the device to use
            device = self.pipe.device if hasattr(self.pipe, 'device') else 'cuda'
            if hasattr(device, 'type'):
                device = device

            # Use pipeline's internal encoding method
            # This handles both text encoders but truncates at 77 tokens
            (prompt_embeds,
             negative_prompt_embeds,
             pooled_prompt_embeds,
             negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,  # Same prompt for both encoders
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )

            return (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds
            )

        except Exception as e:
            logger.error(f"Fallback encoding also failed: {e}")
            raise RuntimeError(f"Both sd_embed and fallback encoding failed: {e}")

    @property
    def is_long_prompt_supported(self) -> bool:
        """Check if long prompt support is available.

        Returns:
            True if sd_embed is available and not in fallback mode
        """
        return SD_EMBED_AVAILABLE and not self._fallback_mode

    @property
    def mode_description(self) -> str:
        """Get a human-readable description of the current mode.

        Returns:
            Description string for logging/display
        """
        if self.is_long_prompt_supported:
            return "sd_embed (unlimited length, A1111 weight syntax)"
        else:
            return "fallback (77-token limit, no weight syntax)"
