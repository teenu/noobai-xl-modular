"""Model loading and initialization."""

import os
from typing import Optional
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    EulerDiscreteScheduler,
    AutoencoderKL,
    ControlNetModel
)
from config import logger
from utils import detect_base_model_precision

# Set by detect_device(); read by load_pipeline() and engine/core.py to gate
# Blackwell-specific code paths (SageAttention, observability logging, etc.)
_BLACKWELL_DETECTED: bool = False

# Set by load_pipeline() when SageAttention is successfully installed on the UNet.
# Signals core.py to skip torch.compile (the two are mutually exclusive).
_SAGE_ATTENTION_ACTIVE: bool = False


def is_sage_attention_active() -> bool:
    """Return whether SageAttention is active for the current runtime."""
    return _SAGE_ATTENTION_ACTIVE


def detect_device() -> str:
    """Detect available device for inference.

    Priority: CUDA > MPS > CPU. CUDA is checked first so that systems with a
    discrete NVIDIA GPU are never silently routed to a slower backend.
    """
    global _BLACKWELL_DETECTED
    _BLACKWELL_DETECTED = False
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 0:
            cc = torch.cuda.get_device_capability(0)
            _BLACKWELL_DETECTED = cc >= (12, 0)
            if _BLACKWELL_DETECTED:
                try:
                    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
                except (ValueError, AttributeError):
                    major, minor = (0, 0)
                if (major, minor) < (2, 7):
                    logger.warning(
                        f"Blackwell GPU (sm_{cc[0]}{cc[1]}) detected but PyTorch {torch.__version__} "
                        f"< 2.7. Upgrade to PyTorch 2.7+ for full sm_120 native support."
                    )
                logger.info(f"Blackwell GPU detected (sm_{cc[0]}{cc[1]}): Blackwell-optimized paths active")
            else:
                logger.info(f"CUDA GPU detected (sm_{cc[0]}{cc[1]})")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_bf16_support(device: str) -> bool:
    """Check if device supports bfloat16."""
    if device == "cuda":
        try:
            compute_capability = torch.cuda.get_device_capability(0)
            return compute_capability[0] >= 8
        except (AttributeError, RuntimeError):
            return False
    elif device == "mps":
        return True
    elif device == "cpu":
        return False
    return False


def load_pipeline(model_path: str, device: str, force_fp32: bool = False, optimize: bool = False) -> tuple:
    """Load diffusion pipeline with proper precision handling.

    Args:
        model_path: Path to model file or directory
        device: Target device (cuda/mps/cpu)
        force_fp32: If True, force FP32 inference even for BF16 models (parity mode)
        optimize: If True, enable torch.compile for faster inference (TF32 is always on)

    Returns:
        tuple: (pipeline, cpu_offload_enabled)
    """
    global _SAGE_ATTENTION_ACTIVE
    _SAGE_ATTENTION_ACTIVE = False

    # TF32 is always enabled on CUDA — safe on all Ampere+ GPUs (compute capability >= 8.0),
    # including Blackwell (sm_120). Provides significant speedup with negligible precision impact.
    if device == "cuda":
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.debug("TF32 enabled for CUDA matmul and cuDNN")

    if optimize and device == "cuda":
        # Enable persistent inductor cache for faster subsequent runs
        import tempfile
        cache_dir = os.path.join(tempfile.gettempdir(), "noobai_inductor_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', cache_dir)
    base_precision = detect_base_model_precision(model_path)
    is_directory = os.path.isdir(model_path)

    if base_precision not in [torch.bfloat16, torch.float32]:
        raise ValueError(
            f"Model precision validation failed: got {base_precision}. "
            f"Only BF16 model or FP32 pre-converted models are supported."
        )

    if base_precision == torch.float32 and is_directory:
        vae_path = os.path.join(model_path, "vae")
        if os.path.isdir(vae_path):
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float32)
            logger.info("VAE loaded as FP32 from directory for lossless decode")
        else:
            raise ValueError(
                f"VAE subdirectory not found at {vae_path}. "
                f"FP32 model directories must contain a 'vae' subdirectory."
            )

        pipe = StableDiffusionXLPipeline.from_pretrained(model_path, vae=vae)

        actual_unet_dtype = next(pipe.unet.parameters()).dtype
        actual_te_dtype = next(pipe.text_encoder.parameters()).dtype
        actual_vae_dtype = next(pipe.vae.parameters()).dtype

        if actual_unet_dtype != torch.float32:
            raise ValueError(
                f"Expected FP32 model but UNet loaded with {actual_unet_dtype}. "
                f"Directory may contain wrong precision weights."
            )
        if actual_te_dtype != torch.float32:
            raise ValueError(
                f"Expected FP32 model but TextEncoder loaded with {actual_te_dtype}. "
                f"Directory may contain wrong precision weights."
            )
        if actual_vae_dtype != torch.float32:
            raise ValueError(
                f"Expected FP32 VAE but loaded with {actual_vae_dtype}. "
                f"This could cause decode quality issues."
            )

        logger.info("FP32 directory model loaded with all components validated as FP32")
    else:
        # Single file loading (safetensors)
        bf16_supported = check_bf16_support(device)

        if force_fp32 or not bf16_supported:
            # Parity mode or unsupported device: Load all components in FP32
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float32,
                use_safetensors=True,
            )
            if force_fp32:
                logger.info("Parity mode: All components loaded as FP32")
            else:
                logger.info("Device does not support BF16; all components loaded as FP32")
        else:
            # BF16 mode with FP32 VAE: single-pass loading
            # Load everything in BF16, then upcast only the VAE in-place.
            # VAE weights are ~84MB BF16 -> ~168MB FP32 — negligible on systems with
            # sufficient VRAM. This avoids the ~25-30GB RAM spike of the old two-pass
            # approach (FP32 full load -> delete -> BF16 reload).
            logger.info("Loading pipeline at BF16 with in-place FP32 VAE (single-pass)...")
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
            )

            # Upcast VAE to FP32 in-place for lossless decode
            pipe.vae = pipe.vae.to(torch.float32)

            # Wrap VAE decode to cast BF16 latents to FP32 before decoding.
            # diffusers SDXL pipeline only auto-upcasts latents when the VAE dtype
            # is float16, not when it is float32 with BF16 latents coming in.
            original_decode = pipe.vae.decode

            def decode_with_upcast(latents, *args, **kwargs):
                if latents.dtype != torch.float32:
                    latents = latents.to(dtype=torch.float32)
                return original_decode(latents, *args, **kwargs)

            pipe.vae.decode = decode_with_upcast
            logger.info("BF16 pipeline loaded with in-place FP32 VAE (single-pass, lossless)")

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        prediction_type="v_prediction",
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing"
    )

    cpu_offload_enabled = False
    if device == "cuda":
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 8.0:
                pipe.enable_sequential_cpu_offload()
                cpu_offload_enabled = True
                logger.info(f"CPU offloading enabled ({vram_gb:.1f}GB VRAM)")
            else:
                pipe = pipe.to(device)
        except Exception as e:
            logger.debug(f"VRAM check failed, moving pipeline to device directly: {e}")
            pipe = pipe.to(device)
    else:
        pipe = pipe.to(device)

    # VAE slicing splits the image into horizontal strips for sequential processing.
    # On high-VRAM systems this is slower and can introduce seam artifacts. Only
    # enable it when VRAM is genuinely constrained.
    if device == "cuda":
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            vram_gb = 0.0
        if vram_gb < 12.0:
            pipe.vae.enable_slicing()
            logger.info(f"VAE slicing enabled ({vram_gb:.1f}GB VRAM)")
        else:
            logger.info(f"VAE slicing disabled ({vram_gb:.1f}GB VRAM — not needed)")
    else:
        pipe.vae.enable_slicing()  # Always enable for MPS/CPU

    # Native SDPA attention via AttnProcessor2_0 — dispatches to Flash Attention kernels
    # on supported hardware (Blackwell, Ada, Ampere) via PyTorch's scaled_dot_product_attention.
    # Safe with DoRA: PEFT LoRA hooks into nn.Linear weights, not the attention processor.
    if device == "cuda":
        from diffusers.models.attention_processor import AttnProcessor2_0
        pipe.unet.set_attn_processor(AttnProcessor2_0())
        logger.info("Native SDPA attention enabled (AttnProcessor2_0)")

        if _BLACKWELL_DETECTED:
            # Log active SDP backends and cuDNN version for observability
            logger.info(
                f"Blackwell SDP backends — "
                f"flash: {torch.backends.cuda.flash_sdp_enabled()}, "
                f"mem_efficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}, "
                f"math: {torch.backends.cuda.math_sdp_enabled()}"
            )
            logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")

            # SageAttention: ~5x attention speedup on Blackwell using FP4 microscaling.
            # Mutually exclusive with torch.compile; core.py reads _SAGE_ATTENTION_ACTIVE
            # and skips compilation when this is True.
            _try_install_sageattention(pipe)

    # Note: torch.compile is applied in core.py AFTER DoRA loading
    # Compiling before DoRA breaks adapter injection

    return pipe, cpu_offload_enabled


def _try_install_sageattention(pipe: StableDiffusionXLPipeline) -> None:
    """Attempt to install SageAttention processor on UNet attention layers.

    SageAttention uses FP4 microscaling on Blackwell tensor cores for ~5x
    attention speedup. Sets _SAGE_ATTENTION_ACTIVE=True on success.
    Silently falls back to SDPA (AttnProcessor2_0) on ImportError.
    """
    global _SAGE_ATTENTION_ACTIVE
    _SAGE_ATTENTION_ACTIVE = False
    try:
        from sageattention import sageattn
        from diffusers.models.attention_processor import Attention

        class SageAttnProcessor:
            """SageAttention processor matching diffusers AttnProcessor2_0 semantics."""
            def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                         attention_mask=None, temb=None, *args, **kwargs):
                residual = hidden_states

                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim
                if input_ndim == 4:
                    batch, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None
                    else encoder_hidden_states.shape
                )

                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = attn.to_q(hidden_states)

                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads

                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                if attn.norm_q is not None:
                    query = attn.norm_q(query)
                if attn.norm_k is not None:
                    key = attn.norm_k(key)

                hidden_states = sageattn(query, key, value, attn_mask=attention_mask, is_causal=False)
                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query.dtype)

                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / attn.rescale_output_factor
                return hidden_states

        pipe.unet.set_attn_processor(SageAttnProcessor())
        _SAGE_ATTENTION_ACTIVE = True
        logger.info("SageAttention active — ~5x attention speedup on Blackwell")

    except ImportError:
        logger.info(
            "SageAttention not installed — using SDPA (AttnProcessor2_0). "
            "For ~5x attention speedup on RTX 5090: pip install sageattention"
        )
        _SAGE_ATTENTION_ACTIVE = False
    except Exception as e:
        _SAGE_ATTENTION_ACTIVE = False
        logger.warning(f"SageAttention installation failed, falling back to SDPA: {e}")


def create_controlnet_pipeline(
    base_pipe: StableDiffusionXLPipeline,
    controlnet: ControlNetModel
) -> StableDiffusionXLControlNetPipeline:
    """Create a ControlNet pipeline from an existing base pipeline.

    This function wraps an existing SDXL pipeline with ControlNet support,
    preserving all components including VAE decode wrapping and scheduler config.

    Args:
        base_pipe: The base StableDiffusionXLPipeline
        controlnet: Loaded ControlNetModel

    Returns:
        StableDiffusionXLControlNetPipeline with ControlNet integrated
    """
    # Store the original/wrapped decode function
    original_vae_decode = base_pipe.vae.decode

    # Create ControlNet pipeline with all components from base
    controlnet_pipe = StableDiffusionXLControlNetPipeline(
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        text_encoder_2=base_pipe.text_encoder_2,
        tokenizer=base_pipe.tokenizer,
        tokenizer_2=base_pipe.tokenizer_2,
        unet=base_pipe.unet,
        controlnet=controlnet,
        scheduler=base_pipe.scheduler,
    )

    # Restore VAE decode wrapper if it was present
    # This ensures FP32 latent upcasting continues to work
    controlnet_pipe.vae.decode = original_vae_decode

    # VAE slicing state is already set on the shared VAE object by load_pipeline().
    # Do not re-enable unconditionally here — slicing was disabled on high-VRAM systems.

    logger.info("ControlNet pipeline created from base pipeline")

    return controlnet_pipe
