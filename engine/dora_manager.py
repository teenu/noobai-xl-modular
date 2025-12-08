"""DoRA adapter management."""

import os
import warnings
from typing import Optional
from config import logger, MODEL_CONFIG
from utils import validate_dora_path, find_dora_path
from engine.memory import clear_memory


class DoRAManager:
    """Manages DoRA adapter loading, unloading, and switching."""

    def __init__(self, pipe, device: str):
        self.pipe = pipe
        self.device = device
        self.dora_path: Optional[str] = None
        self.dora_loaded = False
        self.adapter_strength = MODEL_CONFIG.DEFAULT_ADAPTER_STRENGTH

    def load_adapter(self, dora_path: Optional[str] = None) -> bool:
        """Load DoRA adapter if available and valid."""
        try:
            if not dora_path:
                dora_path = find_dora_path()
                if not dora_path:
                    logger.warning("DoRA enabled but no valid DoRA file found")
                    return False

            is_valid, validated_path = validate_dora_path(dora_path)
            if not is_valid:
                logger.warning(f"DoRA validation failed: {validated_path}")
                return False

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="It seems like you are using a DoRA checkpoint")
                warnings.filterwarnings("ignore", message="No LoRA keys associated to CLIPTextModel")
                warnings.filterwarnings("ignore", message="No LoRA keys associated to CLIPTextModelWithProjection")
                self.pipe.load_lora_weights(
                    os.path.dirname(validated_path),
                    weight_name=os.path.basename(validated_path),
                    adapter_name="noobai_dora"
                )

            self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])
            self.dora_path = validated_path
            self.dora_loaded = True
            logger.info(f"DoRA adapter loaded: {os.path.basename(validated_path)}")
            return True

        except (IOError, OSError) as e:
            logger.error(f"Failed to load DoRA adapter (file error): {e}")
            self.dora_loaded = False
            return False
        except (RuntimeError, ValueError) as e:
            logger.error(f"Failed to load DoRA adapter (runtime/validation error): {e}")
            try:
                self.pipe.unload_lora_weights()
            except Exception as unload_error:
                logger.debug(f"Could not unload LoRA weights during error recovery: {unload_error}")
            clear_memory(self.device)
            self.dora_loaded = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading DoRA adapter: {e}")
            try:
                self.pipe.unload_lora_weights()
            except Exception as unload_error:
                logger.debug(f"Could not unload LoRA weights during error recovery: {unload_error}")
            clear_memory(self.device)
            self.dora_loaded = False
            return False
        finally:
            if not self.dora_loaded:
                self.dora_path = None

    def unload_adapter(self) -> None:
        """Completely unload DoRA adapter with full memory cleanup."""
        try:
            if self.dora_loaded and self.pipe is not None:
                try:
                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
                except Exception as e:
                    logger.warning(f"Could not set adapter weights to 0: {e}")

                try:
                    self.pipe.unload_lora_weights()
                except Exception as e:
                    logger.warning(f"Error unloading LoRA weights: {e}")

                try:
                    if hasattr(self.pipe, 'delete_adapters'):
                        self.pipe.delete_adapters(["noobai_dora"])
                except Exception as e:
                    logger.warning(f"Error deleting adapter references: {e}")

                clear_memory(self.device)
                logger.info("DoRA adapter unloaded")

            self.dora_loaded = False
            self.dora_path = None

        except Exception as e:
            logger.error(f"Error unloading DoRA adapter: {e}")
            self.dora_loaded = False
            self.dora_path = None

    def switch_adapter(self, new_adapter_path: str) -> bool:
        """Switch DoRA adapters with complete cleanup and fresh loading."""
        try:
            if not new_adapter_path:
                logger.error("Cannot switch to empty adapter path")
                return False

            is_valid, validated_path = validate_dora_path(new_adapter_path)
            if not is_valid:
                logger.error(f"Invalid new adapter path: {validated_path}")
                return False

            logger.info(f"Switching DoRA adapter to: {validated_path}")

            old_dora_path = self.dora_path
            if self.dora_loaded:
                self.unload_adapter()

            self.dora_path = validated_path
            success = self.load_adapter(validated_path)

            if success:
                logger.info(f"Successfully switched to DoRA adapter: {os.path.basename(validated_path)}")
                return True
            else:
                self.dora_path = old_dora_path
                logger.error("Failed to load new DoRA adapter")
                return False

        except Exception as e:
            logger.error(f"Error switching DoRA adapter: {e}")
            self.dora_loaded = False
            self.dora_path = None
            return False

    def set_strength(self, strength: float) -> float:
        """Set DoRA adapter strength with clamping."""
        original_strength = strength
        if not (MODEL_CONFIG.MIN_ADAPTER_STRENGTH <= strength <= MODEL_CONFIG.MAX_ADAPTER_STRENGTH):
            strength = max(MODEL_CONFIG.MIN_ADAPTER_STRENGTH, min(strength, MODEL_CONFIG.MAX_ADAPTER_STRENGTH))
            logger.warning(
                f"Adapter strength {original_strength} out of bounds "
                f"[{MODEL_CONFIG.MIN_ADAPTER_STRENGTH}-{MODEL_CONFIG.MAX_ADAPTER_STRENGTH}], "
                f"clamped to {strength}"
            )

        self.adapter_strength = strength

        if self.dora_loaded and self.pipe is not None:
            self.pipe.set_adapters(["noobai_dora"], adapter_weights=[strength])

        return strength

    def set_enabled(self, enabled: bool) -> None:
        """Dynamically enable/disable DoRA adapter."""
        try:
            if enabled:
                if not self.dora_loaded:
                    logger.warning("Cannot enable DoRA: adapter not loaded")
                    return

                if self.pipe is None:
                    logger.warning("Cannot enable DoRA: pipeline not initialized")
                    return

                if not (MODEL_CONFIG.MIN_ADAPTER_STRENGTH <= self.adapter_strength <= MODEL_CONFIG.MAX_ADAPTER_STRENGTH):
                    logger.warning(f"Invalid adapter strength {self.adapter_strength}, using default")
                    self.adapter_strength = MODEL_CONFIG.DEFAULT_ADAPTER_STRENGTH

                self.pipe.set_adapters(["noobai_dora"], adapter_weights=[self.adapter_strength])
                logger.info(f"DoRA adapter enabled (strength: {self.adapter_strength})")
            else:
                if self.dora_loaded and self.pipe is not None:
                    self.pipe.set_adapters(["noobai_dora"], adapter_weights=[0.0])
                logger.info("DoRA adapter disabled")
        except Exception as e:
            logger.warning(f"Error setting DoRA enabled state: {e}")
