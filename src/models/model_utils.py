"""Utility helper functions for modifying Hugging Face model configurations."""
from typing import Any
import transformers


def patch_t5_config() -> None:
    """Patches T5Config to ensure initializer_factor is a float.

    This resolves validation warnings and errors during configuration loading.
    """
    original_init = transformers.models.t5.configuration_t5.T5Config.__init__

    def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
        if "initializer_factor" in kwargs:
            kwargs["initializer_factor"] = float(kwargs["initializer_factor"])
        original_init(self, *args, **kwargs)
        if hasattr(self, "initializer_factor"):
            self.initializer_factor = float(self.initializer_factor)

    transformers.models.t5.configuration_t5.T5Config.__init__ = new_init
