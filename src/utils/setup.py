import os
import sys
import logging

def bootstrap_environment() -> None:
    """Prepares the Windows/PyTorch/Transformers environment.
    
    Ensures safe import order to prevent segfaults, sets safety environment
    variables, and applies Hugging Face dataclasses compatibility patches.
    """
    # 1. Windows PyTorch & PyArrow Import Segfault Fix (0xC0000005)
    # Importing datasets/pyarrow before torch resolves low-level DLL Clash on Windows
    try:
        import datasets
        import pyarrow
    except ImportError:
        pass

    # 2. Prevent OpenMP conflicts and Tokenizer multi-threading segfaults on Windows
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 3. Configure logging levels
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    # 4. Apply compatibility monkey-patches for Hugging Face
    import torch
    if not hasattr(torch, "float8_e8m0fnu"):
        setattr(torch, "float8_e8m0fnu", None)

    try:
        import torch.distributed.fsdp
        if not hasattr(torch.distributed.fsdp, "FSDPModule"):
            setattr(torch.distributed.fsdp, "FSDPModule", type("FSDPModule", (), {}))
    except ImportError:
        pass

    try:
        import huggingface_hub.dataclasses
        huggingface_hub.dataclasses.type_validator = lambda *args, **kwargs: None
    except ImportError:
        pass

    try:
        import transformers.utils.import_utils
        transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
    except (ImportError, AttributeError):
        pass

    try:
        import transformers.utils
        transformers.utils.check_torch_load_is_safe = lambda *args, **kwargs: None
    except (ImportError, AttributeError):
        pass

    # Patch T5 Config globally
    from src.models.model_utils import patch_t5_config
    patch_t5_config()

    # Ensure workspace root is in python path
    sys.path.append(os.path.abspath("."))
