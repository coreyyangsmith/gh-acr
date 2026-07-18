from __future__ import annotations

from typing import Any, Optional
import os
import logging

logger = logging.getLogger(__name__)

try:  # pragma: no cover
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_HF_CACHE_DIR = os.getenv("HF_CACHE_DIR") or os.path.join(REPO_ROOT, "data", "models")
DEFAULT_LOCAL_MODEL_ID = os.getenv("HF_MODEL_ID") or os.getenv("MODEL_ID") or "gpt2"
HF_LOCAL_ONLY = os.getenv("HF_LOCAL_ONLY", "0").strip().lower() in ("1", "true", "yes", "on")
HF_TRUST_REMOTE_CODE = os.getenv("HF_TRUST_REMOTE_CODE", "0").strip().lower() in ("1", "true", "yes", "on")
HF_REVISION = os.getenv("HF_REVISION", "").strip() or None
LOCAL_SEED = int(os.getenv("SEED", "42"))


def get_hf_token() -> Optional[str]:
    for var in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HF_API_TOKEN", "HUGGINGFACE_TOKEN"):
        tok = os.getenv(var)
        if tok and tok.strip():
            return tok.strip()
    return None


def should_use_accelerate() -> bool:
    v = os.getenv("HF_DEVICE_MAP", "").strip().lower()
    if v in {"auto", "cpu"} or v.isdigit() or v.startswith("cuda"):
        return True
    return False


def weights_exist_locally(model_id: str) -> bool:
    try:
        root = os.path.join(DEFAULT_HF_CACHE_DIR, model_id.replace("/", os.sep))
        for base, _, files in os.walk(root):
            if any(f.endswith(".safetensors") for f in files):
                return True
    except Exception:
        pass
    return False


def parse_torch_dtype_env() -> Optional[Any]:  # type: ignore[override]
    val = os.getenv("HF_TORCH_DTYPE", "").strip().lower()
    if not val:
        return None
    if val == "auto":
        return "auto"
    try:
        if torch is not None:
            if val in ("fp16", "float16", "half"):
                return torch.float16
            if val in ("bf16", "bfloat16"):
                return torch.bfloat16
            if val in ("fp32", "float32"):
                return torch.float32
    except Exception:
        pass
    return None


def pick_torch_dtype() -> Any:
    dt = parse_torch_dtype_env()
    if dt == "auto":
        dt = None
    if dt is not None:
        return dt
    try:
        if torch is not None and torch.cuda.is_available():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float32


def get_device_map_from_env() -> Any:  # type: ignore[override]
    v = os.getenv("HF_DEVICE_MAP", "").strip().lower()
    if not v:
        return "auto"
    if v == "auto":
        return "auto"
    if v == "cpu":
        return {"": "cpu"}
    if v.isdigit():
        return {"": int(v)}
    if v.startswith("cuda"):
        idx = int(v.split(":", 1)[1]) if ":" in v else 0
        return {"": idx}
    return "auto"


def collect_model_devices(model: Any) -> list[str]:  # type: ignore[override]
    devices: set[str] = set()
    try:
        dm = getattr(model, "hf_device_map", None)
        if isinstance(dm, dict):
            for v in dm.values():
                if isinstance(v, int):
                    devices.add(f"cuda:{v}")
                else:
                    devices.add(str(v))
    except Exception:
        pass
    try:
        dev = getattr(model, "device", None)
        if dev is not None:
            devices.add(str(dev))
    except Exception:
        pass
    if not devices:
        try:
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                devices.add("cuda:0")
            else:
                devices.add("cpu")
        except Exception:
            devices.add("cpu")
    return sorted(devices)


def hf_device_index() -> int:
    try:
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            return 0
    except Exception:  # pragma: no cover
        pass
    return -1


def format_bytes(num: int) -> str:
    try:
        size = float(num)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
            if size < 1024.0:
                return f"{size:.2f}{unit}"
            size /= 1024.0
        return f"{size:.2f}EiB"
    except Exception:
        return str(num)


def log_gpu_overview(context: str, model: Any | None = None) -> None:
    try:
        cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "") or "<unset>"
        alloc_conf = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "") or "<unset>"
        hf_dev_map_env = os.getenv("HF_DEVICE_MAP", "") or "<unset>"

        if torch is None:
            logger.info("%s CUDA overview: torch not available. CUDA_VISIBLE_DEVICES=%s", context, cuda_visible)
            return
        if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
            logger.info("%s CUDA overview: CUDA not available. CUDA_VISIBLE_DEVICES=%s", context, cuda_visible)
            return

        num = int(torch.cuda.device_count())
        cuda_ver = getattr(torch.version, "cuda", None)
        header = (
            f"{context} CUDA overview: torch={getattr(torch, '__version__', '?')}, "
            f"cuda={cuda_ver}, devices={num}, CUDA_VISIBLE_DEVICES={cuda_visible}, "
            f"HF_DEVICE_MAP={hf_dev_map_env}, PYTORCH_CUDA_ALLOC_CONF={alloc_conf}"
        )
        lines: list[str] = [header]

        try:
            dm = getattr(model, "hf_device_map", None)
            if isinstance(dm, dict) and dm:
                sample = {k: str(v) for k, v in list(dm.items())[:8]}
                extra = "" if len(dm) <= 8 else f" ... +{len(dm) - 8} more"
                lines.append(f"{context} hf_device_map: {sample}{extra}")
        except Exception:
            pass

        for idx in range(num):
            name = f"cuda:{idx}"
            total_prop = 0
            try:
                props = torch.cuda.get_device_properties(idx)
                name = props.name
                total_prop = int(getattr(props, "total_memory", 0))
            except Exception:
                pass
            free_b = None
            total_b = None
            try:
                free_b, total_b = torch.cuda.mem_get_info(idx)
            except Exception:
                pass
            reserved_b = None
            allocated_b = None
            try:
                reserved_b = torch.cuda.memory_reserved(idx)
                allocated_b = torch.cuda.memory_allocated(idx)
            except Exception:
                pass

            parts = [f"cuda:{idx} {name}"]
            parts.append(f"total={format_bytes((total_b or total_prop) or 0)}")
            if free_b is not None:
                parts.append(f"free={format_bytes(int(free_b))}")
            if reserved_b is not None:
                parts.append(f"reserved={format_bytes(int(reserved_b))}")
            if allocated_b is not None:
                parts.append(f"allocated={format_bytes(int(allocated_b))}")
            lines.append(f"{context} " + ", ".join(parts))

        logger.info("\n".join(lines))
    except Exception as e:  # pragma: no cover
        try:
            logger.info("%s CUDA overview logging failed: %s", context, e)
        except Exception:
            pass


__all__ = [
    "REPO_ROOT",
    "DEFAULT_HF_CACHE_DIR",
    "DEFAULT_LOCAL_MODEL_ID",
    "HF_LOCAL_ONLY",
    "HF_TRUST_REMOTE_CODE",
    "HF_REVISION",
    "LOCAL_SEED",
    "get_hf_token",
    "should_use_accelerate",
    "weights_exist_locally",
    "parse_torch_dtype_env",
    "pick_torch_dtype",
    "get_device_map_from_env",
    "collect_model_devices",
    "hf_device_index",
    "format_bytes",
    "log_gpu_overview",
]


