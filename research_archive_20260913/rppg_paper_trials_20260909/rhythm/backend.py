"""Frozen RhythmMamba weights with the authors' pure-Torch reference scan.

This adapter changes execution, not the Mamba class or the checkpoint. It does
not reproduce the official CUDA-kernel speed benchmark. No reference HR enters
this module. Input/output preprocessing belongs to the caller.
"""
from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib.metadata
import math
from pathlib import Path
import re
import sys
import threading
import types
from typing import Optional

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "official_snapshot"
AUDIT_PATH = HERE.parent / "rhythm_source_audit.json"
COMMIT = "1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed"
MAMBA_PATH = "setup/mamba/mamba_ssm/modules/mamba_simple.py"
SCAN_PATH = "setup/mamba/mamba_ssm/ops/selective_scan_interface.py"
MODEL_PATH = "neural_methods/model/RhythmMamba.py"
CONFIG_PATH = "configs/infer_configs/PURE_UBFC-rPPG_RHYTHMMAMBA.yaml"
CHECKPOINT_PATH = "PreTrainedModels/PURE_cross_RhythmMamba.pth"
EXPECTED_HASHES = {
    MAMBA_PATH: "dc035a4dcfbc9f9809e2bb60764f7bfe38259b6397968399fc33cad159300a43",
    SCAN_PATH: "fe606b4c7e81b47bb091cf59dc474aece1112a6ca01eb6f22309090e56030bc4",
    MODEL_PATH: "653476d8fb240f4673d73d8f2b86efdc1bcaef573ae883328366ced7544359c6",
    CONFIG_PATH: "675200ff3a04b776a3e41517363a56b9caf8cbc8fa8c1f7aa42bdd3dee1751a8",
    CHECKPOINT_PATH: "442ec9ac71bfc299f41e2c7f671216cfa4f6749d8dd264e53f8eb9d8b65380ee",
}
CHECKPOINT_BYTES = 20044634
CHECKPOINT_GIT_BLOB = "836e171ef226dc5ca72240f000592d5dbd84b293"
_IMPORT_LOCK = threading.RLock()
_SHIM_NAMES = ("mamba_ssm", "mamba_ssm.modules", "mamba_ssm.modules.mamba_simple")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_sources():
    """Validate immutable text and weights, without importing torch or unpickling."""
    import json

    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if audit["repository"]["commit"] != COMMIT:
        raise ValueError("Unexpected audited RhythmMamba commit")
    if audit["selection"]["checkpoint"] != CHECKPOINT_PATH:
        raise ValueError("The fixed PURE checkpoint selection changed")
    texts, records = {}, {}
    for path in (MAMBA_PATH, SCAN_PATH):
        body = audit["source_content"][path]
        encoded = body.encode("utf-8")
        if _sha(encoded) != EXPECTED_HASHES[path]:
            raise ValueError(f"Embedded official source checksum mismatch: {path}")
        snapshot_path = SNAPSHOT / path
        if snapshot_path.exists() and snapshot_path.read_bytes() != encoded:
            raise ValueError(f"Snapshot differs from audited official source: {path}")
        texts[path] = body
        records[path] = dict(sha256=_sha(encoded), bytes=len(encoded),
                             storage="verbatim audited JSON",
                             optional_snapshot_matches=snapshot_path.exists())
    for path in (MODEL_PATH, CONFIG_PATH, CHECKPOINT_PATH):
        encoded = (SNAPSHOT / path).read_bytes()
        if _sha(encoded) != EXPECTED_HASHES[path]:
            raise ValueError(f"Frozen snapshot checksum mismatch: {path}")
        records[path] = dict(sha256=_sha(encoded), bytes=len(encoded), storage="official_snapshot")
        if path != CHECKPOINT_PATH:
            texts[path] = encoded.decode("utf-8")
        else:
            blob = hashlib.sha1(b"blob " + str(len(encoded)).encode("ascii") + b"\0" + encoded).hexdigest()
            if len(encoded) != CHECKPOINT_BYTES or blob != CHECKPOINT_GIT_BLOB:
                raise ValueError("Checkpoint differs from the fixed official Git blob")
            records[path]["git_blob_sha1"] = blob
    return texts, records


def _compile_original_object(text, name, namespace, filename):
    """Execute exactly one original AST node; imports/CUDA wrappers are excluded."""
    tree = ast.parse(text, filename=filename)
    selected = [node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name]
    if len(selected) != 1:
        raise ValueError(f"Expected one official {name} definition")
    node = selected[0]
    code = compile(ast.Module(body=[node], type_ignores=[]), filename, "exec")
    exec(code, namespace)
    fingerprint = _sha(ast.dump(node, include_attributes=False).encode("utf-8"))
    return namespace[name], dict(name=name, line_start=node.lineno,
                                 line_end=node.end_lineno, ast_sha256=fingerprint)


@lru_cache(maxsize=1)
def _reference_components_cached(mamba_text, scan_text):
    import torch
    from einops import rearrange, repeat

    common = dict(torch=torch, F=torch.nn.functional, rearrange=rearrange, repeat=repeat)
    scan_namespace = dict(common, __name__="rhythm_official_reference_scan")
    scan, scan_ast = _compile_original_object(
        scan_text, "selective_scan_ref", scan_namespace, str(SNAPSHOT / SCAN_PATH))
    mamba_namespace = dict(
        common, __name__="rhythm_official_reference_mamba",
        math=math, nn=torch.nn, Tensor=torch.Tensor, Optional=Optional,
        selective_scan_fn=scan, mamba_inner_fn=None,
        causal_conv1d_fn=None, causal_conv1d_update=None,
        selective_state_update=None, RMSNorm=None, layer_norm_fn=None, rms_norm_fn=None)
    mamba, mamba_ast = _compile_original_object(
        mamba_text, "Mamba", mamba_namespace, str(SNAPSHOT / MAMBA_PATH))
    return mamba, scan, dict(mamba_class=mamba_ast, selective_scan_ref=scan_ast)


def get_reference_components():
    """Return the unmodified official Mamba class, reference function and metadata."""
    texts, records = verify_sources()
    mamba, scan, ast_record = _reference_components_cached(texts[MAMBA_PATH], texts[SCAN_PATH])
    return mamba, scan, dict(ast=ast_record, source_hashes={k: v["sha256"] for k, v in records.items()})


@contextmanager
def _temporary_mamba_imports(mamba_class):
    """Let the original model import Mamba without importing CUDA extensions.

    The shim is scoped to the import and prior sys.modules entries are restored.
    The resulting model keeps its class references after restoration.
    """
    with _IMPORT_LOCK:
        missing = object()
        previous = {name: sys.modules.get(name, missing) for name in _SHIM_NAMES}
        package = types.ModuleType("mamba_ssm")
        package.__path__ = []
        modules = types.ModuleType("mamba_ssm.modules")
        modules.__path__ = []
        leaf = types.ModuleType("mamba_ssm.modules.mamba_simple")
        leaf.__file__ = str(SNAPSHOT / MAMBA_PATH)
        leaf.Mamba = mamba_class
        package.modules = modules
        modules.mamba_simple = leaf
        try:
            sys.modules.update(dict(zip(_SHIM_NAMES, (package, modules, leaf))))
            yield
        finally:
            for name, old in previous.items():
                if old is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_model(device="cuda"):
    """Load the one fixed PURE model; return (eval float32 model, JSON metadata).

    No automatic device fallback, alternate checkpoint, unsafe deserializer or
    non-strict state-dict loading is permitted.
    """
    import torch

    version = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if version is None or tuple(map(int, version.groups())) < (2, 6):
        raise RuntimeError("Use PyTorch >=2.6 for the weights-only loader; the author's 2.1.2 is historical")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Only explicit cpu or cuda devices are supported")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
    texts, records = verify_sources()
    mamba_class, _, ast_record = _reference_components_cached(texts[MAMBA_PATH], texts[SCAN_PATH])
    original = types.ModuleType("rhythm_frozen_official_model")
    original.__file__ = str(SNAPSHOT / MODEL_PATH)
    with _temporary_mamba_imports(mamba_class):
        exec(compile(texts[MODEL_PATH], original.__file__, "exec"), original.__dict__)
    model = original.RhythmMamba().float()
    state = torch.load(SNAPSHOT / CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Expected a nonempty official tensor state dict")
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Checkpoint must contain string keys and tensor values only")
    prefix_flags = [key.startswith("module.") for key in state]
    if any(prefix_flags) and not all(prefix_flags):
        raise ValueError("Mixed DataParallel prefixes in checkpoint")
    prefix_removed = all(prefix_flags)
    clean = {key[7:] if prefix_removed else key: tensor for key, tensor in state.items()}
    expected = model.state_dict()
    if set(clean) != set(expected):
        raise ValueError(f"Checkpoint key mismatch: missing={sorted(set(expected)-set(clean))}, "
                         f"unexpected={sorted(set(clean)-set(expected))}")
    for key, tensor in clean.items():
        if tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype:
            raise ValueError(f"Checkpoint shape/dtype mismatch: {key}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Checkpoint has nonfinite values: {key}")
    incompatible = model.load_state_dict(clean, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict loading unexpectedly left incompatible keys")
    mamba_modules = [module for module in model.modules() if isinstance(module, mamba_class)]
    if len(mamba_modules) != 24:
        raise ValueError("Frozen model must contain exactly 24 Mamba layers")
    for module in mamba_modules:
        module.use_fast_path = False
    model = model.to(target)
    model.eval()
    metadata = dict(
        backend="official_selective_scan_ref_torch",
        adapter_sha256=_sha(Path(__file__).read_bytes()),
        official_repository="https://github.com/zizheng-guo/RhythmMamba",
        official_commit=COMMIT, source_files=records, ast=ast_record,
        official_Mamba_class_unchanged=True, official_model_text_unchanged=True,
        checkpoint=CHECKPOINT_PATH, checkpoint_sha256=records[CHECKPOINT_PATH]["sha256"],
        weights_only=True, strict=True, removed_uniform_module_prefix=prefix_removed,
        state_dict_tensors=len(clean), state_dict_dtype_counts=dict(Counter(str(t.dtype) for t in clean.values())),
        parameters=sum(p.numel() for p in model.parameters()),
        mamba_layers=len(mamba_modules), use_fast_path=False,
        causal_conv_extension=False, selective_scan_cuda_extension=False,
        model_eval=True, model_dtype="float32", input_shape="[B,160,3,128,128]",
        output_shape="[B,160]", device=str(target),
        gpu_name=torch.cuda.get_device_name(target) if target.type == "cuda" else None,
        torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
        timm_version=_package_version("timm"), einops_version=_package_version("einops"),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        official_cuda_kernel_parity_verified=False,
        official_performance_benchmark=False,
        preprocessing="Caller must apply the frozen full-video standardization and 160-frame clip protocol.",
        limitations=["Pure Torch reference execution adapter, not the author's fused CUDA backend.",
                     "No inference speed claim for the original implementation.",
                     "This loader alone does not verify CUDA-kernel numerical parity or physiological accuracy."])
    return model, metadata

