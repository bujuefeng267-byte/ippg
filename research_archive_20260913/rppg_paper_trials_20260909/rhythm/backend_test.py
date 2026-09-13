"""Bounded, reference-HR-free verification of the frozen RhythmMamba backend."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

import numpy as np
import torch
import backend

HERE = Path(__file__).resolve().parent
DETAILS = {}


def manual_real_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    """Independent float64 scalar recurrence for the variable-B/C real-state case."""
    arrays = [v.detach().cpu().numpy().astype(np.float64) if v is not None else None
              for v in (u, delta, A, B, C, D, z, delta_bias)]
    uu, dd, aa, bb, cc, skip, gate, bias = arrays
    batch, channels, length = uu.shape
    state = np.zeros((batch, channels, aa.shape[1]), np.float64)
    out = np.empty_like(uu)
    for t in range(length):
        for b in range(batch):
            for c in range(channels):
                dt = dd[b, c, t] + (bias[c] if bias is not None else 0.0)
                if delta_softplus:
                    dt = np.logaddexp(0.0, dt)
                for n in range(aa.shape[1]):
                    state[b, c, n] = (math.exp(dt * aa[c, n]) * state[b, c, n]
                                      + dt * bb[b, n, t] * uu[b, c, t])
                y = float(sum(state[b, c, n] * cc[b, n, t] for n in range(aa.shape[1])))
                if skip is not None:
                    y += skip[c] * uu[b, c, t]
                if gate is not None:
                    q = gate[b, c, t]
                    y *= q / (1.0 + math.exp(-q))
                out[b, c, t] = y
    return torch.as_tensor(out, dtype=u.dtype), torch.as_tensor(state, dtype=A.dtype)


def fixture(device="cpu"):
    gen = torch.Generator(device="cpu").manual_seed(324)
    rand = lambda *shape: torch.randn(*shape, generator=gen)
    values = dict(u=rand(2, 7, 11), delta=rand(2, 7, 11),
                  A=-torch.exp(rand(7, 3)), B=rand(2, 3, 11), C=rand(2, 3, 11),
                  D=rand(7), z=rand(2, 7, 11), delta_bias=rand(7))
    return {k: v.to(device) for k, v in values.items()}


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Mamba, scan, cls.source_metadata = backend.get_reference_components()
        cls.scan = staticmethod(scan)

    def test_01_source_hashes_and_manual_recurrence_cpu(self):
        inputs = fixture()
        expected, last = manual_real_scan(**inputs, delta_softplus=True)
        with torch.inference_mode():
            actual, actual_last = self.scan(**inputs, delta_softplus=True, return_last_state=True)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(actual_last, last, rtol=2e-5, atol=2e-5)
        DETAILS["scan_cpu"] = dict(shape=list(actual.shape),
            max_abs_vs_float64_scalar=float((actual-expected).abs().max()),
            last_state_max_abs=float((actual_last-last).abs().max()))

    def test_02_scan_zero_and_optional_gates(self):
        inputs = fixture()
        inputs["delta"] = inputs["delta"].abs()
        for name in ("D", "z", "delta_bias"):
            inputs[name] = None
        expected, _ = manual_real_scan(**inputs)
        with torch.inference_mode():
            actual = self.scan(**inputs)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        inputs["u"].zero_()
        with torch.inference_mode():
            zero = self.scan(**inputs)
        self.assertEqual(torch.count_nonzero(zero).item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; GPU parity remains unverified")
    def test_03_scan_gpu_matches_manual_and_cpu(self):
        cpu_inputs = fixture()
        expected, last = manual_real_scan(**cpu_inputs, delta_softplus=True)
        gpu_inputs = {k: v.cuda() for k, v in cpu_inputs.items()}
        with torch.inference_mode():
            cpu, _ = self.scan(**cpu_inputs, delta_softplus=True, return_last_state=True)
            gpu, gpu_last = self.scan(**gpu_inputs, delta_softplus=True, return_last_state=True)
        torch.testing.assert_close(gpu.cpu(), expected, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(gpu_last.cpu(), last, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(gpu.cpu(), cpu, rtol=2e-5, atol=2e-5)
        DETAILS["scan_gpu"] = dict(max_abs_vs_float64_scalar=float((gpu.cpu()-expected).abs().max()),
                                   max_abs_vs_cpu=float((gpu.cpu()-cpu).abs().max()))

    def test_04_original_mamba_forward_matches_its_step_cpu(self):
        torch.manual_seed(108)
        layer = self.Mamba(d_model=96, d_state=48, d_conv=4, expand=2, use_fast_path=False).eval()
        x = torch.randn(2, 13, 96)
        with torch.inference_mode():
            whole = layer(x)
            conv_state, ssm_state = layer.allocate_inference_cache(2, 13)
            steps = []
            for i in range(x.shape[1]):
                out, _, _ = layer.step(x[:, i:i+1], conv_state, ssm_state)
                steps.append(out)
            stepped = torch.cat(steps, dim=1)
        torch.testing.assert_close(whole, stepped, rtol=2e-5, atol=2e-5)
        DETAILS["mamba_forward_vs_step_cpu"] = dict(shape=list(whole.shape),
            max_abs=float((whole-stepped).abs().max()))

    def test_05_checkpoint_hash_guard(self):
        changed = dict(backend.EXPECTED_HASHES)
        changed[backend.CHECKPOINT_PATH] = "0" * 64
        with mock.patch.object(backend, "EXPECTED_HASHES", changed):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                backend.verify_sources()

    def test_06_strict_fixed_model_160_frames(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        missing = object()
        prior = {name: sys.modules.get(name, missing) for name in backend._SHIM_NAMES}
        model, metadata = backend.load_model(device)
        for name in backend._SHIM_NAMES:
            self.assertIs(sys.modules.get(name, missing), prior[name])
        self.assertTrue(metadata["strict"] and metadata["weights_only"])
        self.assertEqual(metadata["mamba_layers"], 24)
        self.assertFalse(model.training)
        self.assertTrue(all(not m.use_fast_path for m in model.modules() if isinstance(m, self.Mamba)))
        self.assertEqual(metadata["checkpoint_sha256"], backend.EXPECTED_HASHES[backend.CHECKPOINT_PATH])
        gen = torch.Generator(device="cpu").manual_seed(734)
        x = torch.randn((1, 160, 3, 128, 128), generator=gen).to(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        self.assertEqual(tuple(output.shape), (1, 160))
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertGreater(output.std().item(), 0)
        self.assertEqual(output.dtype, torch.float32)
        DETAILS["full_model"] = dict(metadata=metadata, input_shape=list(x.shape),
            output_shape=list(output.shape), output_finite=True, raw_output_mean=output.mean().item(),
            raw_output_sample_sd=output.std().item(), execution_seconds=elapsed,
            performance_interpretation="Single synthetic smoke test of the reference backend; not an official benchmark.",
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None)
        del model, x, output
        if device == "cuda":
            torch.cuda.empty_cache()


def main():
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(
        passed=result.wasSuccessful(),
        status="passed" if result.wasSuccessful() else "failed",
        test_count=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=[{"test":str(test),"reason":reason} for test,reason in result.skipped],
        gpu_available=torch.cuda.is_available(), torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda, numpy_version=np.__version__,
        elapsed_seconds=time.perf_counter()-started,
        adapter_sha256=hashlib.sha256((HERE/"backend.py").read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        details=DETAILS,
        failures_detail=[{"test":str(test),"traceback":trace} for test,trace in result.failures+result.errors],
        official_cuda_kernel_parity_verified=False,
        inference_accuracy_claim=False,
        limitation="Manual recurrence and CPU/GPU reference parity are tested. Official fused CUDA kernels are not tested.")
    (HERE/"backend_verification.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
