"""CPU tests for the E4M3 decode reference (RFC 0001, E4M3).

The CUDA kernel in ``areno/accel/csrc/e4m3_linear.cu`` decodes each uint8 E4M3
byte to bf16 in-kernel. These tests validate that reference formula against
``torch.float8_e4m3fn``'s authoritative conversion, across the representable
range and the boundary cases (subnormal, max finite 448, zero, signs). They run
on CPU without a GPU; the kernel's GPU correctness is covered separately.

Decode formula:
    sign = (b>>7)&1; e = (b>>3)&0xF; m = b&0x7
    e==0 (subnormal):  (-1)^s * m * 2^-9
    1<=e<=15 (normal): (-1)^s * 2^(e-7) * (1 + m/8); max finite 448 (e=15,m=6)
    e==15 && m==7 is NaN in torch (a valid quantized weight is clamped at 448).
"""

from __future__ import annotations

import math

import torch

_E4M3_MAX = 448.0


def decode_e4m3(b: int) -> float:
    sign = (b >> 7) & 1
    e = (b >> 3) & 0xF
    m = b & 0x7
    if e == 0:
        val = float(m) * 0.001953125  # m * 2^-9
    elif e == 15 and m == 7:
        val = _E4M3_MAX  # NaN pattern; clamp to max finite
    else:
        val = 2.0 ** (e - 7) * (1.0 + float(m) / 8.0)
    return -val if sign else val


def _torch_decode(byte: int) -> float:
    t = torch.tensor([byte], dtype=torch.uint8).view(torch.float8_e4m3fn)
    return float(t.to(torch.bfloat16).float())


def _is_finite(v: float) -> bool:
    return not (math.isnan(v) or math.isinf(v))


def test_all_representable_e4m3_bytes_match_torch():
    # Every byte whose torch decode is finite must match the reference formula.
    mismatches = 0
    for byte in range(256):
        ref = decode_e4m3(byte)
        refv = float(ref)
        torch_v = _torch_decode(byte)
        if _is_finite(torch_v):
            if abs(refv - torch_v) > 0.0:
                mismatches += 1
    assert mismatches == 0, f"{mismatches} finite bytes diverge from torch"


def test_e4m3_max_finite_and_zero():
    assert decode_e4m3(0x7E) == _E4M3_MAX  # 448
    assert decode_e4m3(0x00) == 0.0
    assert decode_e4m3(0x80) == -0.0
    # symmetric negative max
    assert decode_e4m3(0xFE) == -_E4M3_MAX


def test_e4m3_subnormal_convention():
    # subnormal: e==0, value = m * 2^-9
    assert decode_e4m3(0x01) == 1.0 / 512.0  # 2^-9
    assert decode_e4m3(0x02) == 2.0 / 512.0
    assert decode_e4m3(0x04) == 4.0 / 512.0


def test_e4m3_powers_of_two():
    assert decode_e4m3(0x38) == 1.0  # e=7,m=0
    assert decode_e4m3(0x40) == 2.0  # e=8,m=0
    assert decode_e4m3(0x30) == 0.5  # e=6,m=0


def test_quantize_dequant_matches_reference_matmul():
    from areno.accel.kernels.e4m3_cuda import dequant_e4m3_bf16, quantize_weight_e4m3

    torch.manual_seed(0)
    w = torch.randn(64, 96)
    w = (w / w.abs().amax()) * 10.0
    w_bf16 = w.to(torch.bfloat16)
    w_u8, scale = quantize_weight_e4m3(w_bf16)
    dq = dequant_e4m3_bf16(w_u8, scale)
    x = torch.randn(3, 96, dtype=torch.bfloat16)
    ref = x @ w_bf16.T
    out = x @ dq.T
    rel = float((out.float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-6))
    assert rel < 0.2, f"e4m3 dequant matmul rel err too high: {rel}"
