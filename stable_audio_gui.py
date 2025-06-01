#!/usr/bin/env python3
"""
Stable‑Audio GUI — Gradio Edition v4.3.5 (indent & click‑fix)
--------------------------------------------------------------
* Fixes `IndentationError` in `_wrapped_generate` definition.
* Removes a stray duplicated `generate_btn.click()` call that broke the
  component graph.
Everything else unchanged (local weights, offline, Force CPU/MPS, length slider).
"""

from __future__ import annotations

import argparse, os, sys, time, warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import torch, torchaudio, numpy as np
import gradio as gr
from einops import rearrange

# ───────────────────── warnings & env ─────────────────────
warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API", category=UserWarning)
os.environ.setdefault("DIFFUSERS_DISABLE_TORCHSDE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

try:
    import huggingface_hub as hf
except ImportError:
    sys.exit("❌ huggingface_hub missing — pip install huggingface-hub")

try:
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond
except ImportError:
    sys.exit("❌ stable_audio_tools missing — pip install stable-audio-tools==0.0.19")

# ─────────────── paths & constants ───────────────
ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "generated_clips"; AUDIO_DIR.mkdir(exist_ok=True)
MODEL_ID = "stabilityai/stable-audio-open-1.0"
MODEL_DIR = ROOT / "stable-audio-open-1.0"
if not MODEL_DIR.exists():
    sys.exit(f"❌ Expected model folder not found: {MODEL_DIR}")

# ───────── monkey‑patch hf_hub_download ─────────
_original_hub_download = hf.hf_hub_download

def _local_hf_hub_download(repo_id: str, filename: str | None = None, *args: Any, **kwargs: Any) -> str:  # type: ignore
    if repo_id == MODEL_ID:
        if filename is None:
            raise ValueError("filename must be provided for local override")
        local = MODEL_DIR / filename
        if local.is_file():
            return str(local)
        local = MODEL_DIR / Path(filename)
        if local.is_file():
            return str(local)
        raise FileNotFoundError(local)
    return _original_hub_download(repo_id, filename=filename, *args, **kwargs)

hf.hf_hub_download = _local_hf_hub_download  # type: ignore

# ───────── device helpers ─────────

def device_str(force_cpu=False, force_mps=False):
    if force_cpu:
        return "cpu"
    if force_mps:
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"

_model_cache: dict[str, tuple[torch.nn.Module, dict]] = {}

def load_model(device: str):
    if device not in _model_cache:
        model, cfg = get_pretrained_model(MODEL_ID)
        _model_cache[device] = (model.to(device), cfg)
    return _model_cache[device]

# ───────── core generation ─────────

def generate_audio(prompt: str, length_sec: int, force_cpu: bool, force_mps: bool):
    if not prompt.strip():
        return None, "⚠️ Prompt is empty."

    device = device_str(force_cpu, force_mps)
    model, cfg = load_model(device)

    seconds = int(max(1, min(48, length_sec)))
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_end": seconds,
        "seconds_total": seconds,
    }]

    sample_rate = cfg["sample_rate"]
    sample_size = seconds * sample_rate

    start = time.time()
    with torch.no_grad():
        audio = generate_diffusion_cond(model, conditioning=conditioning, sample_size=sample_size, device=device)

        audio = rearrange(audio, "b d n -> d (b n)")  # (channels, samples)
    peak = audio.abs().max()
    if peak == 0:
        return None, "⚠️ Model returned silence. Try a different prompt."

    # int16 for file save
    audio_int16 = (audio / (peak + 1e-12)).mul(32767).clamp(-32767, 32767).to(torch.int16).cpu().numpy()
    out_path = AUDIO_DIR / f"clip_{datetime.now():%Y%m%d_%H%M%S}.wav"
    torchaudio.save(out_path.as_posix(), torch.from_numpy(audio_int16), sample_rate)

    # float32 [-1,1] + (samples, channels) for Gradio
    audio_float = (audio_int16.astype(np.float32) / 32767.0).T  # (samples, channels)

    return (sample_rate, audio_float), f"✅ {seconds}s clip generated in {time.time()-start:.1f}s — saved as {out_path.name}" f"✅ {seconds}s clip generated in {time.time()-start:.1f}s — saved as {out_path.name}"

# ───────── UI layer ─────────

def build_interface(force_cpu_cli: bool):
    with gr.Blocks(title="Stable‑Audio (offline)") as demo:
        gr.Markdown("## Stable‑Audio Open 1.0 — Local Text‑to‑Music Generator")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", lines=3, placeholder="epic orchestral trailer score …")
        with gr.Row():
            length_slider = gr.Slider(1, 48, 12, 1, label="Length (seconds)")
            cpu_checkbox = gr.Checkbox(force_cpu_cli, label="Force CPU")
            mps_checkbox = gr.Checkbox(False, label="Force MPS")
        generate_btn = gr.Button("Generate 🎵")
        status_md = gr.Markdown("Loading model … (first call only)")
        audio_out = gr.Audio(label="Preview", streaming=False)

        def _wrapped_generate(p, l, cpu_flag, mps_flag):
            return generate_audio(p, int(l), cpu_flag, mps_flag)

        generate_btn.click(
            _wrapped_generate,
            inputs=[prompt, length_slider, cpu_checkbox, mps_checkbox],
            outputs=[audio_out, status_md],
        )

    return demo

# ───────── entry‑point ─────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stable‑Audio Gradio GUI (offline)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--mps", action="store_true")
    args = ap.parse_args()

    iface = build_interface(force_cpu_cli=args.cpu)
    iface.queue(default_concurrency_limit=1)
    iface.launch(share=args.share)
