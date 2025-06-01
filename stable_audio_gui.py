#!/usr/bin/env python3
"""
Stable‑Audio GUI — Gradio Edition v4.3.1 (local‑weights, monkey‑patch)
----------------------------------------------------------------------
Fixes the *HFValidationError* by **monkey‑patching `huggingface_hub.hf_hub_download`** so
any file request for the repo ID ``stabilityai/stable-audio-open-1.0`` is
served directly from your on‑disk folder `./stable-audio-open-1.0/`.

No symlinks or cache tricks needed; the override happens in‑process before
`stable_audio_tools` calls the Hub. Works fully offline.

Only the helper function `local_hf_hub_download()` and the monkey‑patch lines
are new. Everything else (Force CPU/MPS, length slider) remains unchanged.
"""

from __future__ import annotations

import argparse, os, sys, time, warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import torch, torchaudio
import gradio as gr
from einops import rearrange

# ─────────────────────── warnings & environment tweaks ───────────────────────
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API",
    category=UserWarning,
)
os.environ.setdefault("DIFFUSERS_DISABLE_TORCHSDE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # keep hub fully offline

try:
    import huggingface_hub as hf
except ImportError as exc:
    sys.exit("❌ huggingface_hub missing — install with: pip install huggingface-hub")

try:
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond
except ImportError:
    sys.exit("❌ stable_audio_tools missing — install with: pip install stable-audio-tools==0.0.19")

# ─────────────────────────── paths & constants ───────────────────────────────
ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "generated_clips"; AUDIO_DIR.mkdir(exist_ok=True)
MODEL_ID = "stabilityai/stable-audio-open-1.0"  # repo ID expected by SATools
MODEL_DIR = ROOT / "stable-audio-open-1.0"       # your local copy
if not MODEL_DIR.exists():
    sys.exit(f"❌ Expected model folder not found: {MODEL_DIR}")

# ───────────────────── monkey‑patch hf_hub_download ──────────────────────────
_original_hub_download = hf.hf_hub_download  # keep ref to the real one


def _local_hf_hub_download(repo_id: str, filename: str | None = None, *args: Any, **kwargs: Any) -> str:  # type: ignore
    """Intercept download calls for the Stable‑Audio repo and serve from disk."""
    if repo_id == MODEL_ID:
        if filename is None:
            raise ValueError("filename must be provided when repo_id is local override")
        local_path = MODEL_DIR / filename
        if local_path.is_file():
            return str(local_path)
        # SATools sometimes requests files with subdir prefixes already in filename
        # (e.g. "projection_model/diffusion_pytorch_model.safetensors"). For safety,
        # normalise against the local root.
        local_path = MODEL_DIR / Path(filename)
        if local_path.is_file():
            return str(local_path)
        raise FileNotFoundError(f"Requested file not found in local model dir: {local_path}")
    # Fallback to real hub download for any other repo
    return _original_hub_download(repo_id, filename=filename, *args, **kwargs)

# Apply the patch before SATools imports call it
hf.hf_hub_download = _local_hf_hub_download  # type: ignore

# ───────────────────────── device selection helpers ─────────────────────────

def device_str(force_cpu: bool = False, force_mps: bool = False) -> str:
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
        model, cfg = get_pretrained_model(MODEL_ID)  # goes through patched download
        _model_cache[device] = (model.to(device), cfg)
    return _model_cache[device]

# ─────────────────────────── core generation ────────────────────────────────

def generate_audio(prompt: str, length_sec: int, force_cpu: bool, force_mps: bool):
    if not prompt.strip():
        return None, "⚠️ Prompt is empty."

    device = device_str(force_cpu, force_mps)
    model, cfg = load_model(device)

    # ------------------------------------------------------------------
    # Stable‑Audio models expect extra metadata keys in each conditioning
    # dict. If they are missing you get: "ValueError: Conditioner key
    # seconds_start not found in batch metadata". We supply sane defaults.
    # ------------------------------------------------------------------
    seconds = int(max(1, min(48, length_sec)))
        conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_end": seconds,
        "seconds_total": seconds  # some SATools versions look for this key
    }]

    sample_rate = cfg["sample_rate"]  # 44 100 Hz
    sample_size = seconds * sample_rate

    start = time.time()
    with torch.no_grad():
        audio = generate_diffusion_cond(
            model,
            conditioning=conditioning,
            sample_size=sample_size,
            device=device,
        )

    audio = rearrange(audio, "b d n -> d (b n)")
    peak = audio.abs().max()
    if peak == 0:
        return None, "⚠️ Model returned silence. Try a different prompt."

    audio = (audio / (peak + 1e-12)).mul(32767).clamp(-32767, 32767).to(torch.int16).cpu().numpy()

    out_path = AUDIO_DIR / f"clip_{datetime.now():%Y%m%d_%H%M%S}.wav"
    torchaudio.save(str(out_path), torch.from_numpy(audio), sample_rate)

    msg = f"✅ {seconds}s clip generated in {time.time()-start:.1f}s — saved as {out_path.name}"
    return (sample_rate, audio), msg

# ─────────────────────────────── UI layer ──────────────────────────────────

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
        status = gr.Markdown("Loading model … (first call only)")
        audio_out = gr.Audio(label="Preview", streaming=False)

        def _wrapped_generate(p, l, cpu_flag, mps_flag):
            audio, msg = generate_audio(p, int(l), cpu_flag, mps_flag)
            status.update(msg)
            return audio

        generate_btn.click(
            _wrapped_generate,
            inputs=[prompt, length_slider, cpu_checkbox, mps_checkbox],
            outputs=[audio_out],
        )

    return demo

# ───────────────────────────── entry‑point ─────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable‑Audio Gradio GUI (offline)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--share", action="store_true", help="Enable public Gradio share link")
    parser.add_argument("--mps", action="store_true", help="Force MPS inference (Apple‑silicon)")
    args = parser.parse_args()

    iface = build_interface(force_cpu_cli=args.cpu)
    iface.queue(default_concurrency_limit=1)
    iface.launch(share=args.share)
