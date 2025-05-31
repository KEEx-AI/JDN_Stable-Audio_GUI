#!/usr/bin/env python3
"""
Stable‑Audio GUI — v3.1
———————————————
Fixes the **SyntaxError** in `_poll()` caused by a truncated line. The offending
line now correctly calls `messagebox.showerror()`, and the `try/except` block is
properly closed. No other logic changed.
"""

from __future__ import annotations

# ------------------------------------------------------------------ bootstrap
import os, sys, queue, threading, time, traceback, argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

# disable torchsde (recursion bug) – stable_audio_tools doesn’t need it
sys.modules.setdefault("torchsde", None)
os.environ.setdefault("DIFFUSERS_DISABLE_TORCHSDE", "1")

# ------------------------------------------------------------------ deps
import torch, torchaudio, sounddevice as sd, soundfile as sf, tkinter as tk
from tkinter import ttk, messagebox
from einops import rearrange

try:
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond
except ImportError as exc:
    raise RuntimeError(
        "`stable_audio_tools` missing — install with: pip install stable-audio-tools==0.0.19"
    ) from exc

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "generated_clips"; AUDIO_DIR.mkdir(exist_ok=True)
MODEL_NAME = "stabilityai/stable-audio-open-1.0"
_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# ------------------------------------------------------------------ helpers
def device_str(force_cpu=False):
    if force_cpu or (not torch.cuda.is_available() and not torch.backends.mps.is_available()):
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "mps"

def load_model(device: str):
    model, cfg = get_pretrained_model(MODEL_NAME)
    return model.to(device), cfg

def gen_worker(model, cfg, prompt, out_path, device, q):
    try:
        audio = generate_diffusion_cond(
            model,
            conditioning=[{"prompt": prompt}],
            sample_size=cfg["sample_size"],
            device=device,
        )
        audio = rearrange(audio, "b d n -> d (b n)")
        audio = (audio / audio.abs().max()).mul(32767).clamp(-32767,32767).to(torch.int16).cpu()
        torchaudio.save(str(out_path), audio, cfg["sample_rate"])
        q.put(("success", str(out_path)))
    except Exception as e:
        (ROOT/"error.log").write_text(traceback.format_exc())
        q.put(("error", str(e)))

# ------------------------------------------------------------------ GUI
class App(tk.Tk):
    def __init__(self, force_cpu=False):
        super().__init__()
        self.title("Stable‑Audio GUI")
        self.geometry("720x560")
        self.resizable(False, False)
        self.q: "queue.Queue[tuple[str,str]]" = queue.Queue()
        self.model=None; self.cfg=None; self.device=None; self.busy=False; self.spin=0

        # prompts
        pf=ttk.LabelFrame(self,text="Prompt"); pf.pack(fill="x",padx=12,pady=(12,6))
        self.prompt=tk.Text(pf,height=3,wrap="word"); self.prompt.pack(fill="both",padx=6,pady=6)

        # controls
        cf=ttk.Frame(self); cf.pack(fill="x",padx=12)
        self.btn=ttk.Button(cf,text="Generate",command=self.gen); self.btn.pack(side="left",padx=4,pady=8)
        ttk.Label(cf,text="Status:").pack(side="left",padx=(20,4))
        self.status=tk.StringVar(value="Loading model …"); ttk.Label(cf,textvariable=self.status).pack(side="left")

        # list
        lf=ttk.LabelFrame(self,text="Clips (dbl‑click to play)"); lf.pack(fill="both",expand=True,padx=12,pady=12)
        self.list=tk.Listbox(lf); self.list.pack(fill="both",expand=True,padx=6,pady=6)
        self.list.bind("<Double-Button-1>",self.play)

        threading.Thread(target=self.load_bg,args=(force_cpu,),daemon=True).start()
        self.after(120,self.poll)

    def load_bg(self,force_cpu):
        try:
            self.device=device_str(force_cpu)
            self.model,self.cfg=load_model(self.device)
            self.q.put(("ready",""))
        except Exception as e:
            self.q.put(("fatal",str(e)))

    def gen(self):
        prompt=self.prompt.get("1.0","end").strip()
        if not prompt:return messagebox.showwarning("Empty","Type a prompt.")
        if self.model is None:return messagebox.showinfo("Wait","Model loading …")
        if self.busy:return
        out=AUDIO_DIR/f"clip_{datetime.now():%Y%m%d_%H%M%S}.wav"
        self.busy=True; self.btn.config(state="disabled"); self.status.set("Generating …")
        threading.Thread(target=gen_worker,args=(self.model,self.cfg,prompt,out,self.device,self.q),daemon=True).start()

    def poll(self):
        try:
            while True:
                kind,payload=self.q.get_nowait()
                if kind=="ready": self.status.set(f"Model ready on {self.device}")
                elif kind=="success": self.list.insert(tk.END,Path(payload).name); self.status.set("Saved "+Path(payload).name); self.reset()
                elif kind=="error": self.status.set("Error — see error.log"); messagebox.showerror("Error",payload); self.reset()
                elif kind=="fatal": self.status.set("Model load failed"); messagebox.showerror("Fatal",payload); self.btn.config(state="disabled")
        except queue.Empty: pass
        self.title(f"{_SPIN[self.spin%len(_SPIN)]} Stable‑Audio GUI" if self.busy or self.model is None else "Stable‑Audio GUI"); self.spin+=1
        self.after(120,self.poll)

    def reset(self):
        self.busy=False; self.btn.config(state="normal")

    def play(self,_):
        sel=self.list.curselection();
        if not sel:return
        wav=AUDIO_DIR/self.list.get(sel[0])
        try:
            data,sr=sf.read(wav,dtype="int16"); sd.stop(); sd.play(data,sr); self.status.set(f"Playing {wav.name}")
        except Exception as e:
            messagebox.showerror("Playback",str(e))

# ------------------------------------------------------------------ main
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--cpu",action="store_true"); a=p.parse_args()
    App(force_cpu=a.cpu).mainloop()
