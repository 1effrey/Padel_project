"""train_ball.py
Fine-tune TrackNetV2 on the ball labels you produced with `main.py --label-ball`,
and save the weights the detector loads (config["ball"]["weights"]).

This is the step that turns the STUB detector into a real one. Pipeline:

    labeled (frame, u, v) CSV  +  the source video
        -> for each labeled frame: stack of `in_frames` consecutive frames (the
           SAME preprocessing the detector uses at inference) as input
        -> a Gaussian heatmap at (u, v) (or all-zeros if "not visible") as target
        -> train TrackNetV2 to reproduce the heatmap (weighted BCE)
        -> save the best weights to weights/ball_tracknet_v2.pt

USAGE
    # validate your data path WITHOUT training (reads a few samples, prints shapes)
    python train_ball.py --config config-side1.json --dry-run

    # prove the training machinery works on random data (no video/labels needed)
    python train_ball.py --smoke

    # real fine-tune (one or both cameras -- each config needs its labels CSV)
    python train_ball.py --config config-side1.json --config2 config-side2.json --epochs 30

The labels CSV for a clip is output/ball_labels_<clip>.csv (made by --label-ball).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from core.ball_detector import (TrackNetV2, make_gaussian_heatmap, preprocess_frame)
from core.ball_label import _csv_path, _load_existing


# --------------------------------------------------------------------------- #
# Dataset: labeled frames -> (input stack, target heatmap)
# --------------------------------------------------------------------------- #
def _gather_needed(samples: List[tuple], ci: int, in_frames: int) -> set:
    """Every frame index (>=0) needed to build the in_frames stacks for clip ci."""
    needed: set = set()
    for (c, center, _vis, _u, _v) in samples:
        if c == ci:
            for j in range(center - in_frames + 1, center + 1):
                if j >= 0:
                    needed.add(j)
    return needed


class BallDataset(Dataset):
    """One sample per labeled frame.

    PERFORMANCE: frames are CACHED ONCE per clip via a single SEQUENTIAL pass
    (resized to network resolution, kept as uint8 in RAM) -- NOT seeked per sample.
    Per-sample seeking is the trap: the DataLoader shuffles, so every read would jump
    to a different point in a 4K H.264 file, and each seek decodes from the nearest
    keyframe. That is so slow it looks like the program hung. Reading straight through
    once is fast AND frame-exact (no keyframe-seek misalignment), and training then
    becomes an in-memory lookup.

    MEMORY: one cached frame is net_w*net_h*3 bytes (~0.44 MB at 512x288). A few
    thousand labeled frames is ~1-2 GB -- the cache prints its estimate so you know.
    """

    def __init__(self, clips: List[Dict[str, Any]], net_w: int, net_h: int,
                 in_frames: int, sigma: float) -> None:
        self.clips = clips
        self.net_w, self.net_h = int(net_w), int(net_h)
        self.in_frames, self.sigma = int(in_frames), float(sigma)
        self._failed_idx: set = set()    # samples whose frames could not be read

        self.samples: List[tuple] = []   # (clip_idx, center_idx, visible, u, v)
        for ci, clip in enumerate(clips):
            for frame_idx, (vis, u, v) in clip["labels"].items():
                if frame_idx < in_frames - 1:
                    continue              # not enough preceding frames for a stack
                self.samples.append((ci, frame_idx, vis, u, v))

        # ci -> {frame_idx: uint8 BGR (net_h, net_w, 3)}
        self._cache: Dict[int, Dict[int, np.ndarray]] = {}
        self._build_cache()

    def __len__(self) -> int:
        return len(self.samples)

    def _build_cache(self) -> None:
        """Single sequential pass per clip; store each needed frame at net res."""
        for ci, clip in enumerate(self.clips):
            needed = _gather_needed(self.samples, ci, self.in_frames)
            self._cache[ci] = {}
            if not needed:
                continue
            end = max(needed)
            mb = len(needed) * self.net_w * self.net_h * 3 / 1e6
            print(f"[train] caching {len(needed)} frames from {clip['video']} "
                  f"(sequential scan 0..{end}, ~{mb:.0f} MB RAM)...")
            cap = cv2.VideoCapture(clip["video"])
            cur = 0
            while cur <= end:
                ok, fr = cap.read()       # read straight through -> frame-exact index
                if not ok:
                    break
                if cur in needed:
                    self._cache[ci][cur] = cv2.resize(fr, (self.net_w, self.net_h))
                if cur % 1000 == 0:
                    print(f"[train]   ...scanned {cur}/{end} "
                          f"({len(self._cache[ci])} cached)")
                cur += 1
            cap.release()
            print(f"[train]   cached {len(self._cache[ci])}/{len(needed)} frames.")

    def close(self) -> None:
        """Free the cached frames."""
        self._cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, i: int):
        ci, idx, vis, u, v = self.samples[i]
        clip = self.clips[ci]
        cache = self._cache.get(ci, {})
        wins = [cache.get(j) for j in range(idx - self.in_frames + 1, idx + 1)]
        if any(w is None for w in wins):
            # a needed frame was not cached (unreadable). NEVER pair a black input
            # with a real ball target -- that teaches the net to hallucinate. Emit
            # zeros input AND zeros target.
            self._failed_idx.add(i)
            x = np.zeros((3 * self.in_frames, self.net_h, self.net_w), dtype=np.float32)
            u_net = v_net = None
        else:
            x = np.concatenate(
                [preprocess_frame(w, self.net_w, self.net_h) for w in wins], axis=0)
            if vis:
                u_net = u * self.net_w / clip["orig_w"]
                v_net = v * self.net_h / clip["orig_h"]
            else:
                u_net = v_net = None
        y = make_gaussian_heatmap(self.net_w, self.net_h, u_net, v_net, self.sigma)
        return torch.from_numpy(x), torch.from_numpy(y[None, ...])   # (C,H,W),(1,H,W)


# --------------------------------------------------------------------------- #
# Loss + metric
# --------------------------------------------------------------------------- #
def weighted_bce(pred: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """BCE on the heatmap, up-weighting the rare ball pixels so the net doesn't just
    predict 'all zeros' (which would already score well -- the ball is tiny)."""
    eps = 1e-6
    pred = pred.clamp(eps, 1.0 - eps)
    w = 1.0 + (pos_weight - 1.0) * (target > 0).float()
    return -(w * (target * torch.log(pred) + (1 - target) * torch.log(1 - pred))).mean()


def loc_hits(pred: torch.Tensor, target: torch.Tensor, tol_px: float):
    """For VISIBLE samples, count how many predicted peaks land within tol_px of the
    labeled peak. Returns (hits, n_visible)."""
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    hits = n = 0
    for i in range(len(p)):
        tm = t[i, 0]
        if tm.max() <= 0:
            continue                      # 'not visible' sample -> no location to score
        n += 1
        gy, gx = np.unravel_index(int(tm.argmax()), tm.shape)
        py, px = np.unravel_index(int(p[i, 0].argmax()), p[i, 0].shape)
        if ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5 <= tol_px:
            hits += 1
    return hits, n


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pick_device(want: str) -> torch.device:
    w = str(want).lower()
    if "cuda" in w and not torch.cuda.is_available():
        print("[train] WARNING: cuda requested but unavailable -> CPU.")
        w = "cpu"
    return torch.device("cuda" if "cuda" in w else "cpu")


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _build_clips(config_paths: List[str], out_dir_default: str) -> List[Dict[str, Any]]:
    """For each config, attach its labels CSV + source video dimensions (skip configs
    with no labels yet)."""
    clips: List[Dict[str, Any]] = []
    for cp in config_paths:
        cfg = _load_config(cp)
        out_dir = cfg.get("output", {}).get("dir", out_dir_default)
        source = cfg["source"]
        csv_path = _csv_path(out_dir, source)
        labels = _load_existing(csv_path)
        if not labels:
            print(f"[train] no labels found for {source} at {csv_path} -- skipping. "
                  f"(run: python main.py --config {cp} --label-ball)")
            continue
        cap = cv2.VideoCapture(source)
        opened = cap.isOpened()
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if not opened or orig_w <= 0 or orig_h <= 0:
            print(f"[train] could not open video {source} (got {orig_w}x{orig_h}) "
                  f"-- skipping this clip.")
            continue
        clips.append({"video": source, "labels": labels,
                      "orig_w": orig_w, "orig_h": orig_h})
        n_vis = sum(1 for x in labels.values() if x[0] == 1)
        print(f"[train] {source}: {len(labels)} labels ({n_vis} visible), {orig_w}x{orig_h}")
    return clips


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_smoke() -> None:
    """Prove forward + loss + backward + save/reload work, on random data only."""
    print("[train] SMOKE: training machinery on random data (no video/labels).")
    dev = _pick_device("cpu")
    model = TrackNetV2(in_frames=3).to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(2, 9, 288, 512, device=dev)
    y = torch.stack([
        torch.from_numpy(make_gaussian_heatmap(512, 288, 100, 80)[None]),
        torch.from_numpy(make_gaussian_heatmap(512, 288, 300, 200)[None]),
    ]).to(dev)
    for step in range(2):
        opt.zero_grad()
        loss = weighted_bce(model(x), y, pos_weight=50.0)
        loss.backward()
        opt.step()
        print(f"  step {step}: loss={loss.item():.4f}")
    tmp = os.path.join("weights", "_smoke_test.pt")
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), tmp)
    TrackNetV2(in_frames=3).load_state_dict(torch.load(tmp, map_location="cpu"))
    os.remove(tmp)
    print("[train] SMOKE OK: forward/backward/save/reload all work.")


def run_train(args: argparse.Namespace) -> None:
    config_paths = [args.config] + ([args.config2] if args.config2 else [])
    base_cfg = _load_config(args.config)
    ball = base_cfg.get("ball", {})
    net_w = ball.get("input_width", 512)
    net_h = ball.get("input_height", 288)
    in_frames = ball.get("in_frames", 3)
    out_weights = args.out or ball.get("weights", "weights/ball_tracknet_v2.pt")

    clips = _build_clips(config_paths, base_cfg.get("output", {}).get("dir", "output"))
    if not clips:
        print("[train] No labeled clips found. Label some frames first:\n"
              "        python main.py --config config-side1.json --label-ball")
        return

    ds = BallDataset(clips, net_w, net_h, in_frames, sigma=args.sigma)
    if len(ds) == 0:
        print("[train] Dataset is empty (labels too early in the clip for a stack?).")
        ds.close()
        return

    if args.dry_run:
        x, y = ds[0]
        print(f"[train] DRY-RUN ok: input {tuple(x.shape)}, target {tuple(y.shape)}, "
              f"target peak={float(y.max()):.2f}. Data path readable ({len(ds)} samples).")
        ds.close()
        return

    if len(ds) < 2:
        print(f"[train] only {len(ds)} sample(s) -- need >=2 to both train and "
              f"validate. Label more frames.")
        ds.close()
        return

    # deterministic train/val split, guaranteeing a NON-EMPTY train set
    idxs = list(range(len(ds)))
    random.Random(0).shuffle(idxs)
    n_val = min(max(1, int(len(ds) * args.val_frac)), len(ds) - 1)
    val_ds = Subset(ds, idxs[:n_val])
    train_ds = Subset(ds, idxs[n_val:])
    print(f"[train] {len(ds)} samples -> {len(train_ds)} train / {len(val_ds)} val")

    dev = _pick_device(args.device or base_cfg.get("device", "cuda"))
    model = TrackNetV2(in_frames=in_frames).to(dev)
    if args.warm_start and os.path.isfile(args.warm_start):
        model.load_state_dict(torch.load(args.warm_start, map_location=dev), strict=False)
        print(f"[train] warm-started from {args.warm_start}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    nw = max(0, int(args.num_workers))
    pin = dev.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=nw, pin_memory=pin,
                          persistent_workers=(nw > 0))
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=nw, pin_memory=pin,
                        persistent_workers=(nw > 0))

    os.makedirs(os.path.dirname(out_weights) or ".", exist_ok=True)
    best_acc = -1.0
    best_val_loss = float("inf")
    use_amp = args.amp and dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("[train] mixed-precision (FP16) ON")
    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for x, y in train_dl:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                pred = model(x)                 # conv in FP16 (fast) when AMP is on
            # loss in FP32 for numerical safety (the weighted BCE takes logs)
            loss = weighted_bce(pred.float(), y, pos_weight=args.pos_weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * len(x)
        tr_loss /= max(1, len(train_ds))

        model.eval()
        val_loss = 0.0
        hits = seen = 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(dev, non_blocking=True)
                y = y.to(dev, non_blocking=True)
                with torch.autocast(device_type=dev.type, enabled=use_amp):
                    pred = model(x)
                pred = pred.float()
                val_loss += weighted_bce(pred, y, pos_weight=args.pos_weight).item() * len(x)
                h, n = loc_hits(pred, y, tol_px=args.tol_px)
                hits += h
                seen += n
        val_loss /= max(1, len(val_ds))
        acc = (hits / seen) if seen else 0.0
        print(f"[train] epoch {epoch + 1}/{args.epochs}  train_loss={tr_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_loc@{args.tol_px}px={acc * 100:.1f}% ({hits}/{seen})")

        # Save-best: prefer a STRICT improvement in localization accuracy. When the
        # val split happens to have no visible balls to score (seen==0), fall back to
        # val_loss. Always save on epoch 0 so a weights file exists.
        improved = (acc > best_acc) if seen > 0 else (val_loss < best_val_loss)
        if epoch == 0 or improved:
            if seen > 0:                      # only track acc we actually measured
                best_acc = max(best_acc, acc)
            best_val_loss = min(best_val_loss, val_loss)
            torch.save(model.state_dict(), out_weights)
            print(f"[train]   saved -> {out_weights} "
                  f"(val_loc={acc * 100:.1f}%, val_loss={val_loss:.4f})")

    if ds._failed_idx:
        print(f"[train] WARNING: {len(ds._failed_idx)} sample(s) could not be read "
              f"and were zeroed -- check your video path / seeking.")
    ds.close()
    best_str = f"{best_acc * 100:.1f}%" if best_acc >= 0 else "n/a (no visible val labels)"
    print(f"[train] done. best val localization = {best_str}. "
          f"Test it:  python main.py --ball-eval --save-video")


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune TrackNetV2 on labeled padel-ball frames")
    p.add_argument("--config", default="config.json")
    p.add_argument("--config2", help="optional second camera's config (train on both)")
    p.add_argument("--out", help="output weights path (default: config ball.weights)")
    p.add_argument("--warm-start", help="weights to start from (e.g. a public checkpoint)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--sigma", type=float, default=3.0, help="Gaussian target radius (net px)")
    p.add_argument("--pos-weight", type=float, default=50.0, help="up-weight ball pixels in BCE")
    p.add_argument("--tol-px", type=float, default=4.0, help="localization hit tolerance (net px)")
    p.add_argument("--device", help="cuda / cpu (default: config device)")
    # SPEED (opt-in; defaults keep the laptop behaviour unchanged). On a big cloud GPU
    # the data loader, not the GPU, is the bottleneck -- parallel workers + a larger
    # batch + mixed precision keep the GPU fed. Use --num-workers 0 on Windows (the
    # cached dataset pickles badly under 'spawn'); 8 is great on a Linux pod.
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (0 = laptop/Windows-safe; 8 on a Linux pod)")
    p.add_argument("--amp", action="store_true",
                   help="mixed-precision (FP16) training -- faster on CUDA, no quality change")
    p.add_argument("--dry-run", action="store_true", help="check the data path, don't train")
    p.add_argument("--smoke", action="store_true", help="prove the machinery on random data")
    args = p.parse_args()

    if args.smoke:
        run_smoke()
    else:
        run_train(args)


if __name__ == "__main__":
    main()
