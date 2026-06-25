"""
  ░██████   ░██     ░██ ░██         ░██     ░██ ░███     ░███ ░████████     ░██████
 ░██   ░██  ░██     ░██ ░██         ░██     ░██ ░████   ░████ ░██    ░██   ░██   ░██
░██         ░██     ░██ ░██         ░██     ░██ ░██░██ ░██░██ ░██    ░██  ░██     ░██
 ░████████  ░██████████ ░██         ░██     ░██ ░██ ░████ ░██ ░████████   ░██     ░██
        ░██ ░██     ░██ ░██         ░██     ░██ ░██  ░██  ░██ ░██     ░██ ░██     ░██
 ░██   ░██  ░██     ░██ ░██          ░██   ░██  ░██       ░██ ░██     ░██  ░██   ░██
  ░██████   ░██     ░██ ░██████████   ░██████   ░██       ░██ ░█████████    ░██████

SHLUMBO — Shifting Hue by Leveraging Undulation and Motion Based Oscillation
Features: integer LUTs · uint8 HSV pipeline · vectorized indexing · precomputed tables
         direct circular hue rotation · triangle-wave phase offsets · binary motion
         activation · pixelated block-preserving effect · hard-edged glitch chroma blocks
"""

import os, json, time, queue, threading, subprocess
import cv2, numpy as np, imageio_ffmpeg

_SENTINEL = object()

DEFAULTS = {
    "output_fps": 24.0, "proc_width": 640, "proc_height": 480, "block_size": 16,
    "decay_rate": 0.92, "motion_threshold": 0.08, "write_queue_size": 8,
    "rot_freq": 0.08,   # hue-wheel rotations/sec
    "osc_freq": 0.25,   # oscillations/sec
    "osc_amp": 18.0,    # degrees of hue deviation
    "binary_motion_mode": True, "pixelated_motion_blocks": True,
}

# ── Config ───────────────────────────────────────────────────────────────────

def load_or_create_config(path="shlumbo_config.json", reinit=False, overrides=None):
    cfg = {**DEFAULTS, **(overrides or {})} if reinit else (
        {**DEFAULTS, **json.load(open(path))} if os.path.exists(path) else DEFAULTS.copy()
    )
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"{'Wrote' if reinit else 'Loaded'} config → {path}")
    return cfg

# ── Async pipe writer ────────────────────────────────────────────────────────

def _writer(proc, q):
    """Drain the write queue into ffmpeg's stdin on a background thread."""
    while True:
        frame = q.get()
        if frame is _SENTINEL:  # identity check avoids ambiguous array == comparison
            break
        try:
            proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            break
    proc.stdin.close()

# ── Main pipeline ────────────────────────────────────────────────────────────

def run(cfg, input_path, output_path):
    W, H, BS = cfg["proc_width"], cfg["proc_height"], cfg["block_size"]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    cap = cv2.VideoCapture(input_path)
    fps_in   = cap.get(cv2.CAP_PROP_FPS)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step     = fps_in / cfg["output_fps"]   # source frames per output frame

    # spawn ffmpeg: raw BGR in → h264 mp4 out, copy audio if present
    proc = subprocess.Popen([
        ffmpeg, "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{W}x{H}", "-r", str(cfg["output_fps"]), "-i", "pipe:0",
        "-i", input_path,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-preset", "faster", "-acodec", "copy",
        "-movflags", "+faststart", output_path, "-y",
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    q = queue.Queue(maxsize=cfg["write_queue_size"])
    threading.Thread(target=_writer, args=(proc, q), daemon=True).start()

    prev_gray = None
    motion    = np.zeros((H // BS, W // BS), np.float32)
    interp_pix = cv2.INTER_NEAREST if cfg["pixelated_motion_blocks"] else cv2.INTER_LINEAR
    t0 = time.perf_counter()
    out_count  = 0

    print("Running SHLUMBO…")
    for idx in range(1, total + 1):
        ret, frame = cap.read()
        if not ret:
            break

        # ── FPS decimation: skip source frames not needed for output ──────
        if int(idx / step) <= int((idx - 1) / step):
            continue

        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

        # ── Motion detection on downsampled blocks ────────────────────────
        small = cv2.resize(frame, (W // BS, H // BS), interpolation=cv2.INTER_AREA)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray; q.put(frame); continue

        diff = cv2.absdiff(gray, prev_gray).astype(np.float32) / 255.0
        _, mask = cv2.threshold(diff, cfg["motion_threshold"], 1.0, cv2.THRESH_BINARY)
        motion  = np.clip(mask + motion / cfg["decay_rate"], 0.0, 1.0)  # exponential decay
        mask_hr = cv2.resize(motion, (W, H), interpolation=interp_pix)   # upscale to frame

        # ── Hue shift: rotation + sinusoidal oscillation ──────────────────
        t     = time.perf_counter() - t0
        shift = (180.0 * cfg["rot_freq"] * t                          # continuous rotation
                 + cfg["osc_amp"] * np.sin(2*np.pi * cfg["osc_freq"] * t))  # oscillation

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        Hch  = hsv[:, :, 0].astype(np.float32)
        H_rot = (Hch + shift) % 180  # circular hue wrap

        # apply shift only to moving pixels (binary) or blend (continuous)
        if cfg["binary_motion_mode"]:
            H_out = np.where(mask_hr > 0.02, H_rot, Hch)
        else:
            H_out = Hch * (1 - mask_hr) + H_rot * mask_hr

        hsv[:, :, 0] = H_out.astype(np.uint8)
        q.put(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        prev_gray = gray

        # ── Progress bar ──────────────────────────────────────────────────
        out_count += 1
        pct  = idx / total
        bar  = int(pct * 40)
        eta  = (time.perf_counter() - t0) / pct * (1 - pct) if pct else 0
        print(f"\r  [{'█'*bar}{'░'*(40-bar)}] {pct*100:5.1f}%  "
              f"frame {idx}/{total}  ETA {eta:5.1f}s  ", end="", flush=True)

    cap.release()
    q.put(_SENTINEL)
    proc.wait()
    print(f"\n  Done → {output_path}  ({out_count} frames written)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    REINIT = True  # False = reuse saved config

    cfg = load_or_create_config(reinit=REINIT, overrides={
        "output_fps": 30.0, "proc_width": 480, "proc_height": 360,
        "block_size": 1, "decay_rate": 1.0, "motion_threshold": 0.4,
        "rot_freq": 0.5, "osc_freq": 1.0, "osc_amp": 90.0,
        "binary_motion_mode": True, "pixelated_motion_blocks": True,
    } if REINIT else None)

    run(cfg, input_path="cat_dog_480x360_30fps.mp4", output_path="cat_dog_SHLUMBO.mp4")