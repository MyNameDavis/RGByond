import cv2
import numpy as np
import time

# ═══════════════════════════════════════════════════════════════
# CAMERA
# ═══════════════════════════════════════════════════════════════

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError("Camera could not be opened")

# ═══════════════════════════════════════════════════════════════
# CONTROLS WINDOW
# ═══════════════════════════════════════════════════════════════

cv2.namedWindow("SHLUMBO Controls", cv2.WINDOW_NORMAL)

cv2.createTrackbar("osc_amp",        "SHLUMBO Controls",  90,  180, lambda x: None)
cv2.createTrackbar("rot_freq x100",  "SHLUMBO Controls",  10,   50, lambda x: None)
cv2.createTrackbar("osc_freq x100",  "SHLUMBO Controls",  10,   50, lambda x: None)  # NEW
cv2.createTrackbar("motion_th x100", "SHLUMBO Controls",  20,  100, lambda x: None)
cv2.createTrackbar("block_size",     "SHLUMBO Controls",   8,   32, lambda x: None)
cv2.createTrackbar("decay x100",     "SHLUMBO Controls",  80,  100, lambda x: None)  # NEW

# smoothed state
osc_amp_s    = 90.0
rot_freq_s   = 0.1
osc_freq_s   = 0.1   # NEW – oscillation frequency (independent of rotation)
motion_th_s  = 0.2
block_size_s = 8
decay_s      = 0.80  # NEW – per-frame decay applied to the persistent motion mask

# ═══════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════

prev_gray        = None
motion_accum     = None   # NEW – persistent, decaying motion accumulator
t0               = time.perf_counter()
last_t           = time.perf_counter()

print("SHLUMBO running... Press Q to quit.")

# ═══════════════════════════════════════════════════════════════
# LOOP
# ═══════════════════════════════════════════════════════════════

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ───────────────────────────────
    # READ SLIDERS
    # ───────────────────────────────

    osc_amp_t    = cv2.getTrackbarPos("osc_amp",        "SHLUMBO Controls")
    rot_freq_t   = cv2.getTrackbarPos("rot_freq x100",  "SHLUMBO Controls") / 100.0
    osc_freq_t   = cv2.getTrackbarPos("osc_freq x100",  "SHLUMBO Controls") / 100.0   # NEW
    motion_th_t  = cv2.getTrackbarPos("motion_th x100", "SHLUMBO Controls") / 100.0
    block_size_t = max(1, cv2.getTrackbarPos("block_size", "SHLUMBO Controls"))
    decay_t      = cv2.getTrackbarPos("decay x100",     "SHLUMBO Controls") / 100.0   # NEW

    # smoothing
    osc_amp_s   = 0.9 * osc_amp_s   + 0.1 * osc_amp_t
    rot_freq_s  = 0.9 * rot_freq_s  + 0.1 * rot_freq_t
    osc_freq_s  = 0.9 * osc_freq_s  + 0.1 * osc_freq_t   # NEW
    motion_th_s = 0.9 * motion_th_s + 0.1 * motion_th_t
    decay_s     = 0.9 * decay_s     + 0.1 * decay_t       # NEW

    new_block_size_s  = int(0.9 * block_size_s + 0.1 * block_size_t)
    new_block_size_s  = max(1, new_block_size_s)
    block_size_changed = (new_block_size_s != block_size_s)
    block_size_s      = new_block_size_s
    block_size        = block_size_s

    # ───────────────────────────────
    # FRAME PREP
    # ───────────────────────────────

    W, H = 640, 480
    frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

    small = cv2.resize(
        frame,
        (W // block_size, H // block_size),
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if prev_gray is None or block_size_changed:
        prev_gray    = gray
        motion_accum = np.zeros((H, W), dtype=np.float32)
        continue

    # ───────────────────────────────
    # MOTION  +  DECAY ACCUMULATOR
    # ───────────────────────────────

    diff = cv2.absdiff(gray, prev_gray).astype(np.float32) / 255.0
    _, mask_small = cv2.threshold(diff, motion_th_s, 1.0, cv2.THRESH_BINARY)
    mask_hr = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)

    # NEW: decay the accumulator then inject fresh motion
    motion_accum = motion_accum * decay_s + mask_hr * (1.0 - decay_s)
    # clamp to [0, 1]
    np.clip(motion_accum, 0.0, 1.0, out=motion_accum)

    # ───────────────────────────────
    # HSV SHIFT ENGINE
    # ───────────────────────────────

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    Hch = hsv[:, :, 0].astype(np.float32)

    t = time.perf_counter() - t0

    # rotation uses rot_freq; oscillation uses its own independent osc_freq  (NEW)
    shift_rot = 180.0 * rot_freq_s * t
    shift_osc = osc_amp_s * np.sin(2 * np.pi * osc_freq_s * t)   # was rot_freq_s

    shift = shift_rot + shift_osc
    H_rot = (Hch + shift) % 180

    # ───────────────────────────────
    # APPLY MOTION (uses accumulator)
    # ───────────────────────────────

    active = motion_accum > 0.02
    H_out = Hch.copy()
    H_out[active] = H_rot[active]

    hsv[:, :, 0] = H_out.astype(np.uint8)

    # saturation enhancement driven by accumulator instead of raw mask
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    s = 1.0 - (1.0 - s) ** (1.0 + 2.0 * motion_accum)
    hsv[:, :, 1] = np.clip(s * 255, 0, 255).astype(np.uint8)

    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # ───────────────────────────────
    # FPS DISPLAY
    # ───────────────────────────────

    now   = time.perf_counter()
    fps   = 1.0 / (now - last_t)
    last_t = now

    cv2.putText(out, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ───────────────────────────────
    # DISPLAY
    # ───────────────────────────────

    cv2.imshow("SHLUMBO Output", out)
    prev_gray = gray

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# ═══════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════

cap.release()
cv2.destroyAllWindows()