# SHLUMBO_RPI_4B_V3.py - Headless UDP Stream (libcamera optimized)
import cv2
import numpy as np
import time
import threading
import queue
import statistics
import os

# --- CONFIGURATION ---
W, H, TARGET_FPS = 320, 240, 24  # Change FPS to 30 for perfectly smooth cadence, or 24 for cinematic texture
OUTPUT_PATH = "10.0.0.253"       # Mac IP address for UDP streaming
BENCH_MODE, BENCH_SECONDS, WARMUP_FRAMES = False, 10, 30

# --- ENCODER THREAD ---
class EncoderThread(threading.Thread):
    _GSTR_HW = 'appsrc ! videoconvert ! v4l2h264enc extra-controls="controls,repeat_sequence_header=1" ! h264parse ! mpegtsmux ! udpsink host={path} port=5000 sync=false'
    _GSTR_SW = 'appsrc ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency ! mpegtsmux ! udpsink host={path} port=5000 sync=false'

    def __init__(self, path, width, height, fps):
        super().__init__(daemon=True)
        self.path, self.width, self.height, self.fps = path, width, height, fps
        self.q = queue.Queue(maxsize=4)
        self.dropped, self.encoded = 0, 0
        self._running = True
        self._writer, self._backend = None, "pending"

    def _open_writer(self):
        for gst, label in [
            # SW GStreamer moved to the top!
            (self._GSTR_SW.format(path=self.path), "SW GStreamer"),
            (self._GSTR_HW.format(path=self.path), "HW GStreamer")
        ]:
            try:
                w = cv2.VideoWriter(gst, cv2.CAP_GSTREAMER, 0, self.fps, (self.width, self.height))
                if w.isOpened(): return w, label
                w.release()
            except Exception:
                pass

    def run(self):
        self._writer, self._backend = self._open_writer()
        while self._running:
            try:
                frame, _ = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._writer:
                self._writer.write(frame)
            self.encoded += 1

    def submit(self, frame):
        try:
            self.q.put_nowait((frame.copy(), time.perf_counter()))
        except queue.Full:
            self.dropped += 1

    def stop(self):
        self._running = False
        self.join(timeout=5)
        if self._writer: 
            self._writer.release()

    @property
    def backend(self):
        return self._backend

# --- SHLUMBO PIPELINE ---
class ShlumboPipeline:
    def __init__(self):
        self.osc_amp, self.rot_freq, self.osc_freq = 90.0, 1.0, 0.5
        self.motion_th, self.block_size, self.decay = 0.20, 1, 0.9
        self._t0 = time.perf_counter()
        self._prev_gray = None
        self._lut_base = np.arange(256, dtype=np.uint16)
        self._alloc_buffers()

    def _alloc_buffers(self):
        sw, sh = W // self.block_size, H // self.block_size
        self._motion_accum = np.zeros((H, W), dtype=np.uint8)
        self._mask_small = np.zeros((sh, sw), dtype=np.uint8)
        self._mask_hr = np.zeros((H, W), dtype=np.uint8)
        self._active_u8 = np.zeros((H, W), dtype=np.uint8)
        self._mapped = np.zeros((H, W, 3), dtype=np.uint8)

    def _build_lut(self, shift_norm):
        base, offset = self._lut_base, int(shift_norm * 256)
        b = ((base + offset) & 255).astype(np.uint8)
        g = ((base + offset + 85) & 255).astype(np.uint8)
        r = ((base + offset + 170) & 255).astype(np.uint8)
        return np.stack([b, g, r], axis=-1).reshape(256, 1, 3)

    def process(self, frame):
        bs = self.block_size

        small = cv2.resize(frame, (W // bs, H // bs), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            return None

        diff = cv2.absdiff(gray, self._prev_gray)
        cv2.threshold(diff, int(self.motion_th * 255), 255, cv2.THRESH_BINARY, dst=self._mask_small)
        cv2.resize(self._mask_small, (W, H), dst=self._mask_hr, interpolation=cv2.INTER_NEAREST)

        decay_u8, inv_u8 = int(self.decay * 255), 255 - int(self.decay * 255)
        self._motion_accum = ((self._motion_accum.astype(np.uint16) * decay_u8) >> 8).astype(np.uint8)
        self._motion_accum += ((self._mask_hr.astype(np.uint16) * inv_u8) >> 8).astype(np.uint8)

        now = time.perf_counter() - self._t0
        phase = (self.rot_freq * now + (self.osc_amp / 360.0) * np.sin(2 * np.pi * self.osc_freq * now))

        self._mapped[:] = cv2.LUT(frame, self._build_lut(phase % 1.0))
        cv2.threshold(self._motion_accum, 5, 255, cv2.THRESH_BINARY, dst=self._active_u8)
        cv2.copyTo(self._mapped, self._active_u8, frame)

        self._prev_gray[:] = gray
        return frame

# --- SYNTHETIC CAMERA (FOR BENCHMARKING) ---
class SyntheticCamera:
    def __init__(self):
        self._idx = 0
        x, y = np.linspace(0, 4*np.pi, W), np.linspace(0, 4*np.pi, H)
        xx, yy = np.meshgrid(x, y)
        self._base = ((np.sin(xx) * np.cos(yy) + 1) * 127.5).astype(np.uint8)

    def read(self):
        i = self._idx
        self._idx += 1
        cx = int(W * 0.5 + W * 0.3 * np.sin(i * 0.07))
        cy = int(H * 0.5 + H * 0.2 * np.cos(i * 0.05))
        b = self._base.copy()
        cv2.circle(b, (cx, cy), 60, 200, -1)
        return True, np.stack([b, np.roll(b, i % 30, 1), np.roll(b, -(i % 20), 0)], axis=2)

# --- LIVE MODE (HEADLESS) ---
def run_live():
    # 1. Pull native 640x480 @ 60fps from ArduCAM via libcamera
    # 2. Decimate down to TARGET_FPS via videorate
    # 3. Scale cleanly to 320x240 for the BARPHF pipeline
   # Added format=BGR to the final caps filter
    ingest_pipe = (
        f"libcamerasrc ! video/x-raw, width=640, height=480, framerate=60/1 "
        f"! videorate ! video/x-raw, framerate={TARGET_FPS}/1 "
        f"! videoconvert ! videoscale ! video/x-raw, width={W}, height={H}, format=BGR "
        f"! appsink drop=true max-buffers=1"
    ) 
    
    cap = cv2.VideoCapture(ingest_pipe, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened(): 
        raise RuntimeError("Could not open ArduCAM via libcamerasrc. Check ribbon connection.")
    
    encoder = EncoderThread(OUTPUT_PATH, W, H, TARGET_FPS)
    encoder.start()
    
    try: 
        os.sched_setaffinity(encoder.ident, {3})
    except Exception: 
        pass
    
    pipeline = ShlumboPipeline()
    
    print(f"Live UDP stream active on {OUTPUT_PATH}:5000. Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("\n[!] ERROR: Camera connected, but returned an empty frame.")
                break
            
            out = pipeline.process(frame)
            if out is None: continue
            
            encoder.submit(out)
            
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
        
    finally:
        encoder.stop()
        cap.release()

# --- BENCHMARK MODE ---
def run_benchmark():
    print(f"\nSHLUMBO V3 Benchmark ({W}x{H} @ {TARGET_FPS} fps)\n")
    encoder = EncoderThread(OUTPUT_PATH, W, H, TARGET_FPS)
    encoder.start()
    time.sleep(0.5)

    cam = SyntheticCamera()
    pipeline = ShlumboPipeline()

    for _ in range(WARMUP_FRAMES):
        pipeline.process(cam.read()[1])

    t_start = time.perf_counter()
    while (time.perf_counter() - t_start < BENCH_SECONDS):
        _, frame = cam.read()
        out = pipeline.process(frame)
        if out is not None: encoder.submit(out)

    encoder.stop()
    print(f"-> Est Enc: {encoder.backend} | Dropped: {encoder.dropped}")

if __name__ == "__main__":
    if BENCH_MODE:
        run_benchmark()
    else:
        run_live()