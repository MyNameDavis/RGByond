# SHLUMBO_RPI_4B_V3.py - Optimized for Raspberry Pi 4B
import cv2
import numpy as np
import time
import threading
import queue
import statistics
import os

# --- CONFIGURATION ---
W, H, TARGET_FPS = 320, 240, 24
OUTPUT_PATH = "10.0.0.253" #"shlumbo_out.mp4"
BENCH_MODE, BENCH_SECONDS, WARMUP_FRAMES = True, 10, 30
_PI4B_REF = {'resize_gray': 0.013, 'motion': 0.008, 'color_lut': 0.010}

# --- ENCODER THREAD ---
# --- ENCODER THREAD ---
class EncoderThread(threading.Thread):
    # Replaced mp4mux and filesink with mpegtsmux and udpsink
    _GSTR_HW = 'appsrc ! videoconvert ! v4l2h264enc extra-controls="controls,repeat_sequence_header=1" ! h264parse ! mpegtsmux ! udpsink host={path} port=5000 sync=false'
    _GSTR_SW = 'appsrc ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency ! mpegtsmux ! udpsink host={path} port=5000 sync=false'

    def __init__(self, path, width, height, fps):
        super().__init__(daemon=True)
        self.path, self.width, self.height, self.fps = path, width, height, fps
        self.q = queue.Queue(maxsize=4)
        self.dropped, self.encoded = 0, 0
        self.latencies = []
        self._running = True
        self._writer, self._backend = None, "pending"

    def _open_writer(self):
        for gst, label in [
            (self._GSTR_HW.format(path=self.path), "HW GStreamer"),
            (self._GSTR_SW.format(path=self.path), "SW GStreamer")
        ]:
            try:
                w = cv2.VideoWriter(gst, cv2.CAP_GSTREAMER, 0, self.fps, (self.width, self.height))
                if w.isOpened(): return w, label
                w.release()
            except Exception:
                pass

        path_avi = self.path.replace(".mp4", ".avi")
        w = cv2.VideoWriter(path_avi, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (self.width, self.height))
        if w.isOpened(): return w, "MJPEG"
        return None, "None"

    def run(self):
        self._writer, self._backend = self._open_writer()
        while self._running:
            try:
                frame, enqueue_t = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._writer:
                self._writer.write(frame)
            self.latencies.append(time.perf_counter() - enqueue_t)
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

    def set_params(self, **kw):
        old_bs = self.block_size
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, max(1, v) if k == 'block_size' else v)
        if self.block_size != old_bs:
            self._alloc_buffers()
            self._prev_gray = None

    def _build_lut(self, shift_norm):
        base, offset = self._lut_base, int(shift_norm * 256)
        b = ((base + offset) & 255).astype(np.uint8)
        g = ((base + offset + 85) & 255).astype(np.uint8)
        r = ((base + offset + 170) & 255).astype(np.uint8)
        return np.stack([b, g, r], axis=-1).reshape(256, 1, 3)

    def process(self, frame):
        bs, times = self.block_size, {}

        # Resize & Grayscale
        t = time.perf_counter()
        small = cv2.resize(frame, (W // bs, H // bs), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        times['resize_gray'] = time.perf_counter() - t

        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            return None, {}

        # Motion Detection
        t = time.perf_counter()
        diff = cv2.absdiff(gray, self._prev_gray)
        cv2.threshold(diff, int(self.motion_th * 255), 255, cv2.THRESH_BINARY, dst=self._mask_small)
        cv2.resize(self._mask_small, (W, H), dst=self._mask_hr, interpolation=cv2.INTER_NEAREST)

        decay_u8, inv_u8 = int(self.decay * 255), 255 - int(self.decay * 255)
        self._motion_accum = ((self._motion_accum.astype(np.uint16) * decay_u8) >> 8).astype(np.uint8)
        self._motion_accum += ((self._mask_hr.astype(np.uint16) * inv_u8) >> 8).astype(np.uint8)
        times['motion'] = time.perf_counter() - t

        # LUT Color Shift
        t = time.perf_counter()
        now = time.perf_counter() - self._t0
        phase = (self.rot_freq * now + (self.osc_amp / 360.0) * np.sin(2 * np.pi * self.osc_freq * now))

        self._mapped[:] = cv2.LUT(frame, self._build_lut(phase % 1.0))
        cv2.threshold(self._motion_accum, 5, 255, cv2.THRESH_BINARY, dst=self._active_u8)
        cv2.copyTo(self._mapped, self._active_u8, frame)
        times['color_lut'] = time.perf_counter() - t

        self._prev_gray[:] = gray
        return frame, times

# --- SYNTHETIC CAMERA ---
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

# --- REPORT & BENCHMARKING ---
def print_report(stage_times, frame_times, encoder):
    BUDGET = 1.0 / TARGET_FPS
    STAGE_RATIO = {'resize_gray': 15, 'motion': 14, 'color_lut': 12}
    
    s_host = {k: statistics.mean(v) for k, v in stage_times.items()}
    s_pi = {k: s_host[k] * STAGE_RATIO[k] for k in s_host}
    total_host, total_pi = sum(s_host.values()), sum(s_pi.values())
    
    print(f"\n--- SHLUMBO V3 Pi 4B Report ---\n{W}x{H} @ {TARGET_FPS} fps")
    for stage in s_host:
        print(f"{stage:<16} Host: {s_host[stage]*1000:>6.2f}ms | Pi: {s_pi[stage]*1000:>6.2f}ms")
    print(f"TOTAL            Host: {total_host*1000:>6.2f}ms | Pi: {total_pi*1000:>6.2f}ms")
    print(f"-> Est Pi FPS: {1.0 / total_pi if total_pi else 0:.1f} | Enc: {encoder.backend} | Dropped: {encoder.dropped}")

def run_benchmark():
    print("\nSHLUMBO V3 Benchmark\n")
    encoder = EncoderThread(OUTPUT_PATH, W, H, TARGET_FPS)
    encoder.start()
    time.sleep(0.5)

    cam = SyntheticCamera()
    pipeline = ShlumboPipeline()
    stage_times = {'resize_gray': [], 'motion': [], 'color_lut': []}
    frame_times = []

    for _ in range(WARMUP_FRAMES):
        pipeline.process(cam.read()[1])

    t_start = time.perf_counter()
    while (time.perf_counter() - t_start < BENCH_SECONDS):
        _, frame = cam.read()
        t0 = time.perf_counter()
        out, times = pipeline.process(frame)
        
        if out is None: continue
        for k, v in times.items(): stage_times[k].append(v)
        frame_times.append(time.perf_counter() - t0)
        encoder.submit(out)

    encoder.stop()
    print_report(stage_times, frame_times, encoder)

# --- LIVE MODE ---
def run_live():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    if not cap.isOpened(): 
        raise RuntimeError("Could not open camera")
    
    encoder = EncoderThread(OUTPUT_PATH, W, H, TARGET_FPS)
    encoder.start()
    
    try: 
        os.sched_setaffinity(encoder.ident, {3})
    except Exception: 
        pass
    
    pipeline = ShlumboPipeline()
    cv2.namedWindow("SHLUMBO", cv2.WINDOW_NORMAL)
    last_t = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        out, _ = pipeline.process(frame)
        if out is None: continue
        
        encoder.submit(out)
        now = time.perf_counter()
        fps = 1.0 / (now - last_t)
        last_t = now
        
        cv2.putText(out, f"FPS:{fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("SHLUMBO", out)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    encoder.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if BENCH_MODE:
        run_benchmark()
    else:
        run_live()