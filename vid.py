#!/usr/bin/env python3
"""
ASCII Video Player - High-performance terminal-style video player with Tkinter GUI.
Supports real-time ASCII conversion, frame preloading, and keyboard controls.
"""

import cv2
import numpy as np
import threading
import time
import os
import sys
import json
import hashlib
import queue
import logging
from pathlib import Path
from collections import deque, OrderedDict
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, List
from enum import Enum
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import atexit

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ascii_player.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Enums and data classes
# ----------------------------------------------------------------------
class ColorMode(Enum):
    MONOCHROME = "monochrome"
    ANSI_GREYSCALE = "ansi_greyscale"
    ANSI_COLOR = "ansi_color"
    BLOCK_SHADES = "block_shades"


@dataclass
class VideoSettings:
    """User-configurable settings with validation."""
    width: int = 100
    font_size: int = 10
    brightness: float = 1.0
    contrast: float = 1.0
    color_mode: str = "ansi_greyscale"
    target_fps: int = 30
    audio_sync: bool = True
    preload_frames: int = 30
    chars: str = " .:-=+*#%@"
    enable_colors: bool = True          # Toggle colored output in GUI

    def __post_init__(self):
        self.width = max(40, min(300, self.width))
        self.font_size = max(6, min(24, self.font_size))
        self.brightness = max(0.0, min(2.0, self.brightness))
        self.contrast = max(0.0, min(3.0, self.contrast))
        self.target_fps = max(15, min(60, self.target_fps))
        self.preload_frames = max(10, min(100, self.preload_frames))


# ----------------------------------------------------------------------
# Charset and color utilities
# ----------------------------------------------------------------------
class CharsetManager:
    """Manages character sets and optional ANSI color gradients."""

    CHARSETS = {
        "default": " .:-=+*#%@",
        "detailed": " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
        "blocks": " ░▒▓█",
        "simple": " .oO0#",
        "matrix": " .,-:;+=x%#@"
    }

    # 8‑bit grayscale ANSI codes (Windows 10+ compatible)
    ANSI_SHADES = [
        "\033[38;5;232m",  # darkest
        "\033[38;5;235m",
        "\033[38;5;238m",
        "\033[38;5;241m",
        "\033[38;5;244m",
        "\033[38;5;247m",
        "\033[38;5;250m",
        "\033[38;5;253m",
        "\033[38;5;255m",  # brightest
    ]

    @classmethod
    def get_charset(cls, name: str = "default") -> str:
        return cls.CHARSETS.get(name, cls.CHARSETS["default"])

    @classmethod
    def get_ansi_color(cls, intensity: float, color_mode: str) -> str:
        if color_mode == "ansi_color":
            hue = int(intensity * 200)          # map 0‑1 → 0‑200
            return f"\033[38;5;{hue}m"
        elif color_mode == "ansi_greyscale":
            idx = min(7, int(intensity * 8))
            return cls.ANSI_SHADES[idx]
        return ""


# ----------------------------------------------------------------------
# Thread‑safe frame buffer with drop statistics
# ----------------------------------------------------------------------
class ThreadSafeFrameBuffer:
    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.frame_counter = 0
        self.dropped_frames = 0
        self.last_frame_time = 0.0

    def put(self, frame_data: Tuple[int, str, float]) -> None:
        with self.lock:
            self.buffer.append(frame_data)
            self.frame_counter += 1

    def get(self, timeout: float = 0.016) -> Optional[Tuple[int, str, float]]:
        start = time.time()
        while time.time() - start < timeout:
            with self.lock:
                if self.buffer:
                    frame_num, ascii_art, timestamp = self.buffer.popleft()
                    # frame dropping if timestamp is stale (seek or slowdown)
                    if timestamp < self.last_frame_time - 0.1:
                        self.dropped_frames += 1
                        continue
                    self.last_frame_time = timestamp
                    return frame_num, ascii_art, timestamp
            time.sleep(0.001)
        return None

    def get_stats(self) -> Dict:
        with self.lock:
            return {
                'buffered': len(self.buffer),
                'total_frames': self.frame_counter,
                'dropped_frames': self.dropped_frames
            }

    def clear(self) -> None:
        with self.lock:
            self.buffer.clear()
            self.dropped_frames = 0


# ----------------------------------------------------------------------
# LRU cache for ASCII conversion (prevents memory leaks)
# ----------------------------------------------------------------------
class LRUCache:
    """Simple LRU cache with max size."""
    def __init__(self, capacity: int = 200):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[str]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: str) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        self.cache.clear()


# ----------------------------------------------------------------------
# ASCII converter with caching and settings
# ----------------------------------------------------------------------
class ASCIIConverter:
    def __init__(self):
        self.settings = VideoSettings()
        self.cache = LRUCache(capacity=200)
        self._settings_hash = None

    def _get_settings_hash(self) -> int:
        """Hash of current settings for cache invalidation."""
        return hash((
            self.settings.width,
            self.settings.brightness,
            self.settings.contrast,
            self.settings.color_mode,
            self.settings.chars
        ))

    def update_settings(self, new_settings: VideoSettings) -> None:
        if self.settings != new_settings:
            self.settings = new_settings
            self.cache.clear()

    def _frame_hash(self, frame: np.ndarray) -> str:
        """Generate a quick hash from a downsampled frame."""
        small = cv2.resize(frame, (32, 24), interpolation=cv2.INTER_AREA)
        return hashlib.md5(small.tobytes()).hexdigest()[:16]

    def frame_to_ascii(self, frame: np.ndarray) -> str:
        # Generate cache key
        fhash = self._frame_hash(frame)
        shash = self._get_settings_hash()
        cache_key = f"{fhash}_{shash}"

        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Brightness / contrast adjustment
        if self.settings.brightness != 1.0 or self.settings.contrast != 1.0:
            alpha = self.settings.contrast
            beta = (self.settings.brightness - 1.0) * 128
            gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        # Calculate target dimensions (preserve aspect ratio)
        aspect_ratio = 0.45      # typical char width/height ratio
        target_w = self.settings.width
        target_h = int(gray.shape[0] * target_w / gray.shape[1] * aspect_ratio)
        target_h = max(1, target_h)

        resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)

        charset = CharsetManager.get_charset("detailed")
        max_idx = len(charset) - 1

        # Build ASCII art line by line
        lines = []
        for y in range(target_h):
            line_chars = []
            for x in range(target_w):
                intensity = resized[y, x] / 255.0
                idx = min(max_idx, int(intensity * max_idx))
                line_chars.append(charset[idx])
            lines.append(''.join(line_chars))

        ascii_art = '\n'.join(lines)
        self.cache.put(cache_key, ascii_art)
        return ascii_art


# ----------------------------------------------------------------------
# Background video processor (decoding, conversion, preloading)
# ----------------------------------------------------------------------
class VideoProcessor(threading.Thread):
    def __init__(self, video_path: str, converter: ASCIIConverter, buffer_size: int = 30):
        super().__init__(daemon=True, name="VideoProcessor")
        self.video_path = video_path
        self.converter = converter
        self.buffer = ThreadSafeFrameBuffer(max_size=buffer_size)

        # Control flags
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._seek_request = None          # (target_frame, seek_time)
        self._seek_lock = threading.Lock()
        self._preload_index = 0            # next frame to preload
        self._current_frame = 0
        self._state_lock = threading.RLock()

        # Video properties
        self.cap = None
        self.fps = 30.0
        self.total_frames = 0
        self.duration = 0.0
        self.error = None

        # Performance
        self.processing_times = deque(maxlen=30)

        # Communication queue (async to UI)
        self.frame_queue: queue.Queue = queue.Queue(maxsize=5)
        self.settings_queue: queue.Queue = queue.Queue()

        self._open_video()

    def _open_video(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.error = f"Cannot open video: {self.video_path}"
                return False

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30.0
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0.0

            logger.info(f"Loaded: {self.total_frames} frames, {self.fps:.2f} fps, {self.duration:.2f}s")
            return True
        except Exception as e:
            self.error = str(e)
            logger.error(f"Video open error: {e}")
            return False

    def run(self):
        if not self.cap or not self.cap.isOpened():
            self.frame_queue.put(("error", self.error or "Video not loaded"))
            return

        # Start preloader thread
        preloader = threading.Thread(target=self._preload_loop, daemon=True)
        preloader.start()

        frame_delay = 1.0 / self.fps
        last_frame_time = 0.0

        while not self._stop_event.is_set():
            # Apply pending settings
            try:
                new_settings = self.settings_queue.get_nowait()
                self.converter.update_settings(new_settings)
                self.buffer.clear()
            except queue.Empty:
                pass

            # Handle seek requests
            if self._seek_request is not None:
                with self._seek_lock:
                    target, _ = self._seek_request
                    if self.cap.set(cv2.CAP_PROP_POS_FRAMES, target):
                        with self._state_lock:
                            self._current_frame = target
                            self._preload_index = target
                        self.buffer.clear()
                        self._seek_request = None

            if not self._pause_event.is_set():
                now = time.time()
                if last_frame_time > 0:
                    expected = last_frame_time + frame_delay
                    if now < expected - 0.003:
                        time.sleep(max(0, expected - now - 0.002))

                frame_data = self.buffer.get(timeout=0.016)
                if frame_data:
                    frame_num, ascii_art, timestamp = frame_data
                    with self._state_lock:
                        self._current_frame = frame_num

                    conv_time = (time.time() - timestamp) * 1000
                    self.processing_times.append(conv_time)
                    avg_time = sum(self.processing_times) / len(self.processing_times)

                    info = {
                        'current': frame_num,
                        'total': self.total_frames,
                        'video_fps': self.fps,
                        'conversion_ms': conv_time,
                        'avg_conversion_ms': avg_time,
                        'is_playing': not self._pause_event.is_set(),
                        'buffer_stats': self.buffer.get_stats()
                    }
                    try:
                        self.frame_queue.put((ascii_art, info), timeout=0.1)
                    except queue.Full:
                        pass

                    last_frame_time = time.time()
                else:
                    time.sleep(0.005)
            else:
                time.sleep(0.01)

        self.cleanup()

    def _preload_loop(self):
        """Background loop: reads frames, converts them, fills buffer."""
        while not self._stop_event.is_set():
            # Manage buffer fullness
            stats = self.buffer.get_stats()
            if stats['buffered'] >= self.buffer.max_size:
                time.sleep(0.005)
                continue

            # Determine next frame index to read
            with self._seek_lock, self._state_lock:
                if self._seek_request is not None:
                    time.sleep(0.005)   # let seek settle
                    continue
                idx = self._preload_index
                if idx >= self.total_frames:
                    # End of video – wait for seek or stop
                    time.sleep(0.05)
                    continue

            # Read frame at current position
            with self._seek_lock:
                if self.cap is None or not self.cap.isOpened():
                    break
                # Ensure correct position
                cur_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if cur_pos != idx:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self.cap.read()

            if not ret:
                # Possibly end of file – mark as finished but stay alive for seeks
                with self._state_lock:
                    if self._preload_index >= self.total_frames - 1:
                        # EOF reached, just wait
                        time.sleep(0.1)
                    else:
                        # Read error, try to recover by re‑opening? For now, log
                        logger.warning(f"Frame read failed at index {idx}")
                        time.sleep(0.02)
                continue

            # Convert and enqueue
            timestamp = time.time()
            ascii_art = self.converter.frame_to_ascii(frame)
            self.buffer.put((idx, ascii_art, timestamp))

            with self._state_lock:
                self._preload_index = idx + 1

    # ------------------------------------------------------------------
    # Public control methods
    # ------------------------------------------------------------------
    def play(self):
        self._pause_event.clear()

    def pause(self):
        self._pause_event.set()

    def stop(self):
        self._pause_event.set()
        with self._seek_lock:
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                with self._state_lock:
                    self._current_frame = 0
                    self._preload_index = 0
        self.buffer.clear()

    def seek(self, frame_pos: int):
        frame_pos = max(0, min(frame_pos, self.total_frames - 1))
        with self._seek_lock:
            self._seek_request = (frame_pos, time.time())

    def seek_by_time(self, seconds: float):
        target = int(seconds * self.fps)
        self.seek(target)

    def seek_relative(self, delta_frames: int):
        with self._state_lock:
            new_pos = self._current_frame + delta_frames
        self.seek(new_pos)

    def update_settings(self, settings: VideoSettings):
        try:
            self.settings_queue.put(settings, timeout=0.1)
        except queue.Full:
            logger.warning("Settings queue full, skipping")

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Capture current frame without moving playhead."""
        with self._seek_lock:
            if self.cap and self.cap.isOpened():
                orig = self._current_frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, orig)
                ret, frame = self.cap.read()
                if ret:
                    return frame
        return None

    def get_state(self) -> Dict:
        with self._state_lock:
            return {
                'current_frame': self._current_frame,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'duration': self.duration,
                'error': self.error,
                'is_playing': not self._pause_event.is_set() and not self._stop_event.is_set(),
                'buffer_stats': self.buffer.get_stats()
            }

    def cleanup(self):
        self._stop_event.set()
        if self.cap:
            self.cap.release()
        logger.info("Video processor stopped")


# ----------------------------------------------------------------------
# Optimized Tkinter display (batched text insertion)
# ----------------------------------------------------------------------
class OptimizedDisplay:
    def __init__(self, parent, font_size: int = 10, enable_colors: bool = True):
        self.parent = parent
        self.font_size = font_size
        self.enable_colors = enable_colors
        self.current_ascii = ""
        self._update_lock = threading.Lock()
        self._update_scheduled = False

        # Canvas + scrollbars
        self.canvas = tk.Canvas(parent, bg="#000000", highlightthickness=0)
        v_scroll = tk.Scrollbar(parent, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scroll = tk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self.text_frame = tk.Frame(self.canvas, bg="#000000")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.text_frame, anchor="nw")

        # Text widget
        self.text_widget = tk.Text(
            self.text_frame,
            bg="#000000",
            fg="#00ff00",
            font=("Courier", font_size),
            wrap=tk.NONE,
            relief=tk.FLAT,
            highlightthickness=0
        )
        self.text_widget.pack()

        # Predefine color tags for 8‑step grayscale
        if self.enable_colors:
            for i in range(8):
                intensity = 255 - (i * 32)
                color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
                self.text_widget.tag_configure(f"g{i}", foreground=color)

        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.text_frame.bind('<Configure>', self._on_frame_configure)

        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _on_canvas_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def update_ascii(self, ascii_art: str):
        with self._update_lock:
            if ascii_art == self.current_ascii:
                return
            self.current_ascii = ascii_art
            if not self._update_scheduled:
                self._update_scheduled = True
                self.parent.after(10, self._perform_update)

    def _perform_update(self):
        with self._update_lock:
            self._update_scheduled = False
            self.text_widget.delete(1.0, tk.END)

            lines = self.current_ascii.split('\n')
            if not lines:
                return

            if self.enable_colors:
                # Insert each character with a tag based on its position (simple dither)
                for row, line in enumerate(lines):
                    if row > 0:
                        self.text_widget.insert(tk.END, '\n')
                    for col, ch in enumerate(line):
                        tag = f"g{(row + col) % 8}"
                        self.text_widget.insert(tk.END, ch, tag)
            else:
                # Monochrome – much faster
                self.text_widget.insert(tk.END, '\n'.join(lines))

            # Resize text widget to match content
            max_len = max(len(l) for l in lines) if lines else 0
            self.text_widget.configure(width=max_len, height=len(lines))
            self.text_frame.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def set_font_size(self, size: int):
        self.font_size = size
        self.text_widget.configure(font=("Courier", size))
        if self.current_ascii:
            self.update_ascii(self.current_ascii)

    def clear(self):
        self.text_widget.delete(1.0, tk.END)
        self.current_ascii = ""

    def destroy(self):
        self.canvas.destroy()


# ----------------------------------------------------------------------
# Settings persistence
# ----------------------------------------------------------------------
class ConfigManager:
    CONFIG_DIR = Path.home() / '.ascii_video_player'
    CONFIG_FILE = CONFIG_DIR / 'config.json'

    @classmethod
    def load(cls) -> VideoSettings:
        try:
            if cls.CONFIG_FILE.exists():
                with open(cls.CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    return VideoSettings(**data)
        except Exception as e:
            logger.error(f"Load config error: {e}")
        return VideoSettings()

    @classmethod
    def save(cls, settings: VideoSettings):
        try:
            cls.CONFIG_DIR.mkdir(exist_ok=True)
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(asdict(settings), f, indent=2)
            logger.info("Settings saved")
        except Exception as e:
            logger.error(f"Save config error: {e}")


# ----------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------
class ASCIIVideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("ASCII Video Player - High Performance")
        self.root.geometry("1200x800")
        atexit.register(self.cleanup)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Core components
        self.converter = ASCIIConverter()
        self.current_settings = ConfigManager.load()
        self.converter.update_settings(self.current_settings)
        self.video_processor: Optional[VideoProcessor] = None
        self.display: Optional[OptimizedDisplay] = None
        self.video_path: Optional[str] = None

        # UI update scheduling
        self.display_update_job = None
        self._settings_timer = None
        self._settings_lock = threading.Lock()

        # Colors
        self.bg_color = "#1e1e1e"
        self.fg_color = "#ffffff"
        self.accent = "#007acc"

        self.root.configure(bg=self.bg_color)
        self.setup_ui()
        self._load_settings_to_ui()
        self._bind_keys()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def setup_ui(self):
        main = tk.Frame(self.root, bg=self.bg_color)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control bar
        ctrl = tk.Frame(main, bg=self.bg_color)
        ctrl.pack(fill=tk.X, pady=(0, 10))

        # File section
        file_frame = tk.Frame(ctrl, bg=self.bg_color)
        file_frame.pack(side=tk.LEFT, padx=5)

        self.file_label = tk.Label(file_frame, text="No file selected", bg=self.bg_color, fg="#888888")
        self.file_label.pack(side=tk.LEFT, padx=5)

        self.btn_open = tk.Button(file_frame, text="Open Video", command=self.open_video,
                                  bg=self.accent, fg="white", relief=tk.FLAT, padx=10)
        self.btn_open.pack(side=tk.LEFT, padx=5)

        # Playback controls
        play_frame = tk.Frame(ctrl, bg=self.bg_color)
        play_frame.pack(side=tk.LEFT, padx=20)

        self.btn_play = tk.Button(play_frame, text="▶ Play", command=self.toggle_playback,
                                  bg="#2d2d2d", fg="white", relief=tk.FLAT, padx=10, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=2)

        self.btn_stop = tk.Button(play_frame, text="⏹ Stop", command=self.stop_playback,
                                  bg="#2d2d2d", fg="white", relief=tk.FLAT, padx=10, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        self.btn_back = tk.Button(play_frame, text="◀◀ 5s", command=lambda: self.seek_relative(-5),
                                  bg="#2d2d2d", fg="white", relief=tk.FLAT, padx=5, state=tk.DISABLED)
        self.btn_back.pack(side=tk.LEFT, padx=2)

        self.btn_forward = tk.Button(play_frame, text="5s ▶▶", command=lambda: self.seek_relative(5),
                                     bg="#2d2d2d", fg="white", relief=tk.FLAT, padx=5, state=tk.DISABLED)
        self.btn_forward.pack(side=tk.LEFT, padx=2)

        # Progress
        prog_frame = tk.Frame(play_frame, bg=self.bg_color)
        prog_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(prog_frame, text="Progress:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(prog_frame, variable=self.progress_var, length=300, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=5)
        self.position_label = tk.Label(prog_frame, text="0/0", bg=self.bg_color, fg=self.fg_color, width=12)
        self.position_label.pack(side=tk.LEFT, padx=5)

        # Settings panel (right side)
        sett_frame = tk.Frame(ctrl, bg=self.bg_color)
        sett_frame.pack(side=tk.RIGHT, padx=5)
        self._setup_settings_controls(sett_frame)

        # Display area
        disp_frame = tk.Frame(main, bg="#000000")
        disp_frame.pack(fill=tk.BOTH, expand=True)
        self.display = OptimizedDisplay(disp_frame, font_size=self.current_settings.font_size,
                                        enable_colors=self.current_settings.enable_colors)

        # Status bars
        self.status_bar = tk.Label(self.root, text="Ready – Open a video file",
                                   bg=self.accent, fg="white", anchor=tk.W, padx=10)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.info_bar = tk.Label(self.root, text="", bg=self.bg_color, fg="#888888",
                                 font=("Arial", 9), anchor=tk.W, padx=10)
        self.info_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_settings_controls(self, parent):
        """Sliders and spinboxes for live settings."""
        # Width
        tk.Label(parent, text="Width:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.width_var = tk.StringVar(value=str(self.current_settings.width))
        w_spin = tk.Spinbox(parent, from_=40, to=300, textvariable=self.width_var, width=5,
                            bg="#2d2d2d", fg=self.fg_color, relief=tk.FLAT,
                            command=self._schedule_settings_update)
        w_spin.pack(side=tk.LEFT, padx=2)
        self.width_var.trace_add('write', lambda *a: self._schedule_settings_update())

        # Font size
        tk.Label(parent, text="Font:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.font_var = tk.StringVar(value=str(self.current_settings.font_size))
        f_spin = tk.Spinbox(parent, from_=6, to=24, textvariable=self.font_var, width=5,
                            bg="#2d2d2d", fg=self.fg_color, relief=tk.FLAT,
                            command=self._schedule_settings_update)
        f_spin.pack(side=tk.LEFT, padx=2)
        self.font_var.trace_add('write', lambda *a: self._schedule_settings_update())

        # Brightness
        tk.Label(parent, text="Bright:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.bright_var = tk.DoubleVar(value=self.current_settings.brightness)
        b_scale = tk.Scale(parent, from_=0.0, to=2.0, resolution=0.05, orient=tk.HORIZONTAL,
                           length=100, variable=self.bright_var, bg=self.bg_color,
                           highlightthickness=0, command=lambda x: self._schedule_settings_update())
        b_scale.pack(side=tk.LEFT, padx=2)

        # Contrast
        tk.Label(parent, text="Contrast:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.contrast_var = tk.DoubleVar(value=self.current_settings.contrast)
        c_scale = tk.Scale(parent, from_=0.0, to=3.0, resolution=0.05, orient=tk.HORIZONTAL,
                           length=100, variable=self.contrast_var, bg=self.bg_color,
                           highlightthickness=0, command=lambda x: self._schedule_settings_update())
        c_scale.pack(side=tk.LEFT, padx=2)

        # Save / Reset
        self.btn_save = tk.Button(parent, text="Save", command=self.save_settings,
                                  bg="#2d2d2d", fg="white", relief=tk.FLAT, padx=8)
        self.btn_save.pack(side=tk.LEFT, padx=5)
        self.btn_reset = tk.Button(parent, text="Reset", command=self.reset_settings,
                                   bg="#2d2d2d", fg="white", relief=tk.FLAT, padx=5)
        self.btn_reset.pack(side=tk.LEFT, padx=2)

    def _bind_keys(self):
        self.root.bind('<space>', lambda e: self.toggle_playback())
        self.root.bind('<Left>', lambda e: self.seek_relative(-5))
        self.root.bind('<Right>', lambda e: self.seek_relative(5))
        self.root.bind('<Home>', lambda e: self.seek_absolute(0))
        self.root.bind('<End>', lambda e: self.seek_absolute(1.0))
        self.root.bind('<r>', lambda e: self.reset_settings())
        self.root.bind('<s>', lambda e: self.save_settings())

    # ------------------------------------------------------------------
    # Settings management
    # ------------------------------------------------------------------
    def _schedule_settings_update(self):
        with self._settings_lock:
            if self._settings_timer:
                self.root.after_cancel(self._settings_timer)
            self._settings_timer = self.root.after(150, self._apply_settings)

    def _apply_settings(self):
        with self._settings_lock:
            self._settings_timer = None
            try:
                new = VideoSettings(
                    width=int(self.width_var.get()),
                    font_size=int(self.font_var.get()),
                    brightness=float(self.bright_var.get()),
                    contrast=float(self.contrast_var.get()),
                    color_mode=self.current_settings.color_mode,
                    target_fps=self.current_settings.target_fps,
                    audio_sync=self.current_settings.audio_sync,
                    preload_frames=self.current_settings.preload_frames,
                    chars=self.current_settings.chars,
                    enable_colors=self.current_settings.enable_colors
                )
                if new == self.current_settings:
                    return
                self.current_settings = new
                self.converter.update_settings(new)
                self.display.set_font_size(new.font_size)
                if self.video_processor:
                    self.video_processor.update_settings(new)
                    # Refresh current frame if paused
                    if not self.video_processor.get_state()['is_playing']:
                        self.update_current_frame_display()
                self.status_bar.config(text=f"Settings: {new.width}px, font {new.font_size}")
                self.root.after(2000, lambda: self.status_bar.config(text="Ready"))
            except Exception as e:
                logger.error(f"Apply settings error: {e}")
                self.status_bar.config(text=f"Error: {e}")

    def _load_settings_to_ui(self):
        self.width_var.set(str(self.current_settings.width))
        self.font_var.set(str(self.current_settings.font_size))
        self.bright_var.set(self.current_settings.brightness)
        self.contrast_var.set(self.current_settings.contrast)

    def reset_settings(self):
        self.current_settings = VideoSettings()
        self._load_settings_to_ui()
        self._apply_settings()
        self.status_bar.config(text="Settings reset to defaults")

    def save_settings(self):
        ConfigManager.save(self.current_settings)
        self.status_bar.config(text="Settings saved")
        self.root.after(2000, lambda: self.status_bar.config(text="Ready"))

    # ------------------------------------------------------------------
    # Video control
    # ------------------------------------------------------------------
    def open_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("All files", "*.*")]
        )
        if not path:
            return

        self.stop_playback()
        if self.video_processor:
            self.video_processor.cleanup()
            self.video_processor = None

        # Create new processor (__init__ calls _open_video)
        self.video_processor = VideoProcessor(path, self.converter,
                                              buffer_size=self.current_settings.preload_frames)
        if self.video_processor.error:
            messagebox.showerror("Error", f"Cannot load video:\n{self.video_processor.error}")
            self.video_processor = None
            return

        self.video_path = path
        self.file_label.config(text=os.path.basename(path))

        # Enable controls
        self.btn_play.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_back.config(state=tk.NORMAL)
        self.btn_forward.config(state=tk.NORMAL)

        state = self.video_processor.get_state()
        self.status_bar.config(text=f"Loaded: {os.path.basename(path)}  |  {state['fps']:.1f} fps")
        self.video_processor.start()
        self.start_display_update()
        self.update_current_frame_display()

    def start_display_update(self):
        if self.display_update_job:
            self.root.after_cancel(self.display_update_job)
        self.update_display()

    def update_display(self):
        if not self.video_processor:
            self.display_update_job = self.root.after(100, self.update_display)
            return

        # Process up to 2 frames per cycle to keep UI responsive
        for _ in range(2):
            try:
                data = self.video_processor.frame_queue.get_nowait()
            except queue.Empty:
                break

            if data[0] == "error":
                self.status_bar.config(text=f"Error: {data[1]}")
                continue

            ascii_art, info = data
            if self.display:
                self.display.update_ascii(ascii_art)

            if info['is_playing']:
                status = (f"Frame {info['current']}/{info['total']}  |  "
                          f"{info['video_fps']:.1f}fps  |  "
                          f"conv {info['conversion_ms']:.1f}ms  |  "
                          f"buf {info['buffer_stats']['buffered']}")
                self.status_bar.config(text=status)

            self.position_label.config(text=f"{info['current']}/{info['total']}")
            progress = (info['current'] / info['total']) * 100 if info['total'] else 0
            self.progress_var.set(progress)

        self.display_update_job = self.root.after(16, self.update_display)

    def toggle_playback(self):
        if not self.video_processor:
            return
        state = self.video_processor.get_state()
        if state['is_playing']:
            self.video_processor.pause()
            self.btn_play.config(text="▶ Play")
            self.status_bar.config(text="Paused")
        else:
            self.video_processor.play()
            self.btn_play.config(text="⏸ Pause")
            self.status_bar.config(text="Playing")

    def stop_playback(self):
        if not self.video_processor:
            return
        self.video_processor.stop()
        self.btn_play.config(text="▶ Play")
        if self.display:
            self.display.clear()
        self.update_current_frame_display()
        self.progress_var.set(0)
        self.position_label.config(text="0/0")
        self.status_bar.config(text="Stopped")

    def seek_relative(self, seconds: float):
        if not self.video_processor:
            return
        state = self.video_processor.get_state()
        new_sec = (state['current_frame'] / state['fps']) + seconds
        self.video_processor.seek_by_time(new_sec)
        self.status_bar.config(text="Seeking...")

    def seek_absolute(self, fraction: float):
        """Seek to fraction (0 = start, 1 = end)."""
        if not self.video_processor:
            return
        state = self.video_processor.get_state()
        target_sec = fraction * state['duration']
        self.video_processor.seek_by_time(target_sec)
        self.status_bar.config(text="Seeking...")

    def update_current_frame_display(self):
        """Refresh current frame after settings change (useful when paused)."""
        if not self.video_processor:
            return
        if self.video_processor.get_state()['is_playing']:
            return
        frame = self.video_processor.get_current_frame()
        if frame is not None:
            ascii_art = self.converter.frame_to_ascii(frame)
            if self.display:
                self.display.update_ascii(ascii_art)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup(self):
        logger.info("Shutting down...")
        if self._settings_timer:
            try:
                self.root.after_cancel(self._settings_timer)
            except:
                pass
        if self.display_update_job:
            try:
                self.root.after_cancel(self.display_update_job)
            except:
                pass
        if self.video_processor:
            self.video_processor.cleanup()
            # Give it a moment to release resources
            time.sleep(0.05)

    def on_closing(self):
        self.cleanup()
        self.root.destroy()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    try:
        import cv2
        import numpy
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install opencv-python numpy")
        input("Press Enter to exit...")
        return

    root = tk.Tk()
    # Enable ANSI color support on Windows (not strictly needed for GUI, but nice)
    if sys.platform == "win32":
        os.system('color')
    app = ASCIIVideoPlayer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
