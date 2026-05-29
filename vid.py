#!/usr/bin/env python3
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
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from enum import Enum
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ascii_player.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ColorMode(Enum):
    """Color modes for ASCII output"""
    MONOCHROME = "monochrome"
    ANSI_GREYSCALE = "ansi_greyscale"
    ANSI_COLOR = "ansi_color"
    BLOCK_SHADES = "block_shades"


@dataclass
class VideoSettings:
    """User-configurable settings"""
    width: int = 100
    font_size: int = 10
    brightness: float = 1.0
    contrast: float = 1.0
    color_mode: str = "ansi_greyscale"
    target_fps: int = 30
    audio_sync: bool = True
    preload_frames: int = 30
    chars: str = " .:-=+*#%@"
    
    def __post_init__(self):
        self.width = max(40, min(300, self.width))
        self.font_size = max(6, min(24, self.font_size))
        self.brightness = max(0.0, min(2.0, self.brightness))
        self.contrast = max(0.0, min(3.0, self.contrast))
        self.target_fps = max(15, min(60, self.target_fps))
        self.preload_frames = max(10, min(100, self.preload_frames))


class CharsetManager:
    """Manages different character sets for ASCII rendering"""
    
    # Character sets from low to high density
    CHARSETS = {
        "default": " .:-=+*#%@",
        "detailed": " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
        "blocks": " ░▒▓█",
        "simple": " .oO0#",
        "matrix": " .,-:;+=x%#@"
    }
    
    # ANSI color gradients for greyscale simulation (Windows 10+ compatible)
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
        """Return ANSI color code based on pixel intensity"""
        if color_mode == "ansi_color":
            # Map intensity to hue (blue (0) -> red (255))
            hue = int(intensity * 200)
            return f"\033[38;5;{hue}m"
        elif color_mode == "ansi_greyscale":
            # Map intensity (0-1) to 8-bit greyscale
            idx = min(7, int(intensity * 8))
            return cls.ANSI_SHADES[idx]
        else:
            return ""  # Monochrome or tkinter will handle colors differently


class ThreadSafeFrameBuffer:
    """Thread-safe buffer for preloaded frames with frame dropping"""
    
    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.frame_counter = 0
        self.dropped_frames = 0
        self.last_frame_time = 0
        
    def put(self, frame_data: Tuple[int, str, float]):
        """Add frame to buffer, dropping oldest if full"""
        with self.lock:
            self.buffer.append(frame_data)
            self.frame_counter += 1
            
    def get(self, timeout: float = 0.016) -> Optional[Tuple[int, str, float]]:
        """Get next frame with proper timing"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                if self.buffer:
                    frame_num, ascii_art, timestamp = self.buffer.popleft()
                    
                    # Frame dropping logic for A/V sync
                    if self.last_frame_time > 0 and timestamp < self.last_frame_time:
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
    
    def clear(self):
        with self.lock:
            self.buffer.clear()
            self.dropped_frames = 0


class ASCIIConverter:
    """Converts video frames to ASCII with color support"""
    
    def __init__(self):
        self.settings = VideoSettings()
        self.cache: Dict[str, str] = {}
        self.cache_lock = threading.RLock()
        self._settings_hash = None
        
    def update_settings(self, new_settings: VideoSettings):
        """Update settings and clear cache"""
        with self.cache_lock:
            if self.settings != new_settings:
                self.settings = new_settings
                self.cache.clear()
                
    def _generate_cache_key(self, frame: np.ndarray) -> str:
        """Generate cache key from frame hash"""
        small_frame = cv2.resize(frame, (32, 24), interpolation=cv2.INTER_AREA)
        frame_hash = hashlib.md5(small_frame.tobytes()).hexdigest()[:16]
        settings_hash = hash((
            self.settings.width,
            self.settings.brightness,
            self.settings.contrast,
            self.settings.color_mode,
            self.settings.chars
        ))
        return f"{frame_hash}_{settings_hash}"
        
    def frame_to_ascii(self, frame: np.ndarray, use_colors: bool = True) -> str:
        """Convert frame to ASCII art with color codes"""
        cache_key = self._generate_cache_key(frame)
        
        # Check cache
        with self.cache_lock:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Apply brightness/contrast
        if self.settings.brightness != 1.0 or self.settings.contrast != 1.0:
            alpha = self.settings.contrast
            beta = (self.settings.brightness - 1.0) * 128
            gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        # Calculate target dimensions with aspect ratio correction
        aspect_ratio = 0.45  # Character aspect ratio (height/width)
        target_width = self.settings.width
        target_height = int(gray.shape[0] * target_width / gray.shape[1] * aspect_ratio)
        target_height = max(1, target_height)
        
        # Resize frame
        resized = cv2.resize(gray, (target_width, target_height), 
                            interpolation=cv2.INTER_AREA)
        
        # Get character set
        charset = CharsetManager.get_charset("detailed")
        charset_len = len(charset) - 1
        
        # Convert each pixel to character
        lines = []
        for y in range(target_height):
            line_chars = []
            for x in range(target_width):
                intensity = resized[y, x] / 255.0
                char_idx = min(charset_len, int(intensity * charset_len))
                char = charset[char_idx]
                line_chars.append(char)
            lines.append(''.join(line_chars))
        
        ascii_art = '\n'.join(lines)
        
        # Cache result
        with self.cache_lock:
            if len(self.cache) > 200:  # Limit cache size
                self.cache.clear()
            self.cache[cache_key] = ascii_art
            
        return ascii_art


class VideoProcessor(threading.Thread):
    """Background thread for video processing with frame preloading"""
    
    def __init__(self, video_path: str, converter: ASCIIConverter, buffer_size: int = 30):
        super().__init__(daemon=True, name="VideoProcessor")
        self.video_path = video_path
        self.converter = converter
        self.buffer = ThreadSafeFrameBuffer(max_size=buffer_size)
        
        # Control flags
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._seek_request = None
        self._seek_lock = threading.RLock()
        
        # Video properties
        self.cap = None
        self.fps = 30
        self.total_frames = 0
        self.current_frame = 0
        self.video_duration = 0
        self.error = None
        
        # Performance tracking
        self.processing_times = deque(maxlen=30)
        
        # Communication queues
        self.frame_queue: queue.Queue = queue.Queue(maxsize=5)
        self.settings_queue: queue.Queue = queue.Queue()
        
        self._open_video()
        
    def _open_video(self) -> bool:
        """Open video file and read properties"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.error = f"Failed to open video: {self.video_path}"
                return False
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30
            
            self.video_duration = self.total_frames / self.fps if self.fps > 0 else 0
            
            logger.info(f"Video loaded: {self.total_frames} frames, {self.fps:.2f} FPS, {self.video_duration:.2f}s")
            return True
            
        except Exception as e:
            self.error = str(e)
            logger.error(f"Error opening video: {e}")
            return False
    
    def run(self):
        """Main processing loop with preloading"""
        if not self.cap or not self.cap.isOpened():
            self.frame_queue.put(("error", self.error or "Video not loaded"))
            return
        
        # Start preload thread
        preload_thread = threading.Thread(target=self._preload_frames, daemon=True)
        preload_thread.start()
        
        frame_delay = 1.0 / self.fps if self.fps > 0 else 0.033
        last_frame_time = 0
        
        while not self._stop_event.is_set():
            # Check for settings updates
            try:
                new_settings = self.settings_queue.get_nowait()
                self.converter.update_settings(new_settings)
                self.buffer.clear()
            except queue.Empty:
                pass
            
            # Handle seek requests
            if self._seek_request is not None:
                with self._seek_lock:
                    if self.cap:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_request)
                        self.current_frame = self._seek_request
                        self._seek_request = None
                        self.buffer.clear()
            
            if not self._pause_event.is_set():
                current_time = time.time()
                
                # Frame timing for A/V sync
                if last_frame_time > 0:
                    expected_time = last_frame_time + frame_delay
                    if current_time < expected_time - 0.005:
                        time.sleep(max(0, expected_time - current_time - 0.001))
                
                # Get next frame from buffer
                frame_data = self.buffer.get(timeout=0.016)
                
                if frame_data:
                    frame_num, ascii_art, timestamp = frame_data
                    
                    with self._seek_lock:
                        self.current_frame = frame_num
                    
                    # Calculate conversion time
                    conversion_time = (time.time() - timestamp) * 1000
                    self.processing_times.append(conversion_time)
                    avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
                    
                    # Package frame info
                    frame_info = {
                        'current': frame_num,
                        'total': self.total_frames,
                        'video_fps': self.fps,
                        'conversion_ms': conversion_time,
                        'avg_conversion_ms': avg_time,
                        'is_playing': not self._pause_event.is_set(),
                        'buffer_stats': self.buffer.get_stats()
                    }
                    
                    # Send to UI
                    try:
                        self.frame_queue.put((ascii_art, frame_info), timeout=0.1)
                    except queue.Full:
                        pass
                    
                    last_frame_time = time.time()
                else:
                    time.sleep(0.005)
            else:
                time.sleep(0.01)
        
        self.cleanup()
    
    def _preload_frames(self):
        """Background thread for preloading and converting frames"""
        frame_index = self.current_frame
        
        while not self._stop_event.is_set() and frame_index < self.total_frames:
            # Check if buffer has space
            stats = self.buffer.get_stats()
            
            if stats['buffered'] >= self.buffer.max_size:
                time.sleep(0.005)
                continue
            
            # Read frame with proper locking
            with self._seek_lock:
                if not self.cap or not self.cap.isOpened():
                    break
                
                # Ensure we're at the right position
                current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current_pos != frame_index:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                
                ret, frame = self.cap.read()
            
            if not ret:
                if frame_index >= self.total_frames - 1:
                    # End of video
                    time.sleep(0.1)
                    continue
                break
            
            # Convert frame
            timestamp = time.time()
            ascii_art = self.converter.frame_to_ascii(frame)
            
            # Add to buffer
            self.buffer.put((frame_index, ascii_art, timestamp))
            frame_index += 1
    
    def play(self):
        """Resume playback"""
        self._pause_event.clear()
    
    def pause(self):
        """Pause playback"""
        self._pause_event.set()
    
    def stop(self):
        """Stop playback and reset"""
        self._pause_event.set()
        with self._seek_lock:
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
        self.buffer.clear()
    
    def seek(self, frame_position: int):
        """Seek to specific frame"""
        with self._seek_lock:
            self._seek_request = max(0, min(frame_position, self.total_frames - 1))
            self.buffer.clear()
    
    def seek_relative(self, delta: int):
        """Seek relative to current position"""
        with self._seek_lock:
            new_pos = self.current_frame + delta
            self.seek(new_pos)
    
    def update_settings(self, settings: VideoSettings):
        """Update converter settings"""
        try:
            self.settings_queue.put(settings, timeout=0.1)
        except queue.Full:
            logger.warning("Settings queue full, skipping update")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Capture current frame without advancing position"""
        with self._seek_lock:
            if self.cap and self.cap.isOpened():
                original_pos = self.current_frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
                ret, frame = self.cap.read()
                if ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
                    return frame
        return None
    
    def get_state(self) -> Dict:
        with self._seek_lock:
            return {
                'current_frame': self.current_frame,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'duration': self.video_duration,
                'error': self.error,
                'is_playing': not self._pause_event.is_set() and not self._stop_event.is_set(),
                'buffer_stats': self.buffer.get_stats()
            }
    
    def cleanup(self):
        """Release resources"""
        self._stop_event.set()
        if self.cap:
            self.cap.release()
        logger.info("Video processor cleaned up")


class OptimizedDisplay:
    """Tkinter widget for ASCII art rendering with color support"""
    
    def __init__(self, parent, font_size: int = 10):
        self.parent = parent
        self.font_size = font_size
        self.current_ascii = ""
        self._update_lock = threading.Lock()
        self._update_scheduled = False
        
        # Canvas for scrollable display
        self.canvas = tk.Canvas(parent, bg="#000000", highlightthickness=0)
        self.v_scrollbar = tk.Scrollbar(parent, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scrollbar = tk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        
        self.text_frame = tk.Frame(self.canvas, bg="#000000")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.text_frame, anchor="nw")
        
        # Text widget with color support
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
        
        # Configure tags for different colors
        self._setup_color_tags()
        
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.text_frame.bind('<Configure>', self._on_frame_configure)
        
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def _setup_color_tags(self):
        """Setup color tags for ANSI color simulation"""
        # Greyscale tags
        for i in range(8):
            intensity = 255 - (i * 32)
            color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
            self.text_widget.tag_configure(f"grey{i}", foreground=color)
        
        # Basic colors
        colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan']
        for color in colors:
            self.text_widget.tag_configure(color, foreground=color)
    
    def _on_canvas_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def update_ascii(self, ascii_art: str):
        """Schedule UI update with throttling"""
        with self._update_lock:
            if ascii_art == self.current_ascii:
                return
            
            self.current_ascii = ascii_art
            
            if not self._update_scheduled:
                self._update_scheduled = True
                self.parent.after(16, self._perform_update)
    
    def _perform_update(self):
        """Actually update the UI"""
        with self._update_lock:
            self._update_scheduled = False
            
            self.text_widget.delete(1.0, tk.END)
            
            # Insert ASCII art with optional color tags
            lines = self.current_ascii.split('\n')
            for i, line in enumerate(lines):
                if i > 0:
                    self.text_widget.insert(tk.END, '\n')
                
                # Apply random colors for visual interest (or keep monochrome)
                for j, char in enumerate(line):
                    color_idx = (i + j) % 8
                    self.text_widget.insert(tk.END, char, f"grey{color_idx}")
            
            # Resize widget
            max_line_length = max(len(line) for line in lines) if lines else 0
            self.text_widget.configure(width=max_line_length, height=len(lines))
            
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


class ConfigManager:
    """Persist user settings"""
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
            logger.error(f"Error loading config: {e}")
        return VideoSettings()
    
    @classmethod
    def save(cls, settings: VideoSettings):
        try:
            cls.CONFIG_DIR.mkdir(exist_ok=True)
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(settings.__dict__, f, indent=2)
            logger.info("Settings saved")
        except Exception as e:
            logger.error(f"Error saving config: {e}")


class ASCIIVideoPlayer:
    """Main application class"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("ASCII Video Player - Windows Compatible")
        self.root.geometry("1200x800")
        
        atexit.register(self.cleanup)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.converter = ASCIIConverter()
        self.video_processor: Optional[VideoProcessor] = None
        self.display: Optional[OptimizedDisplay] = None
        self.current_settings = ConfigManager.load()
        self.converter.update_settings(self.current_settings)
        
        self.video_path: Optional[str] = None
        self.display_update_job: Optional[str] = None
        self._settings_update_timer = None
        self._settings_update_lock = threading.Lock()
        
        # UI Colors
        self.bg_color = "#1e1e1e"
        self.fg_color = "#ffffff"
        self.accent_color = "#007acc"
        
        self.root.configure(bg=self.bg_color)
        self.setup_ui()
        self._load_settings_to_ui()
    
    def setup_ui(self):
        """Build UI elements"""
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg=self.bg_color)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File controls
        file_frame = tk.Frame(control_frame, bg=self.bg_color)
        file_frame.pack(side=tk.LEFT, padx=5)
        
        self.file_label = tk.Label(file_frame, text="No file selected", 
                                   bg=self.bg_color, fg="#888888")
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        self.btn_open = tk.Button(file_frame, text="Open Video", command=self.open_video,
                                  bg=self.accent_color, fg="white", relief=tk.FLAT, padx=10)
        self.btn_open.pack(side=tk.LEFT, padx=5)
        
        # Playback controls
        playback_frame = tk.Frame(control_frame, bg=self.bg_color)
        playback_frame.pack(side=tk.LEFT, padx=20)
        
        self.btn_play = tk.Button(playback_frame, text="▶ Play", command=self.toggle_playback,
                                  bg="#2d2d2d", fg="white", relief=tk.FLAT, padx=10, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=2)
        
        self.btn_stop = tk.Button(playback_frame, text="⏹ Stop", command=self.stop_playback,
                                  bg="#2d2d2d", fg="white", relief=tk.FLAT, padx=10, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=2)
        
        self.btn_seek_back = tk.Button(playback_frame, text="◀◀ 5s", 
                                       command=lambda: self.seek_relative(-150),
                                       bg="#2d2d2d", fg="white", relief=tk.FLAT, padx=5, state=tk.DISABLED)
        self.btn_seek_back.pack(side=tk.LEFT, padx=2)
        
        self.btn_seek_forward = tk.Button(playback_frame, text="5s ▶▶",
                                          command=lambda: self.seek_relative(150),
                                          bg="#2d2d2d", fg="white", relief=tk.FLAT, padx=5, state=tk.DISABLED)
        self.btn_seek_forward.pack(side=tk.LEFT, padx=2)
        
        # Progress bar
        progress_frame = tk.Frame(playback_frame, bg=self.bg_color)
        progress_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(progress_frame, text="Progress:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, length=300, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=5)
        self.position_label = tk.Label(progress_frame, text="0/0", bg=self.bg_color, fg=self.fg_color, width=12)
        self.position_label.pack(side=tk.LEFT, padx=5)
        
        # Settings panel
        settings_frame = tk.Frame(control_frame, bg=self.bg_color)
        settings_frame.pack(side=tk.RIGHT, padx=5)
        
        self._setup_settings_controls(settings_frame)
        
        # Display area
        display_frame = tk.Frame(main_frame, bg="#000000")
        display_frame.pack(fill=tk.BOTH, expand=True)
        self.display = OptimizedDisplay(display_frame, font_size=self.current_settings.font_size)
        
        # Status bars
        self.status_bar = tk.Label(self.root, text="Ready - Click 'Open Video' to start",
                                   bg=self.accent_color, fg="white", anchor=tk.W, padx=10)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.info_label = tk.Label(self.root, text="", bg=self.bg_color, fg="#888888",
                                   font=("Arial", 9), anchor=tk.W, padx=10)
        self.info_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_settings_controls(self, parent):
        """Setup settings controls"""
        # Width
        tk.Label(parent, text="Width:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.width_var = tk.StringVar(value=str(self.current_settings.width))
        width_spinbox = tk.Spinbox(parent, from_=40, to=300, textvariable=self.width_var, width=5,
                                   bg="#2d2d2d", fg=self.fg_color, relief=tk.FLAT,
                                   command=self._schedule_settings_update)
        width_spinbox.pack(side=tk.LEFT, padx=2)
        self.width_var.trace_add('write', lambda *args: self._schedule_settings_update())
        
        # Font size
        tk.Label(parent, text="Font:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.font_var = tk.StringVar(value=str(self.current_settings.font_size))
        font_spinbox = tk.Spinbox(parent, from_=6, to=24, textvariable=self.font_var, width=5,
                                  bg="#2d2d2d", fg=self.fg_color, relief=tk.FLAT,
                                  command=self._schedule_settings_update)
        font_spinbox.pack(side=tk.LEFT, padx=2)
        self.font_var.trace_add('write', lambda *args: self._schedule_settings_update())
        
        # Brightness
        tk.Label(parent, text="Bright:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.brightness_var = tk.DoubleVar(value=self.current_settings.brightness)
        brightness_scale = tk.Scale(parent, from_=0.0, to=2.0, resolution=0.05, orient=tk.HORIZONTAL,
                                    length=100, variable=self.brightness_var, bg=self.bg_color,
                                    fg=self.fg_color, highlightthickness=0,
                                    command=lambda x: self._schedule_settings_update())
        brightness_scale.pack(side=tk.LEFT, padx=2)
        
        # Contrast
        tk.Label(parent, text="Contrast:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.contrast_var = tk.DoubleVar(value=self.current_settings.contrast)
        contrast_scale = tk.Scale(parent, from_=0.0, to=3.0, resolution=0.05, orient=tk.HORIZONTAL,
                                  length=100, variable=self.contrast_var, bg=self.bg_color,
                                  fg=self.fg_color, highlightthickness=0,
                                  command=lambda x: self._schedule_settings_update())
        contrast_scale.pack(side=tk.LEFT, padx=2)
        
        # Save button
        self.btn_save = tk.Button(parent, text="Save", command=self.save_settings,
                                  bg="#2d2d2d", fg="white", relief=tk.FLAT, padx=10)
        self.btn_save.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        self.btn_reset = tk.Button(parent, text="Reset", command=self.reset_settings,
                                   bg="#2d2d2d", fg="white", relief=tk.FLAT, padx=10)
        self.btn_reset.pack(side=tk.LEFT, padx=2)
    
    def _schedule_settings_update(self):
        """Debounce settings updates"""
        with self._settings_update_lock:
            if self._settings_update_timer:
                self.root.after_cancel(self._settings_update_timer)
            self._settings_update_timer = self.root.after(100, self._apply_settings)
    
    def _apply_settings(self):
        """Apply new settings"""
        with self._settings_update_lock:
            self._settings_update_timer = None
            
            try:
                new_settings = VideoSettings(
                    width=int(self.width_var.get()),
                    font_size=int(self.font_var.get()),
                    brightness=float(self.brightness_var.get()),
                    contrast=float(self.contrast_var.get()),
                    color_mode=self.current_settings.color_mode,
                    target_fps=self.current_settings.target_fps,
                    audio_sync=self.current_settings.audio_sync,
                    preload_frames=self.current_settings.preload_frames,
                    chars=self.current_settings.chars
                )
                
                if new_settings == self.current_settings:
                    return
                
                self.current_settings = new_settings
                self.converter.update_settings(new_settings)
                
                if self.display:
                    self.display.set_font_size(new_settings.font_size)
                
                if self.video_processor:
                    self.video_processor.update_settings(new_settings)
                    
                    # Refresh current frame if paused
                    state = self.video_processor.get_state()
                    if not state['is_playing']:
                        self.update_current_frame_display()
                
                self.status_bar.config(text=f"Settings applied: {new_settings.width}x{new_settings.font_size}")
                self.root.after(2000, lambda: self.status_bar.config(text="Ready"))
                
            except Exception as e:
                logger.error(f"Error applying settings: {e}")
                self.status_bar.config(text=f"Error: {e}")
    
    def _load_settings_to_ui(self):
        """Load settings to UI controls"""
        self.width_var.set(str(self.current_settings.width))
        self.font_var.set(str(self.current_settings.font_size))
        self.brightness_var.set(self.current_settings.brightness)
        self.contrast_var.set(self.current_settings.contrast)
    
    def reset_settings(self):
        """Reset to default settings"""
        self.current_settings = VideoSettings()
        self._load_settings_to_ui()
        self._apply_settings()
        self.status_bar.config(text="Settings reset to defaults")
    
    def save_settings(self):
        """Save current settings"""
        ConfigManager.save(self.current_settings)
        self.status_bar.config(text="Settings saved")
        self.root.after(2000, lambda: self.status_bar.config(text="Ready"))
    
    def open_video(self):
        """Open video file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.m4v"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        # Clean up previous video
        self.stop_playback()
        if self.video_processor:
            self.video_processor.cleanup()
            self.video_processor = None
        
        # Create new processor
        self.video_processor = VideoProcessor(file_path, self.converter, 
                                              buffer_size=self.current_settings.preload_frames)
        
        if not self.video_processor._open_video():
            messagebox.showerror("Error", f"Failed to load video:\n{self.video_processor.error}")
            self.video_processor = None
            return
        
        self.video_path = file_path
        self.file_label.config(text=os.path.basename(file_path))
        
        # Enable controls
        self.btn_play.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_seek_back.config(state=tk.NORMAL)
        self.btn_seek_forward.config(state=tk.NORMAL)
        
        state = self.video_processor.get_state()
        self.status_bar.config(text=f"Loaded: {os.path.basename(file_path)} | FPS: {state['fps']:.1f}")
        
        # Start processing
        self.video_processor.start()
        self.start_display_update()
        self.update_current_frame_display()
    
    def start_display_update(self):
        """Start display update loop"""
        if self.display_update_job:
            self.root.after_cancel(self.display_update_job)
        self.update_display()
    
    def update_display(self):
        """Main display update loop"""
        if not self.video_processor:
            self.display_update_job = self.root.after(100, self.update_display)
            return
        
        try:
            frames_processed = 0
            while frames_processed < 3:
                try:
                    data = self.video_processor.frame_queue.get_nowait()
                except queue.Empty:
                    break
                
                if data[0] == "error":
                    self.status_bar.config(text=f"Error: {data[1]}")
                    break
                else:
                    ascii_art, frame_info = data
                    
                    if self.display:
                        self.display.update_ascii(ascii_art)
                    
                    if frame_info['is_playing']:
                        status_text = f"Frame: {frame_info['current']}/{frame_info['total']} | "
                        status_text += f"FPS: {frame_info['video_fps']:.1f} | "
                        status_text += f"Conv: {frame_info['conversion_ms']:.1f}ms | "
                        status_text += f"Buffer: {frame_info['buffer_stats']['buffered']}"
                        self.status_bar.config(text=status_text)
                    
                    self.position_label.config(text=f"{frame_info['current']}/{frame_info['total']}")
                    progress = (frame_info['current'] / frame_info['total']) * 100 if frame_info['total'] > 0 else 0
                    self.progress_var.set(progress)
                    
                    frames_processed += 1
        except Exception as e:
            logger.error(f"Display update error: {e}")
        
        self.display_update_job = self.root.after(16, self.update_display)
    
    def toggle_playback(self):
        """Toggle play/pause"""
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
        """Stop playback"""
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
    
    def seek_relative(self, delta: int):
        """Seek relative to current position"""
        if not self.video_processor:
            return
        
        self.video_processor.seek_relative(delta)
        self.status_bar.config(text="Seeking...")
    
    def update_current_frame_display(self):
        """Refresh current frame display (used after settings changes)"""
        if not self.video_processor:
            return
        
        state = self.video_processor.get_state()
        if state['is_playing']:
            return
        
        try:
            frame = self.video_processor.get_current_frame()
            if frame is not None:
                ascii_art = self.converter.frame_to_ascii(frame)
                if self.display:
                    self.display.update_ascii(ascii_art)
        except Exception as e:
            logger.error(f"Error updating current frame: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up application...")
        
        if self._settings_update_timer:
            try:
                self.root.after_cancel(self._settings_update_timer)
            except:
                pass
        
        if self.display_update_job:
            try:
                self.root.after_cancel(self.display_update_job)
            except:
                pass
        
        if self.video_processor:
            self.video_processor.cleanup()
        
        logger.info("Cleanup complete")
    
    def on_closing(self):
        """Handle window close"""
        self.cleanup()
        self.root.destroy()


def main():
    """Entry point"""
    try:
        # Check dependencies
        import cv2
        import numpy
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install: pip install opencv-python numpy")
        input("Press Enter to exit...")
        return
    
    root = tk.Tk()
    app = ASCIIVideoPlayer(root)
    
    # Enable ANSI color support on Windows 10+
    if sys.platform == "win32":
        os.system('color')
    
    root.mainloop()


if __name__ == "__main__":
    main()
