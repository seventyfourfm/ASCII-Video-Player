
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import queue
import time
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlaybackState(Enum):
    """Playback states for the video player"""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


@dataclass
class VideoMetadata:
    """Container for video metadata"""
    fps: float
    total_frames: int
    duration: float
    width: int
    height: int
    codec: str


class ASCIIConverter:
    """Handles all ASCII art conversion logic"""
    
    DEFAULT_CHARS = " .:-=+*#%@"
    MIN_WIDTH = 10
    MAX_WIDTH = 300
    MIN_FONT_SIZE = 6
    MAX_FONT_SIZE = 30
    DEFAULT_ASPECT_RATIO = 0.55
    
    def __init__(self, width: int = 100, chars: str = None, aspect_ratio: float = None):
        """
        Initialize ASCII converter
        
        Args:
            width: ASCII art width in characters
            chars: Character set for brightness mapping
            aspect_ratio: Character aspect ratio (height/width)
        """
        self.width = max(self.MIN_WIDTH, min(width, self.MAX_WIDTH))
        self.chars = chars or self.DEFAULT_CHARS
        self.aspect_ratio = aspect_ratio or self.DEFAULT_ASPECT_RATIO
        
        # Pre-calculate character mapping for speed
        self._char_array = np.array(list(self.chars))
        
    def set_width(self, width: int) -> None:
        """Set ASCII width with validation"""
        self.width = max(self.MIN_WIDTH, min(width, self.MAX_WIDTH))
    
    def frame_to_ascii(self, frame: np.ndarray) -> str:
        """
        Convert video frame to ASCII art
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            ASCII art string
        """
        if frame is None or frame.size == 0:
            return ""
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate height maintaining aspect ratio
            height = max(1, int(gray.shape[0] * self.width / gray.shape[1] * self.aspect_ratio))
            height = max(1, min(height, 1000))  # Cap height for performance
            
            # Resize image
            resized = cv2.resize(gray, (self.width, height), interpolation=cv2.INTER_LINEAR)
            
            # Vectorized pixel-to-character mapping
            indices = (resized / 255 * (len(self.chars) - 1)).astype(np.uint8)
            
            # Convert to ASCII using vectorized lookup
            ascii_rows = [''.join(self._char_array[row]) for row in indices]
            
            return '\n'.join(ascii_rows)
            
        except Exception as e:
            logger.error(f"Error converting frame to ASCII: {e}")
            return ""


class VideoPlayer:
    """Handles video loading, playback, and frame management"""
    
    MAX_FPS_CAP = 30  # Cap rendering at 30 FPS maximum
    FRAME_SKIP_THRESHOLD = 60  # Skip frames when FPS > this value
    
    def __init__(self):
        """Initialize video player"""
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[str] = None
        self.metadata: Optional[VideoMetadata] = None
        self._lock = threading.Lock()
        
    def load_video(self, path: str) -> bool:
        """
        Load video file
        
        Args:
            path: Path to video file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Release existing video
            self.release()
            
            # Validate file exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"Video file not found: {path}")
            
            # Open video
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {path}")
            
            # Extract metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 24.0  # Default fallback
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Get codec info
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]) if fourcc else "Unknown"
            
            with self._lock:
                self.cap = cap
                self.video_path = path
                self.metadata = VideoMetadata(
                    fps=fps,
                    total_frames=total_frames,
                    duration=duration,
                    width=width,
                    height=height,
                    codec=codec
                )
            
            logger.info(f"Loaded video: {path} - {width}x{height}, {fps:.2f}fps, {total_frames} frames")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load video: {e}")
            self.release()
            raise
    
    def get_frame(self, frame_num: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get a specific frame from the video
        
        Args:
            frame_num: Frame number to retrieve (None for next frame)
            
        Returns:
            Frame as numpy array or None if failed
        """
        with self._lock:
            if self.cap is None or not self.cap.isOpened():
                return None
            
            try:
                if frame_num is not None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                
                ret, frame = self.cap.read()
                if not ret:
                    # Loop back to beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    
                return frame if ret else None
                
            except Exception as e:
                logger.error(f"Error getting frame: {e}")
                return None
    
    def get_current_frame_num(self) -> int:
        """Get current frame number"""
        with self._lock:
            if self.cap and self.cap.isOpened():
                return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            return 0
    
    def set_frame_position(self, position: float) -> bool:
        """
        Set playback position (0.0 to 1.0)
        
        Args:
            position: Relative position (0.0 to 1.0)
            
        Returns:
            True if successful
        """
        if not self.metadata:
            return False
            
        frame_num = int(position * self.metadata.total_frames)
        with self._lock:
            if self.cap and self.cap.isOpened():
                return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        return False
    
    def release(self) -> None:
        """Release video capture resources"""
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.video_path = None
            self.metadata = None
    
    def is_loaded(self) -> bool:
        """Check if video is loaded"""
        return self.cap is not None and self.cap.isOpened()


class SettingsManager:
    """Manages application settings persistence"""
    
    SETTINGS_FILE = "ascii_video_settings.json"
    
    DEFAULT_SETTINGS = {
        "ascii_width": 100,
        "font_size": 10,
        "charset": " .:-=+*#%@",
        "aspect_ratio": 0.55,
        "theme": "dark",
        "recent_files": []
    }
    
    def __init__(self):
        """Initialize settings manager"""
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.load()
    
    def load(self) -> None:
        """Load settings from file"""
        try:
            if os.path.exists(self.SETTINGS_FILE):
                with open(self.SETTINGS_FILE, 'r') as f:
                    loaded = json.load(f)
                    self.settings.update(loaded)
                    logger.info(f"Loaded settings from {self.SETTINGS_FILE}")
        except Exception as e:
            logger.warning(f"Could not load settings: {e}")
    
    def save(self) -> None:
        """Save settings to file"""
        try:
            with open(self.SETTINGS_FILE, 'w') as f:
                json.dump(self.settings, f, indent=2)
                logger.info(f"Saved settings to {self.SETTINGS_FILE}")
        except Exception as e:
            logger.error(f"Could not save settings: {e}")
    
    def get(self, key: str, default=None):
        """Get setting value"""
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set setting value"""
        self.settings[key] = value
        self.save()
    
    def add_recent_file(self, filepath: str) -> None:
        """Add file to recent files list"""
        recent = self.settings.get("recent_files", [])
        if filepath in recent:
            recent.remove(filepath)
        recent.insert(0, filepath)
        self.settings["recent_files"] = recent[:10]  # Keep last 10
        self.save()


class ASCIIVideoGUI:
    """Main GUI application for ASCII Video Player"""
    
    # Theme colors
    THEMES = {
        "dark": {
            "bg": "#1e1e1e",
            "fg": "#ffffff",
            "accent": "#007acc",
            "button_bg": "#2d2d2d",
            "canvas_bg": "#000000",
            "status_bg": "#007acc"
        },
        "matrix": {
            "bg": "#000000",
            "fg": "#00ff00",
            "accent": "#008800",
            "button_bg": "#003300",
            "canvas_bg": "#000000",
            "status_bg": "#008800"
        },
        "amber": {
            "bg": "#000000",
            "fg": "#ffb000",
            "accent": "#cc8800",
            "button_bg": "#332200",
            "canvas_bg": "#000000",
            "status_bg": "#cc8800"
        }
    }
    
    def __init__(self, root: tk.Tk):
        """Initialize the ASCII Video Player GUI"""
        self.root = root
        self.root.title("ASCII Video Player")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize components
        self.video_player = VideoPlayer()
        self.ascii_converter = ASCIIConverter()
        self.settings_manager = SettingsManager()
        
        # State variables
        self.playback_state = PlaybackState.STOPPED
        self.update_job: Optional[str] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self.processing_thread: Optional[threading.Thread] = None
        self.render_fps = 0
        self.last_render_time = 0
        
        # Load saved settings
        self.load_settings()
        
        # Setup UI
        self.setup_ui()
        
        # Start frame processor
        self.start_frame_processor()
        
        # Bind keyboard shortcuts
        self.setup_keyboard_shortcuts()
        
    def load_settings(self) -> None:
        """Load settings from file"""
        try:
            self.ascii_converter.width = self.settings_manager.get("ascii_width", 100)
            self.ascii_converter.chars = self.settings_manager.get("charset", " .:-=+*#%@")
            self.ascii_converter.aspect_ratio = self.settings_manager.get("aspect_ratio", 0.55)
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
    
    def setup_ui(self) -> None:
        """Build the complete user interface"""
        # Apply theme
        theme = self.settings_manager.get("theme", "dark")
        self.colors = self.THEMES.get(theme, self.THEMES["dark"])
        self.root.configure(bg=self.colors["bg"])
        
        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors["bg"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        self.create_control_panel(main_frame)
        
        # Display area
        self.create_display_area(main_frame)
        
        # Progress bar
        self.create_progress_bar(main_frame)
        
        # Status bar
        self.create_status_bar()
        
    def create_control_panel(self, parent: tk.Frame) -> None:
        """Create the control panel with all buttons and settings"""
        control_frame = tk.Frame(parent, bg=self.colors["bg"])
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File section
        file_frame = tk.Frame(control_frame, bg=self.colors["bg"])
        file_frame.pack(side=tk.LEFT, padx=5)
        
        self.file_label = tk.Label(
            file_frame, text="No file selected",
            bg=self.colors["bg"], fg="#888888", font=("Arial", 10)
        )
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        self.btn_open = tk.Button(
            file_frame, text="Open Video", command=self.open_video,
            bg=self.colors["accent"], fg="white", relief=tk.FLAT, padx=10
        )
        self.btn_open.pack(side=tk.LEFT, padx=5)
        
        # Recent files dropdown
        self.recent_var = tk.StringVar()
        self.recent_menu = tk.OptionMenu(
            file_frame, self.recent_var, "", *self.settings_manager.get("recent_files", [])
        )
        self.recent_menu.config(bg=self.colors["button_bg"], fg=self.colors["fg"], relief=tk.FLAT)
        self.recent_var.trace('w', lambda *args: self.load_recent_file())
        self.recent_menu.pack(side=tk.LEFT, padx=5)
        
        # Playback section
        playback_frame = tk.Frame(control_frame, bg=self.colors["bg"])
        playback_frame.pack(side=tk.LEFT, padx=20)
        
        self.btn_play = tk.Button(
            playback_frame, text="Play", command=self.toggle_playback,
            bg=self.colors["button_bg"], fg=self.colors["fg"],
            relief=tk.FLAT, padx=10, state=tk.DISABLED
        )
        self.btn_play.pack(side=tk.LEFT, padx=2)
        
        self.btn_stop = tk.Button(
            playback_frame, text="Stop", command=self.stop_playback,
            bg=self.colors["button_bg"], fg=self.colors["fg"],
            relief=tk.FLAT, padx=10, state=tk.DISABLED
        )
        self.btn_stop.pack(side=tk.LEFT, padx=2)
        
        # Settings section
        settings_frame = tk.Frame(control_frame, bg=self.colors["bg"])
        settings_frame.pack(side=tk.RIGHT, padx=5)
        
        # Width control
        tk.Label(settings_frame, text="Width:", bg=self.colors["bg"], fg=self.colors["fg"]).pack(side=tk.LEFT, padx=2)
        self.width_var = tk.StringVar(value=str(self.ascii_converter.width))
        width_spin = tk.Spinbox(
            settings_frame, from_=10, to=300, textvariable=self.width_var, width=5,
            bg=self.colors["button_bg"], fg=self.colors["fg"], relief=tk.FLAT
        )
        width_spin.pack(side=tk.LEFT, padx=2)
        
        # Font size control
        tk.Label(settings_frame, text="Font:", bg=self.colors["bg"], fg=self.colors["fg"]).pack(side=tk.LEFT, padx=2)
        self.font_var = tk.StringVar(value=str(self.settings_manager.get("font_size", 10)))
        font_spin = tk.Spinbox(
            settings_frame, from_=6, to=30, textvariable=self.font_var, width=5,
            bg=self.colors["button_bg"], fg=self.colors["fg"], relief=tk.FLAT
        )
        font_spin.pack(side=tk.LEFT, padx=2)
        
        # Theme selector
        tk.Label(settings_frame, text="Theme:", bg=self.colors["bg"], fg=self.colors["fg"]).pack(side=tk.LEFT, padx=2)
        self.theme_var = tk.StringVar(value=self.settings_manager.get("theme", "dark"))
        theme_combo = ttk.Combobox(
            settings_frame, textvariable=self.theme_var, values=list(self.THEMES.keys()),
            width=8, state="readonly"
        )
        theme_combo.pack(side=tk.LEFT, padx=2)
        theme_combo.bind('<<ComboboxSelected>>', self.change_theme)
        
        self.btn_apply = tk.Button(
            settings_frame, text="Apply", command=self.apply_settings,
            bg=self.colors["button_bg"], fg=self.colors["fg"], relief=tk.FLAT, padx=10
        )
        self.btn_apply.pack(side=tk.LEFT, padx=5)
        
    def create_display_area(self, parent: tk.Frame) -> None:
        """Create the ASCII art display area with scrollbars"""
        display_frame = tk.Frame(parent, bg="#000000")
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(display_frame, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scrollbar = tk.Scrollbar(display_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        h_scrollbar = tk.Scrollbar(display_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # ASCII frame
        self.ascii_frame = tk.Frame(self.canvas, bg="#000000")
        self.canvas.create_window((0, 0), window=self.ascii_frame, anchor=tk.NW)
        
        # ASCII label
        font_size = self.settings_manager.get("font_size", 10)
        self.ascii_label = tk.Label(
            self.ascii_frame, text="", bg="#000000",
            fg=self.colors["fg"], font=("Courier", font_size), justify=tk.LEFT
        )
        self.ascii_label.pack()
        
    def create_progress_bar(self, parent: tk.Frame) -> None:
        """Create seek slider and time display"""
        progress_frame = tk.Frame(parent, bg=self.colors["bg"])
        progress_frame.pack(fill=tk.X, pady=10)
        
        # Time labels
        self.time_current = tk.Label(progress_frame, text="00:00:00", bg=self.colors["bg"], fg=self.colors["fg"])
        self.time_current.pack(side=tk.LEFT, padx=5)
        
        # Seek slider
        self.seek_var = tk.DoubleVar()
        self.seek_slider = tk.Scale(
            progress_frame, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.seek_var, command=self.on_seek, bg=self.colors["bg"],
            highlightthickness=0, length=400
        )
        self.seek_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # Total time
        self.time_total = tk.Label(progress_frame, text="00:00:00", bg=self.colors["bg"], fg=self.colors["fg"])
        self.time_total.pack(side=tk.RIGHT, padx=5)
        
        # FPS counter
        self.fps_label = tk.Label(progress_frame, text="0 fps", bg=self.colors["bg"], fg="#888888")
        self.fps_label.pack(side=tk.RIGHT, padx=10)
        
    def create_status_bar(self) -> None:
        """Create status bar for messages"""
        self.status_bar = tk.Label(
            self.root, text="Ready", bg=self.colors["status_bg"],
            fg="white", anchor=tk.W, padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_keyboard_shortcuts(self) -> None:
        """Setup keyboard shortcuts"""
        self.root.bind('<space>', lambda e: self.toggle_playback())
        self.root.bind('<Escape>', lambda e: self.stop_playback())
        self.root.bind('<Left>', lambda e: self.frame_step(-1))
        self.root.bind('<Right>', lambda e: self.frame_step(1))
        self.root.bind('<Control-o>', lambda e: self.open_video())
        self.root.bind('<Control-q>', lambda e: self.on_closing())
        
    def open_video(self) -> None:
        """Open video file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_video(file_path)
            
    def load_recent_file(self) -> None:
        """Load recently selected file"""
        file_path = self.recent_var.get()
        if file_path and os.path.exists(file_path):
            self.load_video(file_path)
        elif file_path:
            messagebox.showwarning("File Not Found", f"File no longer exists:\n{file_path}")
            self.settings_manager.add_recent_file(file_path)  # This will remove it
            self.update_recent_menu()
            
    def load_video(self, file_path: str) -> None:
        """Load and prepare video file"""
        try:
            self.status_bar.config(text=f"Loading: {os.path.basename(file_path)}...")
            self.root.update()
            
            # Load video
            if self.video_player.load_video(file_path):
                self.file_label.config(text=os.path.basename(file_path))
                self.btn_play.config(state=tk.NORMAL)
                self.btn_stop.config(state=tk.NORMAL)
                
                # Update UI with metadata
                if self.video_player.metadata:
                    duration = self.video_player.metadata.duration
                    self.time_total.config(text=str(timedelta(seconds=int(duration))))
                    self.seek_slider.config(state=tk.NORMAL)
                
                # Add to recent files
                self.settings_manager.add_recent_file(file_path)
                self.update_recent_menu()
                
                self.status_bar.config(text=f"Loaded: {os.path.basename(file_path)}")
                logger.info(f"Video loaded: {file_path}")
                
                # Show first frame
                self.update_frame_display()
            else:
                raise ValueError("Failed to load video")
                
        except Exception as e:
            error_msg = f"Failed to load video: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
            self.status_bar.config(text="Error loading video")
            
    def update_recent_menu(self) -> None:
        """Update recent files dropdown menu"""
        recent_files = self.settings_manager.get("recent_files", [])
        if recent_files:
            menu = self.recent_menu["menu"]
            menu.delete(0, "end")
            for file in recent_files:
                menu.add_command(label=os.path.basename(file), command=lambda f=file: self.load_video(f))
            self.recent_menu.config(state="normal")
        else:
            self.recent_menu.config(state="disabled")
            
    def toggle_playback(self) -> None:
        """Toggle play/pause state"""
        if not self.video_player.is_loaded():
            return
            
        if self.playback_state == PlaybackState.PLAYING:
            self.playback_state = PlaybackState.PAUSED
            self.btn_play.config(text="Play")
            self.status_bar.config(text="Paused")
        else:
            self.playback_state = PlaybackState.PLAYING
            self.btn_play.config(text="Pause")
            self.status_bar.config(text="Playing")
            self.start_playback()
            
    def start_playback(self) -> None:
        """Start the playback loop"""
        if self.playback_state == PlaybackState.PLAYING:
            self.update_frame_display()
            
    def stop_playback(self) -> None:
        """Stop playback and reset to beginning"""
        self.playback_state = PlaybackState.STOPPED
        self.btn_play.config(text="Play")
        self.btn_stop.config(state=tk.DISABLED)
        
        if self.video_player.is_loaded():
            self.video_player.set_frame_position(0)
            self.update_frame_display()
            
        self.status_bar.config(text="Stopped")
        
    def frame_step(self, delta: int) -> None:
        """Step forward or backward by frames"""
        if not self.video_player.is_loaded():
            return
            
        # Pause playback
        was_playing = self.playback_state == PlaybackState.PLAYING
        if was_playing:
            self.playback_state = PlaybackState.PAUSED
            
        # Move frame
        current = self.video_player.get_current_frame_num()
        new_frame = current + delta
        new_frame = max(0, min(new_frame, self.video_player.metadata.total_frames - 1))
        
        if self.video_player.get_frame(new_frame) is not None:
            self.update_frame_display()
            
        if was_playing:
            self.playback_state = PlaybackState.PLAYING
            
    def on_seek(self, value: str) -> None:
        """Handle seek slider movement"""
        if not self.video_player.is_loaded():
            return
            
        position = float(value) / 100.0
        if self.video_player.set_frame_position(position):
            self.update_frame_display()
            
            # Update time display
            if self.video_player.metadata:
                current_time = position * self.video_player.metadata.duration
                self.time_current.config(text=str(timedelta(seconds=int(current_time))))
                
    def update_frame_display(self) -> None:
        """Update the displayed frame (called from main thread)"""
        if not self.video_player.is_loaded():
            return
            
        # Get frame from queue or directly
        try:
            # Try to get frame from queue (from processing thread)
            frame = self.frame_queue.get_nowait() if not self.frame_queue.empty() else None
            if frame is None:
                frame = self.video_player.get_frame()
        except:
            frame = self.video_player.get_frame()
            
        if frame is not None:
            # Calculate and display FPS
            current_time = time.time()
            if self.last_render_time > 0:
                elapsed = current_time - self.last_render_time
                if elapsed > 0:
                    self.render_fps = 1.0 / elapsed
                    self.fps_label.config(text=f"{self.render_fps:.1f} fps")
            self.last_render_time = current_time
            
            # Convert to ASCII
            ascii_art = self.ascii_converter.frame_to_ascii(frame)
            self.ascii_label.config(text=ascii_art)
            
            # Update scroll region
            self.ascii_frame.update_idletasks()
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            
            # Update progress
            if self.video_player.metadata:
                current_frame = self.video_player.get_current_frame_num()
                position = current_frame / self.video_player.metadata.total_frames
                self.seek_var.set(position * 100)
                
                current_time = position * self.video_player.metadata.duration
                self.time_current.config(text=str(timedelta(seconds=int(current_time))))
                
        # Schedule next update
        if self.playback_state == PlaybackState.PLAYING and self.video_player.metadata:
            # Calculate delay based on video FPS (capped)
            fps = min(self.video_player.metadata.fps, VideoPlayer.MAX_FPS_CAP)
            delay = max(33, int(1000 / fps))  # Minimum 33ms for 30fps
            self.update_job = self.root.after(delay, self.update_frame_display)
            
    def start_frame_processor(self) -> None:
        """Start background thread for frame processing"""
        def process_frames():
            while True:
                if self.playback_state == PlaybackState.PLAYING and self.video_player.is_loaded():
                    frame = self.video_player.get_frame()
                    if frame is not None and self.frame_queue.qsize() < 5:
                        # Clear old frames if queue is filling up
                        while self.frame_queue.qsize() > 2:
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                break
                        self.frame_queue.put(frame)
                time.sleep(0.033)  # ~30fps processing
                
        self.processing_thread = threading.Thread(target=process_frames, daemon=True)
        self.processing_thread.start()
        
    def apply_settings(self) -> None:
        """Apply user settings"""
        try:
            # Validate width
            width = int(self.width_var.get())
            if width <= 0:
                raise ValueError("Width must be positive")
            width = max(10, min(width, 300))
            
            # Validate font size
            font_size = int(self.font_var.get())
            if font_size <= 0:
                raise ValueError("Font size must be positive")
            font_size = max(6, min(font_size, 30))
            
            # Apply settings
            self.ascii_converter.set_width(width)
            self.ascii_label.config(font=("Courier", font_size))
            
            # Save settings
            self.settings_manager.set("ascii_width", width)
            self.settings_manager.set("font_size", font_size)
            
            # Refresh display
            self.update_frame_display()
            
            self.status_bar.config(text="Settings applied")
            logger.info(f"Settings applied: width={width}, font_size={font_size}")
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid settings: {str(e)}")
            
    def change_theme(self, event=None) -> None:
        """Change UI theme"""
        theme = self.theme_var.get()
        if theme in self.THEMES:
            self.settings_manager.set("theme", theme)
            self.colors = self.THEMES[theme]
            # Update colors (simplified - full theme change would need rebuild)
            self.status_bar.config(bg=self.colors["status_bg"])
            self.fps_label.config(fg="#888888")
            self.ascii_label.config(fg=self.colors["fg"])
            self.status_bar.config(text=f"Theme changed to {theme}")
            
    def on_closing(self) -> None:
        """Clean up resources on window close"""
        logger.info("Shutting down...")
        
        # Cancel scheduled updates
        if self.update_job:
            self.root.after_cancel(self.update_job)
            
        # Release video resources
        self.video_player.release()
        
        # Save settings
        self.settings_manager.save()
        
        # Destroy window
        self.root.quit()
        self.root.destroy()


def main():
    """Main entry point"""
    try:
        root = tk.Tk()
        app = ASCIIVideoGUI(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        messagebox.showerror("Fatal Error", f"Application failed to start:\n{str(e)}")


if __name__ == "__main__":
    main()
