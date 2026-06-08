#!/usr/bin/env python3
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


import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("log.log"), logging.StreamHandler()] # log1
)
log = logging.getLogger(__name__)


class AsciiArtConverter:
    CHARS = " .:-=+*#%@"
    
    def __init__(self, width=100):
        self.width = max(10, min(300, width))
        self.chars = self.CHARS
        self.aspect = 0.55  # seems ok
        
    def convert(self, frame):
        if frame is None:
            return ""
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            new_h = int(h * self.width / w * self.aspect) # not magic number its ratio for monospace fonts
            
            if new_h > 500:
                new_h = 500
                
            resized = cv2.resize(gray, (self.width, new_h), interpolation=cv2.INTER_LINEAR)
            

            chars = np.array(list(self.chars))
            indices = (resized / 255.0 * (len(chars) - 1)).astype(np.int32)
            

            return "\n".join("".join(chars[row]) for row in indices)
            
        except Exception as e:
            log.error(f"Conversion failed: {e}")
            return ""


class VideoLoader:

    
    def __init__(self):
        self.cap = None
        self.fps = 24.0
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.path = None
        
    def load(self, path):
        self.close()
        
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Can't open video: {path}")
        

        self.fps = cap.get(cv2.CAP_PROP_FPS) # probably need to use this  https://docs.opencv.org/4.x/dd/d43/classcv_1_1VideoCapture.html
        if self.fps <= 0 or self.fps > 120: #make so default fps if video has invalid value
            self.fps = 24.0 #maybe warn usr?
            
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            self.total_frames = 1
            
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        ret, test = cap.read()
        if not ret or test is None:
            cap.release()
            raise ValueError("Can't read frames from this video")
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cap = cap
        self.path = str(path)
        
        return {
            'fps': self.fps,
            'frames': self.total_frames,
            'duration': self.total_frames / self.fps,
            'size': f"{self.width}x{self.height}"
        }
    
    def get_frame(self, frame_num=None):
        if not self.cap:
            return None
            
        try:
            if frame_num is not None:
                frame_num = max(0, min(frame_num, self.total_frames - 1))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                
            ret, frame = self.cap.read()
            if ret and frame is not None:
                pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                return {
                    'frame': frame,
                    'num': pos,
                    'time': pos / self.fps if self.fps > 0 else 0
                }
        except Exception as e:
            log.error(f"Frame read error: {e}")
            
        return None
    
    def seek(self, pos_01): # dont work properly with CV_PROP_POS_AVI_RATIO 
        if not self.cap:
            return False
            
        try:
            pos_01 = max(0.0, min(1.0, pos_01))
            self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, pos_01)
            return True
        except:
            return False
    
    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None


class Settings:
    
    def __init__(self):
        self.file = Path.home() / ".config.json" #second one
        self.data = {
            'width': 100,
            'font_size': 10,
            'charset': " .:-=+*#%@",
            'aspect': 0.55,
            'theme': 'dark',
            'loop': False,
            'recent': []
        }
        self.load()
    
    def load(self):
        if self.file.exists():
            try:
                with open(self.file, 'r') as f:
                    saved = json.load(f)
                    self.data.update(saved)
                    self.data['recent'] = [p for p in self.data.get('recent', []) if Path(p).exists()]
            except:
                pass
    
    def save(self):
        try:
            with open(self.file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except:
            pass
    
    def add_recent(self, path):
        recent = self.data.get('recent', [])
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        self.data['recent'] = recent[:10]
        self.save()
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def set(self, key, value):
        self.data[key] = value
        self.save()


class ASCIIVideoGUI:
    
    THEMES = {
        'dark':   {'bg': '#1e1e1e', 'fg': '#ffffff', 'accent': '#007acc', 'btn': '#2d2d2d'},
        'matrix': {'bg': '#000000', 'fg': '#00ff00', 'accent': '#008800', 'btn': '#003300'},
        'amber':  {'bg': '#000000', 'fg': '#ffb000', 'accent': '#cc8800', 'btn': '#332200'},
    }
    
    def __init__(self, root):
        self.root = root
        root.title("ASCII Video Player")
        root.geometry("1200x800")
        
        self.video = VideoLoader()
        self.ascii = AsciiArtConverter()
        self.settings = Settings()
        
        self.playing = False
        
        self.worker_run = True
        self.frame_queue = queue.Queue(maxsize=3) #balance between smooth playback and memory usage
        self.seeking = False
        self.current_frame = None
        self.delay = int(1000 / max(self.video.fps, 1.0))
        self._setup_ui()
        self._load_settings()
        self._start_worker()
        
        root.protocol("WM_DELETE_WINDOW", self._quit)
        
        log.info("loaded")
    
    def _setup_ui(self):

        theme = self.THEMES[self.settings.get('theme', 'dark')]
        self.root.configure(bg=theme['bg'])
        
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Video...", command=self.open_video, accelerator="Ctrl+O")
        file_menu.add_command(label="Save ASCII", command=self.save_ascii, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._quit, accelerator="Ctrl+Q")
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Shortcuts", command=self.show_shortcuts)
        help_menu.add_command(label="About", command=self.show_about)
        
        main = tk.Frame(self.root, bg=theme['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        controls = tk.Frame(main, bg=theme['bg'])
        controls.pack(fill=tk.X, pady=(0,10))
        
        left = tk.Frame(controls, bg=theme['bg'])
        left.pack(side=tk.LEFT)
        
        self.file_label = tk.Label(left, text="No video", bg=theme['bg'], fg='#888')
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        self.open_btn = tk.Button(left, text="Open", command=self.open_video, 
                                   bg=theme['accent'], fg='white', relief=tk.FLAT, padx=10)
        self.open_btn.pack(side=tk.LEFT, padx=5)
        
        self.recent_var = tk.StringVar()
        self.recent_menu = tk.OptionMenu(left, self.recent_var, "")
        self.recent_menu.config(bg=theme['btn'], fg=theme['fg'], relief=tk.FLAT)
        self.recent_var.trace_add('write', lambda *_: self._load_recent())
        self.recent_menu.pack(side=tk.LEFT, padx=5)
        
        play_frame = tk.Frame(controls, bg=theme['bg'])
        play_frame.pack(side=tk.LEFT, padx=20)
        
        self.play_btn = tk.Button(play_frame, text="Play", command=self.toggle_play,
                                   bg=theme['btn'], fg=theme['fg'], relief=tk.FLAT, padx=10)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = tk.Button(play_frame, text="Stop", command=self.stop,
                                   bg=theme['btn'], fg=theme['fg'], relief=tk.FLAT, padx=10)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        self.loop_var = tk.BooleanVar(value=self.settings.get('loop', False))
        self.loop_cb = tk.Checkbutton(play_frame, text="Loop", variable=self.loop_var,
                                       command=lambda: self.settings.set('loop', self.loop_var.get()),
                                       bg=theme['bg'], fg=theme['fg'], selectcolor=theme['bg'])
        self.loop_cb.pack(side=tk.LEFT, padx=10)
        
        right = tk.Frame(controls, bg=theme['bg'])
        right.pack(side=tk.RIGHT)
        
        tk.Label(right, text="W:", bg=theme['bg'], fg=theme['fg']).pack(side=tk.LEFT)
        self.width_var = tk.StringVar(value="100")
        tk.Spinbox(right, from_=10, to=300, textvariable=self.width_var, width=5,
                   bg=theme['btn'], fg=theme['fg'], relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        
        tk.Label(right, text="Chars:", bg=theme['bg'], fg=theme['fg']).pack(side=tk.LEFT)
        self.charset_entry = tk.Entry(right, width=12, bg=theme['btn'], fg=theme['fg'], relief=tk.FLAT)
        self.charset_entry.pack(side=tk.LEFT, padx=2)
        
        tk.Label(right, text="Theme:", bg=theme['bg'], fg=theme['fg']).pack(side=tk.LEFT)
        self.theme_combo = ttk.Combobox(right, values=list(self.THEMES.keys()), width=8, state='readonly')
        self.theme_combo.pack(side=tk.LEFT, padx=2)
        self.theme_combo.bind('<<ComboboxSelected>>', self._change_theme)
        
        tk.Button(right, text="Apply", command=self._apply,
                 bg=theme['btn'], fg=theme['fg'], relief=tk.FLAT, padx=8).pack(side=tk.LEFT, padx=5)
        
        disp_frame = tk.Frame(main, bg='#000000')
        disp_frame.pack(fill=tk.BOTH, expand=True)
        
        self.text = tk.Text(disp_frame, bg='#000000', fg=theme['fg'],
                           font=("Courier", 10), wrap=tk.NONE, relief=tk.FLAT)
        self.text.pack(fill=tk.BOTH, expand=True)
        
        v_scroll = tk.Scrollbar(disp_frame, orient=tk.VERTICAL, command=self.text.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll = tk.Scrollbar(disp_frame, orient=tk.HORIZONTAL, command=self.text.xview)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.text.config(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        prog_frame = tk.Frame(main, bg=theme['bg'])
        prog_frame.pack(fill=tk.X, pady=10)
        
        self.time_label = tk.Label(prog_frame, text="00:00:00", bg=theme['bg'], fg=theme['fg'])
        self.time_label.pack(side=tk.LEFT, padx=5)
        
        self.slider = tk.Scale(prog_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                               bg=theme['bg'], highlightthickness=0)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.slider.bind('<ButtonPress-1>', lambda e: setattr(self, 'seeking', True))
        self.slider.bind('<ButtonRelease-1>', lambda e: self._do_seek())
        
        self.total_label = tk.Label(prog_frame, text="00:00:00", bg=theme['bg'], fg=theme['fg'])
        self.total_label.pack(side=tk.RIGHT, padx=5)
        
        self.fps_label = tk.Label(prog_frame, text="0 fps", bg=theme['bg'], fg='#888')
        self.fps_label.pack(side=tk.RIGHT, padx=10)
        
        self.status = tk.Label(self.root, text="Ready | Press F1 for help",
                               bg=theme['accent'], fg='white', anchor=tk.W, padx=10)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        
        # elseifelseifesleif
        self.root.bind('<space>', lambda e: self.toggle_play())
        self.root.bind('<Escape>', lambda e: self.stop())
        self.root.bind('<Left>', lambda e: self._step(-1))
        self.root.bind('<Right>', lambda e: self._step(1))
        self.root.bind('<Control-o>', lambda e: self.open_video())
        self.root.bind('<Control-s>', lambda e: self.save_ascii()) # third doweneed it?
        self.root.bind('<Control-q>', lambda e: self._quit())
        self.root.bind('<F1>', lambda e: self.show_shortcuts())
        self.root.bind('<Home>', lambda e: self._seek_to_start())
        self.root.bind('<End>', lambda e: self._seek_to_end())
        self.root.bind('<Control-plus>', lambda e: self._adjust_font(1))
        self.root.bind('<Control-minus>', lambda e: self._adjust_font(-1))
        
        self.play_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        
        self._update_recent_menu()
    
    def _load_settings(self):
        self.ascii.width = self.settings.get('width', 100)
        self.ascii.chars = self.settings.get('charset', AsciiArtConverter.CHARS)
        self.ascii.aspect = self.settings.get('aspect', 0.55)
        
        self.width_var.set(str(self.ascii.width))
        self.charset_entry.insert(0, self.ascii.chars)
        self.theme_combo.set(self.settings.get('theme', 'dark'))
        
        font_size = self.settings.get('font_size', 10)
        self.text.config(font=("Courier", font_size))
    
    def _update_recent_menu(self):

        recent = self.settings.get('recent', [])
        menu = self.recent_menu['menu']
        menu.delete(0, 'end')
        
        if recent:
            for path in recent:
                name = Path(path).name
                if len(name) > 40:
                    name = name[:37] + "..."
                menu.add_command(label=name, command=lambda p=path: self._load_video(p))
            self.recent_menu.config(state=tk.NORMAL)
        else:
            menu.add_command(label="No recent files", command=lambda: None)
            self.recent_menu.config(state=tk.DISABLED)
    
    def _load_recent(self):
        path = self.recent_var.get()
        if path and Path(path).exists():
            self._load_video(path)
    
    def open_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.webm"), ("All", "*.*")]
        )
        if path:
            self._load_video(path)
    
    def _load_video(self, path): # why do we need second one to actually load
        try:
            self.stop()
            
            self.status.config(text=f"Loading {Path(path).name}...")
            self.root.update()
            
            info = self.video.load(path)
            
            self.file_label.config(text=Path(path).name)
            self.play_btn.config(state=tk.NORMAL, text="Play")
            self.stop_btn.config(state=tk.NORMAL)
            self.total_label.config(text=self._format_time(info['duration']))
            
            self.settings.add_recent(str(path))
            self._update_recent_menu()

            #show first image thing
            self.video.seek(0)
            frame = self.video.get_frame(0)
            if frame:
                self._show_frame(frame)
            
            self.status.config(text=f"Loaded: {Path(path).name} | {info['size']} | {info['fps']:.1f}fps")
            
        except Exception as e:
            log.error(f"Load failed: {e}")
            messagebox.showerror("Error", f"Can't load video:\n{e}")
            self.status.config(text="Load failed")
    
    def toggle_play(self):
        if not self.video.cap:
            return
            
        if self.playing:
            self.playing = False
            self.play_btn.config(text="Play")
            self.status.config(text="Paused")
        else:
            self.playing = True
            self.play_btn.config(text="Pause")
            self.status.config(text="Playing")
            self._schedule_display()
    
    def stop(self):
        """Stop and reset to beginning"""
        self.playing = False
        self.play_btn.config(text="Play")
        self.status.config(text="Stopped")
        
        while not self.frame_queue.empty(): # inaficient breaks sometimes
            try:
                self.frame_queue.get_nowait()
            except:
                break #very retarted
        
        if self.video.cap:
            self.video.seek(0)
            frame = self.video.get_frame(0)
            if frame:
                self._show_frame(frame)
    
    def _step(self, delta):
        if not self.video.cap:
            return
            
        was_playing = self.playing
        if was_playing:
            self.playing = False
        
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        current = self.video.get_frame()
        if current:
            new_num = max(0, min(current['num'] + delta, self.video.total_frames - 1))
            frame = self.video.get_frame(new_num)
            if frame:
                self._show_frame(frame)
        
        if was_playing:
            self.playing = True
            self._schedule_display()
    
    def _seek_to_start(self):
        if self.video.cap:
            was_playing = self.playing
            if was_playing:
                self.playing = False
            self.video.seek(0)
            frame = self.video.get_frame(0)
            if frame:
                self._show_frame(frame)
            if was_playing:
                self.playing = True
                self._schedule_display()
    
    def _seek_to_end(self):
        if self.video.cap:
            was_playing = self.playing
            if was_playing:
                self.playing = False
            self.video.seek(0.99)
            frame = self.video.get_frame()
            if frame:
                self._show_frame(frame)
            if was_playing:
                self.playing = True
                self._schedule_display()
    
    def _do_seek(self): # we have this seaking mechanism but also we have _step
        if not self.video.cap: # need to do something abt this issue (maybe merge?)
            self.seeking = False
            return
            
        pos = self.slider.get() / 100.0
        was_playing = self.playing
        
        if was_playing:
            self.playing = False
        
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        self.video.seek(pos)
        frame = self.video.get_frame()
        if frame:
            self._show_frame(frame)
        
        if was_playing:
            self.playing = True
            self._schedule_display()
        
        self.seeking = False
    
    def _show_frame(self, frame_data):
        ascii_text = self.ascii.convert(frame_data['frame'])
        self.current_frame = frame_data
        
        self.text.delete(1.0, tk.END)
        self.text.insert(1.0, ascii_text)
        
        if not self.seeking:
            progress = frame_data['num'] / max(1, self.video.total_frames) * 100
            self.slider.set(progress)
            self.time_label.config(text=self._format_time(frame_data['time']))
    
    def _format_time(self, seconds):
        """Convert seconds to HH:MM:SS"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def _start_worker(self):
        """Background thread for reading video frames"""
        def worker():
            last_frame_time = 0
            
            while self.worker_run:
                if self.playing and self.video.cap:
                    fps = self.video.fps
                    min_interval = 1.0 / fps if fps > 0 else 0.04
                    
                    now = time.time()
                    if now - last_frame_time >= min_interval:
                        frame = self.video.get_frame()
                        if frame:
                            try:
                                self.frame_queue.put_nowait(frame)
                            except queue.Full:
                                try:
                                    self.frame_queue.get_nowait()
                                    self.frame_queue.put_nowait(frame)
                                except:
                                    pass
                            last_frame_time = now
                        else:
                            if self.loop_var.get():
                                self.video.seek(0)
                            else:
                                self.root.after(0, self.stop)
                                self.playing = False
                    else:
                        time.sleep(0.005)
                else:
                    time.sleep(0.02)
        
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
    
    def _schedule_display(self):
        """Schedule next frame display"""
        if self.playing and self.video.cap:
            try:
                frame = self.frame_queue.get_nowait()
                self._show_frame(frame)
                
                if hasattr(self, '_frame_count'):
                    self._frame_count += 1
                    if self._frame_count % 30 == 0:
                        fps = self.video.fps
                        self.fps_label.config(text=f"{fps:.1f} fps")
                else:
                    self._frame_count = 0
                    
            except queue.Empty:
                pass
            
            # shedule next frame update
            delay = int(1000 / max(self.video.fps, 1.0))
            self.root.after(delay, self._schedule_display)
    
    def _apply(self):

        try:
            width = int(self.width_var.get())
            charset = self.charset_entry.get().strip()
            theme = self.theme_combo.get()
            font_size = int(self.text.cget('font').split()[-1])  # what font size
            
            if width < 10 or width > 300:
                raise ValueError("Width must be 10-300")
            if len(charset) < 2:
                raise ValueError("Charset needs at least 2 chars")
            if theme not in self.THEMES:
                raise ValueError("Bad theme")
            self.ascii.width = width
            self.ascii.chars = charset
            
            self.settings.set('width', width)
            self.settings.set('charset', charset)
            self.settings.set('theme', theme)
            
            if self.current_frame:
                self._show_frame(self.current_frame)
            
            self.status.config(text="Settings applied")
            
        except Exception as e:
            messagebox.showerror("Bad setting", str(e))
    
    def _change_theme(self, e=None):
        theme_name = self.theme_combo.get()
        colors = self.THEMES.get(theme_name, self.THEMES['dark'])
        
        self.root.configure(bg=colors['bg'])
        self.status.config(bg=colors['accent'])
        self.open_btn.config(bg=colors['accent'])
        self.play_btn.config(bg=colors['btn'], fg=colors['fg'])
        self.stop_btn.config(bg=colors['btn'], fg=colors['fg'])
        self.text.config(fg=colors['fg'])
        
        self.settings.set('theme', theme_name)
        self.status.config(text=f"Theme: {theme_name}")
    
    def _adjust_font(self, delta):
        current = int(self.text.cget('font').split()[-1])
        new = max(6, min(30, current + delta))
        self.text.config(font=("Courier", new))
        self.settings.set('font_size', new)
    
    def save_ascii(self):
        content = self.text.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("Nothing to save", "No ASCII art displayed")
            return
            
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("Text files", "*.txt")])
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.status.config(text=f"Saved to {Path(path).name}")
            except Exception as e:
                messagebox.showerror("Save failed", str(e))
    
    def show_shortcuts(self):
        shortcuts = """
Space           - Play/Pause
Escape          - Stop & rewind
← / →           - Previous/next frame
Home / End      - Jump to start/end
Ctrl+O          - Open video
Ctrl+S          - Save ASCII
Ctrl+Q          - Quit
Ctrl+Plus/Minus - Zoom text
F1              - This help"""
        messagebox.showinfo("Shortcuts", shortcuts)
    
    def show_about(self):
        about = """ASCII Video Player
made by 74fm""" #need to add soemthing more here its bare
        messagebox.showinfo("About", about) 
    
    def _quit(self):
        """Clean shutdown"""
        log.info("Shutting down...")
        self.worker_run = False
        self.playing = False
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join(timeout=1)
        self.video.close()
        self.settings.save()
        self.root.quit()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ASCIIVideoGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main() # note some ai was used but i needed for help ty
