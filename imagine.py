#!/usr/bin/env python3

# autopep8: off

import os
import sys

TQDM_UPDATE_INTERVAL = 0.05
TQDM_MIN_ITERATIONS = 1
TQDM_MAX_INTERVAL = 1.0

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['HF_HUB_TQDM_MININTERVAL'] = str(TQDM_UPDATE_INTERVAL)
os.environ['HF_HUB_TQDM_ASCII'] = '1'

if not os.environ.get('PYTHONIOENCODING'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import filecmp
import shutil
from pathlib import Path
import json
from datetime import datetime
import threading
import ctypes
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf, GLib, Pango
from transformers import CLIPTokenizer
import warnings
warnings.filterwarnings("ignore", message=".*device_type of 'cuda', but CUDA is not available.*")
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, TCDScheduler, LCMScheduler, logging
import torch
import gc
import signal

torch.backends.nnpack.enabled = False

logging.set_verbosity_error()

warnings.filterwarnings("ignore", message=".*peft_config.*multiple adapters.*")

try:
    import tqdm as tqdm_module
    _original_tqdm_init = tqdm_module.tqdm.__init__

    def _patched_tqdm_init(self, *args, **kwargs):
        kwargs = kwargs.copy() if kwargs else {}

        if 'ascii' not in kwargs:
            kwargs['ascii'] = True

        if 'mininterval' not in kwargs:
            kwargs['mininterval'] = TQDM_UPDATE_INTERVAL

        if 'maxinterval' not in kwargs:
            kwargs['maxinterval'] = TQDM_MAX_INTERVAL

        if 'miniters' not in kwargs:
            kwargs['miniters'] = TQDM_MIN_ITERATIONS

        return _original_tqdm_init(self, *args, **kwargs)

    tqdm_module.tqdm.__init__ = _patched_tqdm_init
except (ImportError, AttributeError) as e:
    print(f"Warning: Could not patch tqdm: {e}", file=sys.stderr)
except Exception as e:
    print(f"Warning: Unexpected error while patching tqdm: {e}", file=sys.stderr)

CONFIG_FILE = Path.home() / ".config" / "imagine" / "settings.json"
OUTPUT_DIR = Path.home() / ".config" / "imagine" / "outputs"
LORA_DIR = Path.home() / ".config" / "imagine" / "loras"

TOKEN_LIMIT = 75
TOKEN_WARNING_THRESHOLD = 60

GENERATION_THREAD_TIMEOUT = 5.0
LOAD_THREAD_TIMEOUT = 3.0

MODEL_TYPE_SDXL = 0
MODEL_TYPE_SD15 = 1

_gui_instance = None

# autopep8: on


def raise_exception_in_thread(thread_obj):
    if thread_obj is None or not thread_obj.is_alive():
        return False

    thread_id = None
    try:
        for tid, tobj in threading._active.items():
            if tobj is thread_obj:
                thread_id = tid
                break
    except Exception as e:
        print(f"Warning: Could not access thread ID: {e}")
        return False

    if thread_id is None:
        return False

    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread_id),
        ctypes.py_object(KeyboardInterrupt)
    )

    if res == 0:
        return False
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(thread_id), None)
        return False

    return True


def is_direct_url(path_str):
    if not path_str or not isinstance(path_str, str):
        return False
    return path_str.startswith(('http://', 'https://'))


def normalize_huggingface_url(url):
    if not url or not isinstance(url, str):
        return url

    if 'huggingface.co' in url:
        url = url.replace('/raw/main/', '/')
        url = url.replace('/resolve/main/', '/')

    return url


def is_huggingface_repo(path_str):
    if not path_str or not isinstance(path_str, str):
        return False

    path_obj = Path(path_str)

    if path_obj.exists():
        return False

    if len(path_str) >= 2 and path_str[1] == ':' and path_str[0].isalpha():
        return False

    parts = path_str.split('/')
    if len(parts) == 2 and not path_str.startswith(('.', '/')):
        if not any(c in path_str for c in ['\\', '~']):
            return True

    return False


def copy_lora_to_config(source_path, lora_name=None):
    try:
        source = Path(source_path)
        if not source.exists():
            print(
                f"Failed to copy LoRA: source path does not exist: {source_path}")
            print("Please verify the file exists and try again.")
            return None

        LORA_DIR.mkdir(parents=True, exist_ok=True)

        if lora_name is None:
            lora_name = source.name

        dest = LORA_DIR / lora_name

        if dest.exists():
            if source.is_file() and dest.is_file():
                if filecmp.cmp(source, dest, shallow=False):
                    print(
                        f"LoRA file already exists in config directory: {dest}")
                    return dest
            elif source.is_dir() and dest.is_dir():
                print(
                    f"LoRA directory already exists in config directory: {dest}")
                return dest

        if source.is_file():
            print(f"Copying LoRA file from {source} to {dest}")
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.copy2(source, dest)
        elif source.is_dir():
            print(f"Copying LoRA directory from {source} to {dest}")
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.copytree(source, dest)

        return dest

    except Exception as e:
        print(f"Error copying LoRA to config directory: {e}")
        return None


class ConsoleRedirector:
    def __init__(self, text_view, original_stream):
        self.text_view = text_view
        self.text_buffer = text_view.get_buffer()
        self.original_stream = original_stream
        self.needs_newline = False
        self._lock = threading.Lock()

    def _update_newline_tracking(self, text):
        if text:
            with self._lock:
                self.needs_newline = not text.endswith('\n')

    def write(self, text):
        self.original_stream.write(text)
        self.original_stream.flush()

        GLib.idle_add(self._append_text, text, False)

        self._update_newline_tracking(text)

    def write_with_newline(self, text):
        self.original_stream.write(text)
        self.original_stream.flush()

        with self._lock:
            needs_newline = self.needs_newline

        GLib.idle_add(self._append_text, text, needs_newline)

        self._update_newline_tracking(text)

    def _append_text(self, text, prepend_newline):
        end_iter = self.text_buffer.get_end_iter()
        if prepend_newline:
            self.text_buffer.insert(end_iter, '\n' + text)
        else:
            self.text_buffer.insert(end_iter, text)

        mark = self.text_buffer.create_mark(
            None, self.text_buffer.get_end_iter(), False)
        self.text_view.scroll_mark_onscreen(mark)
        self.text_buffer.delete_mark(mark)

        return False

    def flush(self):
        self.original_stream.flush()


class ImagineGUI(Gtk.Window):
    def __init__(self):
        super().__init__(title="Imagine")
        warnings.filterwarnings("ignore")
        self.set_wmclass("Imagine", "Imagine")
        self.set_default_size(800, 900)
        self.set_border_width(10)

        self.pipe = None
        self.generating = False
        self.loading_model = False
        self.stop_event = threading.Event()
        self.generation_thread = None
        self.load_thread = None
        self.current_image_path = None
        self.loras_fused = False
        self.stop_click_count = 0

        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14")
        except Exception as e:
            print(f"Warning: Could not load tokenizer for token counting: {e}")
            self.tokenizer = None

        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        LORA_DIR.mkdir(parents=True, exist_ok=True)

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.add(main_box)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_size_request(-1, 400)

        controls_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        controls_box.set_border_width(10)
        scrolled.add(controls_box)
        main_box.pack_start(scrolled, False, False, 0)

        top_spacer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        top_spacer.set_size_request(-1, 20)
        controls_box.pack_start(top_spacer, False, False, 0)

        model_type_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        model_type_label = Gtk.Label(label="Model Type:")
        model_type_label.set_size_request(100, -1)
        model_type_label.set_xalign(0)
        model_type_box.pack_start(model_type_label, False, False, 0)

        self.model_type_combo = Gtk.ComboBoxText()
        self.model_type_combo.append_text("SDXL")
        self.model_type_combo.append_text("SD 1.5")
        self.model_type_combo.set_active(0)
        model_type_box.pack_start(self.model_type_combo, False, True, 0)
        controls_box.pack_start(model_type_box, False, False, 0)

        model_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        model_label = Gtk.Label(label="Model:")
        model_label.set_size_request(100, -1)
        model_label.set_xalign(0)
        model_box.pack_start(model_label, False, False, 0)

        self.model_entry = Gtk.Entry()
        self.model_entry.set_text("segmind/SSD-1B")
        model_box.pack_start(self.model_entry, True, True, 0)
        controls_box.pack_start(model_box, False, False, 0)

        dtype_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        dtype_label = Gtk.Label(label="Precision:")
        dtype_label.set_size_request(100, -1)
        dtype_label.set_xalign(0)
        dtype_box.pack_start(dtype_label, False, False, 0)

        self.dtype_combo = Gtk.ComboBoxText()
        self.dtype_combo.append_text("float32")
        self.dtype_combo.append_text("float16")
        self.dtype_combo.append_text("float16+float32 hybrid")
        self.dtype_combo.set_active(0)
        dtype_box.pack_start(self.dtype_combo, False, True, 0)
        controls_box.pack_start(dtype_box, False, False, 0)

        fuse_lora_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self.fuse_lora_check = Gtk.CheckButton(label="Fuse LoRAs")
        self.fuse_lora_check.set_active(False)
        fuse_lora_box.pack_start(self.fuse_lora_check, False, False, 0)
        controls_box.pack_start(fuse_lora_box, False, False, 0)

        self.lora_entries = []
        self.lora_weight_entries = []

        for i in range(4):
            lora_frame = Gtk.Frame(label=f"LoRA {i + 1}")
            lora_frame_box = Gtk.Box(
                orientation=Gtk.Orientation.VERTICAL, spacing=5)
            lora_frame_box.set_border_width(5)
            lora_frame.add(lora_frame_box)

            lora_path_box = Gtk.Box(
                orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
            lora_path_label = Gtk.Label(label="Path/Repo:")
            lora_path_label.set_size_request(80, -1)
            lora_path_label.set_xalign(0)
            lora_path_box.pack_start(lora_path_label, False, False, 0)

            lora_entry = Gtk.Entry()
            lora_path_box.pack_start(lora_entry, True, True, 0)
            self.lora_entries.append(lora_entry)

            browse_btn = Gtk.Button(label="Browse...")
            browse_btn.connect("clicked", self.on_browse_lora, i)
            lora_path_box.pack_start(browse_btn, False, False, 0)

            clear_btn = Gtk.Button(label="Clear")
            clear_btn.connect("clicked", self.on_clear_lora, i)
            lora_path_box.pack_start(clear_btn, False, False, 0)

            lora_frame_box.pack_start(lora_path_box, False, False, 0)

            lora_weight_box = Gtk.Box(
                orientation=Gtk.Orientation.HORIZONTAL, spacing=5)

            weight_name_label = Gtk.Label(label="Weight Name:")
            weight_name_label.set_size_request(80, -1)
            weight_name_label.set_xalign(0)
            lora_weight_box.pack_start(weight_name_label, False, False, 0)

            weight_name_entry = Gtk.Entry()
            lora_weight_box.pack_start(weight_name_entry, True, True, 0)

            weight_label = Gtk.Label(label="Weight:")
            weight_label.set_size_request(50, -1)
            lora_weight_box.pack_start(weight_label, False, False, 0)

            weight_spin = Gtk.SpinButton()
            weight_adj = Gtk.Adjustment(
                value=1.0, lower=0.0, upper=2.0, step_increment=0.1, page_increment=0.5)
            weight_spin.set_adjustment(weight_adj)
            weight_spin.set_digits(2)
            weight_spin.set_size_request(80, -1)
            lora_weight_box.pack_start(weight_spin, False, False, 0)

            self.lora_weight_entries.append((weight_name_entry, weight_spin))
            lora_frame_box.pack_start(lora_weight_box, False, False, 0)

            controls_box.pack_start(lora_frame, False, False, 0)

        steps_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        steps_label = Gtk.Label(label="Steps:")
        steps_label.set_size_request(100, -1)
        steps_label.set_xalign(0)
        steps_box.pack_start(steps_label, False, False, 0)

        self.steps_spin = Gtk.SpinButton()
        steps_adj = Gtk.Adjustment(
            value=6, lower=1, upper=100, step_increment=1, page_increment=5)
        self.steps_spin.set_adjustment(steps_adj)
        self.steps_spin.set_size_request(100, -1)
        steps_box.pack_start(self.steps_spin, False, False, 0)
        controls_box.pack_start(steps_box, False, False, 0)

        guidance_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        guidance_label = Gtk.Label(label="Guidance Scale:")
        guidance_label.set_size_request(100, -1)
        guidance_label.set_xalign(0)
        guidance_box.pack_start(guidance_label, False, False, 0)

        self.guidance_spin = Gtk.SpinButton()
        guidance_adj = Gtk.Adjustment(
            value=7.5, lower=0.0, upper=20.0, step_increment=0.5, page_increment=1.0)
        self.guidance_spin.set_adjustment(guidance_adj)
        self.guidance_spin.set_digits(1)
        self.guidance_spin.set_size_request(100, -1)
        guidance_box.pack_start(self.guidance_spin, False, False, 0)
        controls_box.pack_start(guidance_box, False, False, 0)

        eta_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        eta_label = Gtk.Label(label="Eta (SSS):")
        eta_label.set_size_request(100, -1)
        eta_label.set_xalign(0)
        eta_box.pack_start(eta_label, False, False, 0)

        self.eta_spin = Gtk.SpinButton()
        eta_adj = Gtk.Adjustment(
            value=0.3, lower=0.0, upper=1.0, step_increment=0.05, page_increment=0.1)
        self.eta_spin.set_adjustment(eta_adj)
        self.eta_spin.set_digits(2)
        self.eta_spin.set_size_request(100, -1)
        eta_box.pack_start(self.eta_spin, False, False, 0)

        controls_box.pack_start(eta_box, False, False, 0)

        clip_skip_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        clip_skip_label = Gtk.Label(label="Clip Skip:")
        clip_skip_label.set_size_request(100, -1)
        clip_skip_label.set_xalign(0)
        clip_skip_box.pack_start(clip_skip_label, False, False, 0)

        self.clip_skip_spin = Gtk.SpinButton()
        clip_skip_adj = Gtk.Adjustment(
            value=2, lower=0, upper=10, step_increment=1, page_increment=1)
        self.clip_skip_spin.set_adjustment(clip_skip_adj)
        self.clip_skip_spin.set_size_request(100, -1)
        clip_skip_box.pack_start(self.clip_skip_spin, False, False, 0)
        controls_box.pack_start(clip_skip_box, False, False, 0)

        trigger_label = Gtk.Label(label="Trigger:")
        trigger_label.set_xalign(0)
        controls_box.pack_start(trigger_label, False, False, 0)

        trigger_scroll = Gtk.ScrolledWindow()
        trigger_scroll.set_policy(
            Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        trigger_scroll.set_size_request(-1, 60)

        self.trigger_text = Gtk.TextView()
        self.trigger_text.set_wrap_mode(Gtk.WrapMode.WORD)
        self.trigger_text.get_buffer().connect("changed", self.on_text_changed)
        trigger_scroll.add(self.trigger_text)
        controls_box.pack_start(trigger_scroll, False, False, 0)

        self.trigger_token_label = Gtk.Label(label="0/75")
        self.trigger_token_label.set_xalign(1)
        controls_box.pack_start(self.trigger_token_label, False, False, 0)

        prompt_label = Gtk.Label(label="Prompt:")
        prompt_label.set_xalign(0)
        controls_box.pack_start(prompt_label, False, False, 0)

        prompt_scroll = Gtk.ScrolledWindow()
        prompt_scroll.set_policy(
            Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        prompt_scroll.set_size_request(-1, 60)

        self.prompt_text = Gtk.TextView()
        self.prompt_text.set_wrap_mode(Gtk.WrapMode.WORD)
        self.prompt_text.get_buffer().connect("changed", self.on_text_changed)
        prompt_scroll.add(self.prompt_text)
        controls_box.pack_start(prompt_scroll, False, False, 0)

        self.prompt_token_label = Gtk.Label(label="0/75")
        self.prompt_token_label.set_xalign(1)
        controls_box.pack_start(self.prompt_token_label, False, False, 0)

        neg_prompt_label = Gtk.Label(label="Negative Prompt:")
        neg_prompt_label.set_xalign(0)
        controls_box.pack_start(neg_prompt_label, False, False, 0)

        neg_prompt_scroll = Gtk.ScrolledWindow()
        neg_prompt_scroll.set_policy(
            Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        neg_prompt_scroll.set_size_request(-1, 60)

        self.neg_prompt_text = Gtk.TextView()
        self.neg_prompt_text.set_wrap_mode(Gtk.WrapMode.WORD)
        neg_buffer = self.neg_prompt_text.get_buffer()
        neg_buffer.set_text("")
        neg_buffer.connect("changed", self.on_text_changed)
        neg_prompt_scroll.add(self.neg_prompt_text)
        controls_box.pack_start(neg_prompt_scroll, False, False, 0)

        self.neg_prompt_token_label = Gtk.Label(label="0/75")
        self.neg_prompt_token_label.set_xalign(1)
        controls_box.pack_start(self.neg_prompt_token_label, False, False, 0)

        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)

        self.load_button = Gtk.Button(label="Load Model & LoRAs")
        self.load_button.connect("clicked", self.on_load_clicked)
        button_box.pack_start(self.load_button, True, True, 0)

        self.generate_button = Gtk.Button(label="Generate")
        self.generate_button.connect("clicked", self.on_generate_clicked)
        self.generate_button.set_sensitive(False)
        button_box.pack_start(self.generate_button, True, True, 0)

        self.stop_button = Gtk.Button(label="Stop")
        self.stop_button.connect("clicked", self.on_stop_clicked)
        self.stop_button.set_sensitive(False)
        self.stop_button.set_no_show_all(True)
        button_box.pack_start(self.stop_button, True, True, 0)

        self.restore_defaults_button = Gtk.Button(label="Restore Defaults")
        self.restore_defaults_button.connect(
            "clicked", self.on_restore_defaults_clicked)
        button_box.pack_start(self.restore_defaults_button, True, True, 0)

        controls_box.pack_start(button_box, False, False, 0)

        self.status_label = Gtk.Label(
            label="Ready. Click 'Load Model & LoRAs' to begin.")
        self.status_label.set_xalign(0)
        self.status_label.set_ellipsize(Pango.EllipsizeMode.END)
        self.status_label.set_max_width_chars(80)
        main_box.pack_start(self.status_label, False, False, 0)

        self.notebook = Gtk.Notebook()
        main_box.pack_start(self.notebook, True, True, 0)

        console_scrolled = Gtk.ScrolledWindow()
        console_scrolled.set_policy(
            Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        self.console_text = Gtk.TextView()
        self.console_text.set_editable(False)
        self.console_text.set_wrap_mode(Gtk.WrapMode.CHAR)
        self.console_text.set_cursor_visible(False)

        try:
            css_provider = Gtk.CssProvider()
            css_provider.load_from_data(b"""
                textview {
                    font-family: monospace;
                    font-size: 12pt;
                }
            """)
            style_context = self.console_text.get_style_context()
            style_context.add_provider(
                css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        except Exception as e:
            print(f"Could not set monospace font: {e}")

        self.console_buffer = self.console_text.get_buffer()
        console_scrolled.add(self.console_text)

        console_label = Gtk.Label(label="Console Output")
        self.notebook.append_page(console_scrolled, console_label)

        image_scrolled = Gtk.ScrolledWindow()
        image_scrolled.set_policy(
            Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        image_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)

        self.image_display = Gtk.Image()
        image_box.pack_start(self.image_display, True, True, 0)

        image_button_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        image_button_box.set_halign(Gtk.Align.CENTER)

        self.delete_image_button = Gtk.Button(label="Delete Image")
        self.delete_image_button.connect(
            "clicked", self.on_delete_image_clicked)
        self.delete_image_button.set_sensitive(False)
        image_button_box.pack_start(self.delete_image_button, False, False, 0)

        image_box.pack_start(image_button_box, False, False, 5)

        image_scrolled.add(image_box)

        image_label = Gtk.Label(label="Generated Image")
        self.notebook.append_page(image_scrolled, image_label)

        self.notebook.set_current_page(0)

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = ConsoleRedirector(self.console_text, self.original_stdout)
        sys.stderr = ConsoleRedirector(self.console_text, self.original_stderr)

        self.load_settings()

    def update_status(self, message):
        if hasattr(sys.stdout, 'write_with_newline'):
            sys.stdout.write_with_newline(f"{message}\n")
        else:
            print(f"{message}\n", end='')

        GLib.idle_add(self._update_status_idle, message)

    def _update_status_idle(self, message):
        first_line = message.split('\n')[0]
        self.status_label.set_text(first_line)
        return False

    def count_tokens(self, text):
        if not self.tokenizer or not text:
            return 0
        try:
            tokens = self.tokenizer.encode(text)
            token_count = len(tokens) - 2
            return max(0, token_count)
        except Exception as e:
            print(f"Error counting tokens: {e}")
            return 0

    def on_text_changed(self, widget=None):
        if not self.tokenizer:
            return

        trigger_buffer = self.trigger_text.get_buffer()
        trigger_text = trigger_buffer.get_text(
            trigger_buffer.get_start_iter(),
            trigger_buffer.get_end_iter(),
            False
        )

        prompt_buffer = self.prompt_text.get_buffer()
        prompt_text = prompt_buffer.get_text(
            prompt_buffer.get_start_iter(),
            prompt_buffer.get_end_iter(),
            False
        )

        neg_buffer = self.neg_prompt_text.get_buffer()
        neg_text = neg_buffer.get_text(
            neg_buffer.get_start_iter(),
            neg_buffer.get_end_iter(),
            False
        )

        trigger_tokens = self.count_tokens(trigger_text)

        if trigger_text and prompt_text:
            combined_prompt = f"{trigger_text}, {prompt_text}"
        elif trigger_text:
            combined_prompt = trigger_text
        else:
            combined_prompt = prompt_text
        combined_tokens = self.count_tokens(combined_prompt)

        neg_tokens = self.count_tokens(neg_text)

        self._update_token_label(self.trigger_token_label, trigger_tokens)
        self._update_token_label(self.prompt_token_label, combined_tokens)
        self._update_token_label(self.neg_prompt_token_label, neg_tokens)

    def _get_model_type_index(self):
        index = self.model_type_combo.get_active()
        return index if index >= 0 else MODEL_TYPE_SDXL

    def _update_token_label(self, label, count):
        label.set_text(f"{count}/{TOKEN_LIMIT}")
        if count > TOKEN_LIMIT:
            label.set_markup(
                f'<span foreground="red" weight="bold">{count}/{TOKEN_LIMIT}</span>')
        elif count > TOKEN_WARNING_THRESHOLD:
            label.set_markup(
                f'<span foreground="orange" weight="bold">{count}/{TOKEN_LIMIT}</span>')
        else:
            label.set_markup(
                f'<span foreground="green">{count}/{TOKEN_LIMIT}</span>')

    def _set_default_lora(self):
        model_type_index = self._get_model_type_index()
        self.lora_entries[0].set_text('ByteDance/Hyper-SD')
        weight_name_entry, weight_spin = self.lora_weight_entries[0]
        if model_type_index == MODEL_TYPE_SD15:
            weight_name_entry.set_text('Hyper-SD15-4steps-lora.safetensors')
        else:
            weight_name_entry.set_text('Hyper-SDXL-4steps-lora.safetensors')

    def load_settings(self):
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, 'r') as f:
                    settings = json.load(f)

                if 'model_type' in settings:
                    self.model_type_combo.set_active(settings['model_type'])

                if 'model' in settings:
                    self.model_entry.set_text(settings['model'])

                if 'loras' in settings:
                    for i, lora_data in enumerate(settings['loras'][:4]):
                        if i < len(self.lora_entries):
                            self.lora_entries[i].set_text(
                                lora_data.get('path', ''))
                            weight_name_entry, weight_spin = self.lora_weight_entries[i]
                            weight_name_entry.set_text(
                                lora_data.get('weight_name', ''))
                            weight_spin.set_value(
                                lora_data.get('weight_value', 1.0))
                else:
                    self._set_default_lora()

                if 'steps' in settings:
                    self.steps_spin.set_value(settings['steps'])
                if 'guidance' in settings:
                    self.guidance_spin.set_value(settings['guidance'])
                if 'eta' in settings:
                    self.eta_spin.set_value(settings['eta'])
                if 'clip_skip' in settings:
                    self.clip_skip_spin.set_value(settings['clip_skip'])
                if 'trigger' in settings:
                    trigger_buffer = self.trigger_text.get_buffer()
                    trigger_buffer.set_text(settings['trigger'])
                if 'prompt' in settings:
                    prompt_buffer = self.prompt_text.get_buffer()
                    prompt_buffer.set_text(settings['prompt'])
                if 'negative_prompt' in settings:
                    neg_buffer = self.neg_prompt_text.get_buffer()
                    neg_buffer.set_text(settings['negative_prompt'])
                if 'dtype' in settings:
                    self.dtype_combo.set_active(settings['dtype'])
                if 'fuse_lora' in settings:
                    self.fuse_lora_check.set_active(settings['fuse_lora'])
            else:
                self._set_default_lora()
        except Exception as e:
            print(f"Error loading settings: {e}")

        self.on_text_changed(None)

    def save_settings(self):
        try:
            trigger_buffer = self.trigger_text.get_buffer()
            trigger = trigger_buffer.get_text(
                trigger_buffer.get_start_iter(),
                trigger_buffer.get_end_iter(),
                False
            )

            prompt_buffer = self.prompt_text.get_buffer()
            prompt = prompt_buffer.get_text(
                prompt_buffer.get_start_iter(),
                prompt_buffer.get_end_iter(),
                False
            )

            neg_buffer = self.neg_prompt_text.get_buffer()
            negative_prompt = neg_buffer.get_text(
                neg_buffer.get_start_iter(),
                neg_buffer.get_end_iter(),
                False
            )

            settings = {
                'model_type': self.model_type_combo.get_active(),
                'model': self.model_entry.get_text(),
                'loras': [],
                'steps': int(self.steps_spin.get_value()),
                'guidance': float(self.guidance_spin.get_value()),
                'eta': float(self.eta_spin.get_value()),
                'clip_skip': int(self.clip_skip_spin.get_value()),
                'trigger': trigger,
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'dtype': self.dtype_combo.get_active(),
                'fuse_lora': self.fuse_lora_check.get_active()
            }

            for i, (lora_entry, (weight_name_entry, weight_spin)) in enumerate(zip(self.lora_entries, self.lora_weight_entries)):
                lora_data = {
                    'path': lora_entry.get_text(),
                    'weight_name': weight_name_entry.get_text(),
                    'weight_value': weight_spin.get_value()
                }
                settings['loras'].append(lora_data)

            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

            with open(CONFIG_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def on_browse_lora(self, button, lora_index):
        dialog = Gtk.FileChooserDialog(
            title="Select LoRA Weight File or Directory",
            parent=self,
            action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            "Select Current Folder", Gtk.ResponseType.APPLY,
            Gtk.STOCK_OPEN, Gtk.ResponseType.OK
        )

        filter_safetensors = Gtk.FileFilter()
        filter_safetensors.set_name("SafeTensors files")
        filter_safetensors.add_pattern("*.safetensors")
        dialog.add_filter(filter_safetensors)

        filter_all = Gtk.FileFilter()
        filter_all.set_name("All files")
        filter_all.add_pattern("*")
        dialog.add_filter(filter_all)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            selected_path = dialog.get_filename()
            if not selected_path:
                dialog.destroy()
                return

            path_obj = Path(selected_path)

            if path_obj.is_file():
                weight_name_entry, _ = self.lora_weight_entries[lora_index]
                parent_dir = path_obj.parent

                if not is_huggingface_repo(str(parent_dir)):
                    weight_stem = path_obj.stem
                    lora_subdir = LORA_DIR / weight_stem
                    lora_subdir.mkdir(parents=True, exist_ok=True)

                    dest_file = lora_subdir / path_obj.name

                    try:
                        if not dest_file.exists() or dest_file.stat().st_size != path_obj.stat().st_size:
                            print(
                                f"Copying LoRA weight file to config: {path_obj.name}")
                            shutil.copy2(path_obj, dest_file)

                        self.lora_entries[lora_index].set_text(
                            str(lora_subdir))
                        weight_name_entry.set_text(path_obj.name)
                    except Exception as e:
                        print(f"Failed to copy LoRA weight file: {e}")
                        print(
                            "Please verify you have write permissions and try again.")
                        weight_name_entry.set_text(path_obj.name)
                        self.lora_entries[lora_index].set_text(str(parent_dir))
                else:
                    weight_name_entry.set_text(path_obj.name)
                    self.lora_entries[lora_index].set_text(str(parent_dir))
            elif path_obj.is_dir():
                if not is_huggingface_repo(selected_path):
                    copied_path = copy_lora_to_config(selected_path)
                    if copied_path:
                        self.lora_entries[lora_index].set_text(
                            str(copied_path))
                    else:
                        self.lora_entries[lora_index].set_text(selected_path)
                else:
                    self.lora_entries[lora_index].set_text(selected_path)
            else:
                print(
                    f"Warning: Selected path is neither a file nor directory: {selected_path}")
        elif response == Gtk.ResponseType.APPLY:
            current_folder = dialog.get_current_folder()
            if current_folder:
                if not is_huggingface_repo(current_folder):
                    copied_path = copy_lora_to_config(current_folder)
                    if copied_path:
                        self.lora_entries[lora_index].set_text(
                            str(copied_path))
                    else:
                        self.lora_entries[lora_index].set_text(current_folder)
                else:
                    self.lora_entries[lora_index].set_text(current_folder)

        dialog.destroy()

    def on_clear_lora(self, button, lora_index):
        self.lora_entries[lora_index].set_text("")
        weight_name_entry, weight_spin = self.lora_weight_entries[lora_index]
        weight_name_entry.set_text("")
        weight_spin.set_value(1.0)

    def on_load_clicked(self, button):
        if self.generating or self.loading_model:
            return

        self.loading_model = True
        self.stop_event.clear()
        self.stop_click_count = 0
        self.load_button.set_sensitive(False)
        self.generate_button.set_sensitive(False)
        self.stop_button.set_sensitive(True)
        self.stop_button.show()

        if self.pipe is not None:
            self.update_status(
                "Reloading model... This may take a while. Click Stop to cancel.")
        else:
            self.update_status(
                "Loading model... This may take a while. Click Stop to cancel.")

        self.load_thread = threading.Thread(target=self.load_model_thread)
        self.load_thread.start()

    def _check_stop_loading(self, cleanup_pipe=False):
        if self.stop_event.is_set():
            self.update_status("Interrupted by user.")
            if cleanup_pipe and self.pipe is not None:
                self.pipe = None
                gc.collect()
            return True
        return False

    def load_model_thread(self):
        try:
            if self._check_stop_loading():
                return

            model_name = self.model_entry.get_text().strip()
            dtype_index = self.dtype_combo.get_active()
            model_type_index = self.model_type_combo.get_active()

            if dtype_index == 0:
                dtype = torch.float32
                use_hybrid = False
            elif dtype_index == 1:
                dtype = torch.float16
                use_hybrid = False
            else:
                dtype = torch.float16
                use_hybrid = True

            if self.pipe is not None:
                if self._check_stop_loading():
                    return
                self.update_status("Cleaning up existing model...")
                if self.loras_fused:
                    self.update_status("Unfusing previous LoRAs...")
                    self.pipe.unfuse_lora()
                    self.loras_fused = False
                del self.pipe
                self.pipe = None
                gc.collect()

            if self._check_stop_loading():
                return

            self.update_status(f"Loading model: {model_name}...")

            if model_type_index == MODEL_TYPE_SD15:
                PipelineClass = StableDiffusionPipeline
            else:
                PipelineClass = StableDiffusionXLPipeline

            if is_direct_url(model_name):
                if self._check_stop_loading():
                    return

                normalized_url = normalize_huggingface_url(model_name)
                if normalized_url != model_name:
                    print(f"Normalized URL: {model_name} -> {normalized_url}")

                self.update_status(
                    f"Loading model from URL using from_single_file...")
                try:
                    self.pipe = PipelineClass.from_single_file(
                        normalized_url,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        safety_checker=None
                    )
                except Exception as e:
                    error_msg = str(e)
                    if "404" in error_msg or "Not Found" in error_msg:
                        raise Exception(
                            f"URL not found (404): {model_name}. Please check the URL is correct.")
                    elif "safetensors" in error_msg.lower():
                        raise Exception(
                            f"Failed to load safetensors file from URL. The file may not be in safetensors format or may be corrupted: {error_msg}")
                    elif "connect" in error_msg.lower() or "network" in error_msg.lower():
                        raise Exception(
                            f"Network error while downloading model from URL. Check your internet connection: {error_msg}")
                    else:
                        raise Exception(
                            f"Failed to load model from URL: {error_msg}")
            else:
                if self._check_stop_loading():
                    return

                self.pipe = PipelineClass.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    local_files_only=False,
                    low_cpu_mem_usage=True,
                    safety_checker=None
                )

            if self._check_stop_loading(cleanup_pipe=True):
                return

            gc.collect()

            self.update_status("Moving pipeline to CPU...")
            self.pipe = self.pipe.to("cpu")
            gc.collect()

            if self._check_stop_loading(cleanup_pipe=True):
                return

            if use_hybrid:
                self.update_status(
                    "Applying hybrid precision (converting to float32)...")
                self._apply_hybrid_precision()

            if self._check_stop_loading(cleanup_pipe=True):
                return

            self.update_status("Model loaded. Loading LoRAs...")

            loras_to_load = []
            for i, (lora_entry, (weight_name_entry, weight_spin)) in enumerate(zip(self.lora_entries, self.lora_weight_entries)):
                lora_path = lora_entry.get_text().strip()
                weight_name = weight_name_entry.get_text().strip()
                weight_value = weight_spin.get_value()
                if lora_path and weight_name:
                    loras_to_load.append({
                        "path": lora_path,
                        "weight_name": weight_name,
                        "weight_value": weight_value,
                        "adapter_name": f"lora_{i}"
                    })

            if loras_to_load:
                lora_kwargs = {}
                for lora_info in loras_to_load:
                    if self._check_stop_loading(cleanup_pipe=True):
                        return

                    self.update_status(
                        f"Loading LoRA: {lora_info['path']}/{lora_info['weight_name']}...")
                    try:
                        self.pipe.load_lora_weights(
                            lora_info["path"],
                            weight_name=lora_info["weight_name"],
                            adapter_name=lora_info["adapter_name"]
                        )
                        lora_kwargs[lora_info["adapter_name"]
                                    ] = lora_info["weight_value"]
                    except Exception as e:
                        self.update_status(
                            f"Warning: Could not load LoRA {lora_info['adapter_name']}: {str(e)} - Check path and weight name. Skipping.")

                if self._check_stop_loading(cleanup_pipe=True):
                    return

                if lora_kwargs:
                    fuse_lora_enabled = self.fuse_lora_check.get_active()
                    if fuse_lora_enabled:
                        self.update_status(f"Fusing {len(lora_kwargs)} LoRA(s)...")
                        self.pipe.fuse_lora(
                            lora_scale=1.0, lora_kwargs=lora_kwargs)
                        self.loras_fused = True
                    else:
                        self.update_status(f"Setting weights for {len(lora_kwargs)} LoRA(s)...")
                        adapter_names = list(lora_kwargs.keys())
                        adapter_weights = list(lora_kwargs.values())
                        self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
                        self.loras_fused = False

            if self._check_stop_loading(cleanup_pipe=True):
                return

            self.update_status("Setting up scheduler...")
            if model_type_index == MODEL_TYPE_SD15:
                self.pipe.scheduler = LCMScheduler.from_config(
                    self.pipe.scheduler.config)
            else:
                self.pipe.scheduler = TCDScheduler.from_config(
                    self.pipe.scheduler.config)

            if self._check_stop_loading(cleanup_pipe=True):
                return

            self.update_status("Enabling memory optimizations...")
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()

            GLib.idle_add(self._enable_generate_and_load)
            self.update_status(
                "Ready! You can now generate images or reload with different LoRAs.")

        except KeyboardInterrupt:
            self.update_status("Interrupted by user.")
            if self.pipe is not None:
                self.pipe = None
                gc.collect()
            GLib.idle_add(self._enable_load)
        except Exception as e:
            if self.stop_event.is_set():
                self.update_status("Interrupted by user.")
            else:
                self.update_status(f"Error loading model: {str(e)}")
            GLib.idle_add(self._enable_load)
        finally:
            self.loading_model = False
            GLib.idle_add(self._hide_stop_button)

    def _apply_hybrid_precision(self):
        try:
            self.pipe = self.pipe.to(dtype=torch.float32)
            gc.collect()
        except Exception as e:
            print(f"Warning: Could not fully apply hybrid precision: {e}")

    def _enable_generate_and_load(self):
        self.generate_button.set_sensitive(True)
        self.load_button.set_sensitive(True)
        self.load_button.set_label("Reload Model & LoRAs")
        self.stop_button.hide()
        self.stop_button.set_sensitive(False)
        return False

    def _enable_generate(self):
        self.generate_button.set_sensitive(True)
        return False

    def _reset_generate_button(self):
        self.generate_button.set_sensitive(True)
        self.generate_button.show()
        self.stop_button.hide()
        self.stop_button.set_sensitive(False)
        return False

    def _enable_load(self):
        self.load_button.set_sensitive(True)
        self.stop_button.hide()
        self.stop_button.set_sensitive(False)
        return False

    def _hide_stop_button(self):
        self.stop_button.hide()
        self.stop_button.set_sensitive(False)
        return False

    def on_generate_clicked(self, button):
        if self.generating or self.loading_model or self.pipe is None:
            return

        self.save_settings()

        self.generating = True
        self.stop_event.clear()
        self.stop_click_count = 0
        self.generate_button.set_sensitive(False)
        self.generate_button.hide()
        self.stop_button.set_sensitive(True)
        self.stop_button.show()

        self.generation_thread = threading.Thread(
            target=self.generate_image_thread)
        self.generation_thread.start()

    def on_stop_clicked(self, button):
        if self.generating or self.loading_model:
            self.stop_event.set()
            self.stop_click_count += 1

            if self.generating:
                self.update_status(
                    f"FORCE STOP #{self.stop_click_count} - Sending interrupt...")
                print(
                    f"\n*** STOP BUTTON CLICKED #{self.stop_click_count} - Sending interrupt... ***")
            else:
                self.update_status(
                    f"FORCE STOP #{self.stop_click_count} - Sending interrupt...")
                print(
                    f"\n*** STOP BUTTON CLICKED #{self.stop_click_count} - Sending interrupt... ***")

            target_thread = self.generation_thread if self.generating else self.load_thread
            if target_thread and target_thread.is_alive():
                success = raise_exception_in_thread(target_thread)
                if success:
                    print(f"  → Sent KeyboardInterrupt to worker thread")
                else:
                    print(f"  → Thread interrupt failed")

            print(f"  → Sending SIGINT to process (PID {os.getpid()})")
            os.kill(os.getpid(), signal.SIGINT)

    def generate_image_thread(self):
        try:
            prompt_buffer = self.prompt_text.get_buffer()
            user_prompt = prompt_buffer.get_text(
                prompt_buffer.get_start_iter(),
                prompt_buffer.get_end_iter(),
                False
            ).strip()

            if not user_prompt:
                self.update_status("Error: Prompt cannot be empty!")
                GLib.idle_add(self._enable_generate)
                self.generating = False
                return

            neg_buffer = self.neg_prompt_text.get_buffer()
            negative_prompt = neg_buffer.get_text(
                neg_buffer.get_start_iter(),
                neg_buffer.get_end_iter(),
                False
            ).strip()

            trigger_buffer = self.trigger_text.get_buffer()
            trigger = trigger_buffer.get_text(
                trigger_buffer.get_start_iter(),
                trigger_buffer.get_end_iter(),
                False
            ).strip()

            if trigger:
                full_prompt = f"{trigger}, {user_prompt}"
            else:
                full_prompt = user_prompt

            steps = int(self.steps_spin.get_value())
            guidance = float(self.guidance_spin.get_value())
            eta = float(self.eta_spin.get_value())
            clip_skip = int(self.clip_skip_spin.get_value())

            self.update_status(
                f"Generating image with {steps} steps (eta={eta}, clip_skip={clip_skip})...")

            def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
                if self.stop_event.is_set():
                    self.update_status(
                        f"Stopping after step {step_index + 1}...")
                    pipe._interrupt = True
                return callback_kwargs

            result = None
            with torch.inference_mode():
                result = self.pipe(
                    prompt=full_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    eta=eta,
                    clip_skip=clip_skip,
                    callback_on_step_end=callback_on_step_end,
                )

                image = getattr(result, 'images', [None])[
                    0] if result else None

            if result is not None:
                del result
            gc.collect()

            if self.stop_event.is_set():
                self.update_status("Interrupted by user.")
            elif image is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = OUTPUT_DIR / f"imagine_{timestamp}.png"
                image.save(str(output_path))

                GLib.idle_add(self._display_image, str(output_path))
                self.update_status(f"Done! Image saved to {output_path}.")
            else:
                self.update_status("No image produced.")

        except KeyboardInterrupt:
            self.update_status("Interrupted by user.")
            GLib.idle_add(self._reset_generate_button)
        except Exception as e:
            if self.stop_event.is_set():
                self.update_status("Interrupted by user.")
            else:
                self.update_status(f"Error generating image: {str(e)}")
        finally:
            self.generating = False
            GLib.idle_add(self._reset_generate_button)

    def _display_image(self, path):
        try:
            if not Path(path).exists():
                print(f"Error: Image file not found: {path}", file=sys.stderr)
                self._clear_image_display()
                return False
            pixbuf = GdkPixbuf.Pixbuf.new_from_file(path)
            self.image_display.set_from_pixbuf(pixbuf)
            self.current_image_path = path
            self.delete_image_button.set_sensitive(True)
            self.notebook.set_current_page(1)
        except Exception as e:
            print(f"Error displaying image: {e}", file=sys.stderr)
        return False

    def _clear_image_display(self):
        self.image_display.clear()
        self.current_image_path = None
        self.delete_image_button.set_sensitive(False)

    def on_delete_image_clicked(self, button):
        if self.current_image_path:
            try:
                if Path(self.current_image_path).exists():
                    os.remove(self.current_image_path)
                    print(f"Deleted image: {self.current_image_path}")
                    self.update_status("Image deleted successfully.")
                else:
                    print(
                        f"Image file no longer exists: {self.current_image_path}")
                    self.update_status("Image file no longer exists.")
                self._clear_image_display()
            except Exception as e:
                error_msg = f"Error deleting image: {e}"
                print(error_msg, file=sys.stderr)
                self.update_status(error_msg)

    def on_restore_defaults_clicked(self, button):
        model_type_index = self._get_model_type_index()

        if model_type_index == MODEL_TYPE_SD15:
            self.model_entry.set_text("digiplay/Photon_v1")
        else:
            self.model_entry.set_text("segmind/SSD-1B")

        self.dtype_combo.set_active(0)

        self.lora_entries[0].set_text('ByteDance/Hyper-SD')
        weight_name_entry, weight_spin = self.lora_weight_entries[0]
        if model_type_index == MODEL_TYPE_SD15:
            weight_name_entry.set_text('Hyper-SD15-4steps-lora.safetensors')
        else:
            weight_name_entry.set_text('Hyper-SDXL-4steps-lora.safetensors')
        weight_spin.set_value(1.0)

        for i in range(1, 4):
            self.lora_entries[i].set_text("")
            weight_name_entry, weight_spin = self.lora_weight_entries[i]
            weight_name_entry.set_text("")
            weight_spin.set_value(1.0)

        self.steps_spin.set_value(4)

        self.guidance_spin.set_value(7.5)

        self.eta_spin.set_value(0.3)

        self.clip_skip_spin.set_value(2)

        trigger_buffer = self.trigger_text.get_buffer()
        trigger_buffer.set_text("")

        prompt_buffer = self.prompt_text.get_buffer()
        prompt_buffer.set_text("")

        neg_buffer = self.neg_prompt_text.get_buffer()
        neg_buffer.set_text("")

        self.on_text_changed()

        self.update_status("Settings restored to defaults.")

    def on_window_close(self, widget, event):
        self.save_settings()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self._cleanup_threads()
        return False

    def _cleanup_threads(self):
        if self.generating or self.loading_model:
            self.stop_event.set()
            self.update_status("Waiting for operations to stop...")

        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=GENERATION_THREAD_TIMEOUT)

        if self.load_thread and self.load_thread.is_alive():
            self.load_thread.join(timeout=LOAD_THREAD_TIMEOUT)


def main():
    global _gui_instance

    def sigint_handler(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, sigint_handler)

    win = ImagineGUI()
    _gui_instance = win
    win.connect("delete-event", win.on_window_close)
    win.connect("destroy", Gtk.main_quit)
    win.show_all()

    try:
        Gtk.main()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received - shutting down...")
        if _gui_instance:
            if _gui_instance.generating or _gui_instance.loading_model:
                print("Stopping operations...")
                _gui_instance.stop_event.set()
        sys.exit(0)


if __name__ == "__main__":
    main()
