import os
import sys # For finding executable path when bundled
import subprocess
import threading
import datetime
import re
import time
import io 
import queue 
import random 
import json # For storing API key in a config file

from groq import Groq
from groq.types.chat import ChatCompletion # For type hinting if needed
from groq import AuthenticationError, APIConnectionError # To catch specific errors

import tkinter as tk
from tkinter import scrolledtext
from tkinter import filedialog, messagebox, ttk, simpledialog 

import pyaudiowpatch as pa 
import wave as wav_writer

# --- Application Configuration File ---
APP_NAME = "MeetingTranscriberRealtime" 
def get_app_data_dir():
    if os.name == 'win32': 
        path = os.path.join(os.getenv('LOCALAPPDATA', os.path.expanduser("~")), APP_NAME)
    elif os.name == 'darwin': 
        path = os.path.join(os.path.expanduser("~"), "Library", "Application Support", APP_NAME)
    else: 
        path = os.path.join(os.getenv('XDG_DATA_HOME', os.path.join(os.path.expanduser("~"), ".local", "share")), APP_NAME)
    os.makedirs(path, exist_ok=True)
    return path

CONFIG_FILE_PATH = os.path.join(get_app_data_dir(), "config.json")

# --- Summary Prompts ---
PROMPT_INTERVIEW = (
    "You are an expert interview performance analyzer. Given a complete interview transcript, you will:\n\n"
    "1. Conduct comprehensive interviewee performance analysis evaluating questioning techniques, communication skills, and overall effectiveness.\n"
    "2. Identify strengths and weaknesses with specific examples from the transcript.\n"
    "3. Highlight key successful moments where the interviewee excelled (e.g., probing questions, building rapport, handling difficult responses).\n"
    "4. Pinpoint areas needing improvement with concrete instances where opportunities were missed.\n"
    "5. Provide actionable recommendations for enhancing future interview performance.\n\n"
    "Analysis should cover:\n"
    "• Question quality and structure (open vs. closed, follow-up effectiveness)\n"
    "• Active listening and response to candidate answers\n"
    "• Interview flow and time management\n"
    "• Bias detection and fairness assessment\n"
    "• Rapport building and candidate experience\n\n"
    "Output format:\n"
    "• Executive summary of overall performance\n"
    "• Strengths (with transcript examples)\n"
    "• Areas for improvement (with specific instances)\n"
    "• Top 3-5 actionable recommendations\n"
    "• Performance rating or score (if applicable)\n\n"
    "Focus on constructive feedback that helps the interviewee develop better skills for future interviews.\n"
)

PROMPT_MEETING = (
    "You are an expert Scrum meeting note-taker and summarizer. After each meeting, you will:\n"
    "1. Generate a concise meeting summary with clear structure and key discussion points.\n"
    "2. Extract and list all action items with assigned owners and due dates (if mentioned).\n"
    "3. Create formal meeting minutes including attendees, decisions made, and next steps.\n"
    "4. Identify tasks specifically assigned to me and highlight my responsibilities.\n"
    "5. Present everything in a well-organized format using headers, bullet points, and clear sections.\n\n"
    "Output format should include:\n"
    "• Meeting overview and key outcomes\n"
    "• Action items (What | Who | When)\n"
    "• My assigned tasks and deadlines\n"
    "• Important decisions or blockers discussed\n"
    "• Next meeting date/focus\n\n"
    "Keep the summary brief but comprehensive, focusing on actionable information rather than detailed conversation transcripts.\n"
)

# --- API Key Management ---
client = None 
GROQ_API_KEY_STORED = None

def _validate_api_key(api_key_to_test, temp_client):
    """Attempts a lightweight API call to validate the key."""
    if not temp_client: # Should not happen if api_key_to_test is provided
        return False
    try:
        # A simple call, e.g., listing models.
        # This might change depending on the Groq SDK's capabilities.
        # If models.list() is too heavy or not available without more setup,
        # a very short, dummy transcription could be an alternative, but that's less ideal.
        temp_client.models.list() # This is a common way to check API key validity
        print("API Key validation call successful.")
        return True
    except AuthenticationError:
        print("API Key Validation Failed: AuthenticationError (Invalid Key)")
        messagebox.showerror("API Key Error", "The provided Groq API Key is invalid. Please check the key and try again.", parent=root_gui if root_gui else None)
        return False
    except APIConnectionError as e:
        print(f"API Key Validation Failed: APIConnectionError - {e}")
        messagebox.showerror("API Connection Error", f"Could not connect to Groq API to validate the key. Please check your internet connection.\nError: {e}", parent=root_gui if root_gui else None)
        return False
    except Exception as e: # Catch any other Groq or network errors during validation
        print(f"API Key Validation Failed: Unexpected error - {e}")
        messagebox.showerror("API Key Error", f"An unexpected error occurred while validating the API key: {e}", parent=root_gui if root_gui else None)
        return False

def load_api_key_from_config():
    global GROQ_API_KEY_STORED, client
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, 'r') as f:
                config = json.load(f)
                loaded_key = config.get("GROQ_API_KEY")
            if loaded_key:
                temp_client = Groq(api_key=loaded_key) # Initialize a temporary client for validation
                if _validate_api_key(loaded_key, temp_client):
                    GROQ_API_KEY_STORED = loaded_key
                    client = temp_client # Use the validated client
                    print("Groq API key loaded and validated from config.")
                    return True
                else:
                    # Validation failed, so don't use this key or client
                    GROQ_API_KEY_STORED = None
                    client = None
                    # Optionally, remove the invalid key from config
                    # save_api_key_to_config(None) # This would clear it
                    return False 
    except Exception as e:
        print(f"Error loading API key from config: {e}")
    GROQ_API_KEY_STORED = None # Ensure reset if any error
    client = None
    return False

def save_api_key_to_config(api_key):
    global GROQ_API_KEY_STORED, client
    if not api_key or not api_key.strip(): # Handle empty or whitespace-only keys
        try:
            if os.path.exists(CONFIG_FILE_PATH): os.remove(CONFIG_FILE_PATH)
            GROQ_API_KEY_STORED = None
            client = None
            print("API Key cleared from config.")
            return True # Successfully "saved" an empty key (i.e., removed it)
        except Exception as e:
            print(f"Error clearing API key from config: {e}")
            messagebox.showerror("Error", f"Could not clear API key: {e}", parent=root_gui if root_gui else None)
            return False

    try:
        temp_client = Groq(api_key=api_key) # Initialize a temporary client for validation
        if _validate_api_key(api_key, temp_client):
            with open(CONFIG_FILE_PATH, 'w') as f:
                json.dump({"GROQ_API_KEY": api_key}, f)
            GROQ_API_KEY_STORED = api_key
            client = temp_client # Use the validated client
            print("Groq API key validated, saved to config, and client initialized.")
            return True
        else:
            # Validation failed, client and GROQ_API_KEY_STORED will be None from _validate_api_key's side effects or here
            client = None 
            GROQ_API_KEY_STORED = None
            return False # Indicate save failed due to invalid key
    except Exception as e: # Catch errors during file writing
        print(f"Error saving API key to config file: {e}")
        messagebox.showerror("Error", f"Could not save API key to configuration file: {e}", parent=root_gui if root_gui else None)
        client = None # Ensure client is None if save fails
        GROQ_API_KEY_STORED = None
    return False

def prompt_and_set_api_key(parent_window=None):
    global client, GROQ_API_KEY_STORED # client is now directly affected by save_api_key_to_config
    
    current_key_partial = ""
    if GROQ_API_KEY_STORED and len(GROQ_API_KEY_STORED) > 10:
        current_key_partial = f"{GROQ_API_KEY_STORED[:5]}...{GROQ_API_KEY_STORED[-5:]}"
    elif GROQ_API_KEY_STORED:
        current_key_partial = "Set (short key)"
    else:
        current_key_partial = "Not Set"

    msg_current_key = f"Current API Key: {current_key_partial}\n\n"

    # Ask if user wants to update even if key is set and client is valid
    if GROQ_API_KEY_STORED: # Check if a key is stored, not if client is valid yet
         if not messagebox.askyesno("Update API Key?", 
                                   f"{msg_current_key}Do you want to enter/update your Groq API Key?", 
                                   parent=parent_window):
            return client is not None # Return current client status (might be valid from load)

    new_key_input = simpledialog.askstring("Set Groq API Key", 
                                     f"{msg_current_key}Please enter your Groq API Key (leave blank to clear):", 
                                     parent=parent_window)

    if new_key_input is not None: # User didn't press Cancel
        if new_key_input.strip(): # User entered something
            if save_api_key_to_config(new_key_input.strip()):
                messagebox.showinfo("API Key Set", "Groq API Key has been configured and validated successfully.", parent=parent_window)
            # save_api_key_to_config will show an error if validation or save fails
        else: # User entered blank string, intending to clear
            if save_api_key_to_config(None): # Pass None to clear
                 messagebox.showinfo("API Key Cleared", "Groq API Key has been cleared.", parent=parent_window)
            # save_api_key_to_config handles errors if clearing fails

    return client is not None # Return true if client is now validly initialized


# --- FFmpeg Path Configuration (Store App Friendly) ---
FFMPEG_CMD = None
def configure_ffmpeg():
    global FFMPEG_CMD
    # Determine base path whether running as script or PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle (e.g., by PyInstaller)
        application_path = os.path.dirname(sys.executable)
    else:
        # If run as a normal Python script
        application_path = os.path.dirname(os.path.abspath(__file__))

    # Paths to check for bundled ffmpeg
    bundled_paths_to_check = [
        os.path.join(application_path, "ffmpeg.exe"),                # ffmpeg.exe in the same directory as the app
        os.path.join(application_path, "ffmpeg_bin", "ffmpeg.exe")   # ffmpeg.exe in a 'ffmpeg_bin' subdirectory
    ]

    for path_to_check in bundled_paths_to_check:
        if os.path.exists(path_to_check):
            FFMPEG_CMD = path_to_check
            print(f"Using bundled FFmpeg from: {FFMPEG_CMD}")
            return

    # Fallback to system PATH if not bundled (less ideal for Store, but good for dev)
    FFMPEG_CMD = "ffmpeg"
    try:
        subprocess.run([FFMPEG_CMD, '-version'], capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        print("Using FFmpeg from system PATH (bundled version not found).")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: FFmpeg not found in bundled locations or system PATH. Application may not function correctly.")
        FFMPEG_CMD = None # Explicitly set to None if not found
configure_ffmpeg() # Call at startup


# --- Globals for Real-Time Recording ---
# ... (Keep all your existing real-time globals) ...
ffmpeg_mic_process_rt = None; mic_capture_thread_rt = None; stop_mic_capture_event_rt = threading.Event(); mic_audio_processor_thread_rt = None; current_mic_time_offset_rt = 0.0
system_audio_capture_thread_rt = None; stop_system_audio_event_rt = threading.Event(); system_audio_processor_thread_rt = None; current_system_time_offset_rt = 0.0
AUDIO_CHUNK_DURATION_SECONDS = 10; AUDIO_SAMPLE_RATE = 16000; AUDIO_CHANNELS_MIC = 1; AUDIO_SAMPLE_WIDTH_MIC = 2
mic_audio_chunk_queue_rt = queue.Queue(); system_audio_chunk_queue_rt = queue.Queue(); transcript_display_queue_rt = queue.Queue()
all_transcribed_segments_rt = []; all_segments_lock_rt = threading.Lock(); is_recording_realtime_flag = False; root_gui = None
MIN_TIME_BETWEEN_API_CALLS_SECONDS = 3.5; last_api_call_time_rt = 0; api_call_lock_rt = threading.Lock()
last_saved_transcript_path = None
live_transcript_widget = None
last_generated_summary_text = None
MIN_TIME_BETWEEN_SUMMARY_CALLS_SECONDS = 2.5; last_summary_call_time = 0; summarize_api_lock = threading.Lock()

# --- Output Directory Helper ---
def get_application_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def get_transcript_output_dir():
    base = get_application_base_path()
    return os.path.join(base, "Interview Calls Transcript")

def read_full_transcript_text_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ""

def get_summary_prompt(selection_text):
    sel = (selection_text or "").strip().lower()
    if "meeting" in sel:
        return PROMPT_MEETING
    return PROMPT_INTERVIEW

def generate_summary_with_groq(transcript_text, system_prompt):
    global client
    if not client or not transcript_text.strip():
        return None
    # If short enough, single-pass; otherwise map-reduce chunking
    MAX_CHARS_SINGLE_PASS = 24000
    if len(transcript_text) <= MAX_CHARS_SINGLE_PASS:
        single = _call_groq_chat_with_retry(system_prompt, transcript_text)
        if single == "__SPLIT_REQUIRED__" or single is None:
            # Fall back to chunking with small chunk sizes
            return _summarize_via_chunks(transcript_text, system_prompt, 15000, 20000)
        return single
    # Map-Reduce pipeline for long text
    return _summarize_via_chunks(transcript_text, system_prompt, 15000, 20000)

def _summarize_via_chunks(text, system_prompt, target_chunk_chars, hard_max_chars):
    chunks = _split_text_into_chunks(text, target_chunk_chars=target_chunk_chars, hard_max_chars=hard_max_chars)
    chunk_summaries = []
    for idx, chunk in enumerate(chunks, start=1):
        head = f"Part {idx}/{len(chunks)} of transcript. Summarize this part with the same instructions.\n\n"
        part_summary = _call_groq_chat_with_retry(system_prompt, head + chunk)
        if part_summary == "__SPLIT_REQUIRED__":
            # Further split this chunk and summarize sub-parts
            sub = _split_text_into_chunks(chunk, target_chunk_chars=max(8000, target_chunk_chars//2), hard_max_chars=max(10000, hard_max_chars//2))
            sub_summaries = []
            for jdx, subchunk in enumerate(sub, start=1):
                head2 = f"Subpart {jdx}/{len(sub)} of part {idx}. Summarize this subpart.\n\n"
                sub_summary = _call_groq_chat_with_retry(system_prompt, head2 + subchunk)
                if not sub_summary or sub_summary == "__SPLIT_REQUIRED__":
                    sub_summary = f"[Warning] Subpart {jdx} of part {idx} could not be summarized due to size limits."
                sub_summaries.append(sub_summary)
            synth_prompt = system_prompt + "\n\nSynthesize the following sub-summaries into one coherent part summary."
            part_summary = _call_groq_chat_with_retry(synth_prompt, "\n\n--- SUB SUMMARY ---\n\n".join(sub_summaries))
            if not part_summary:
                part_summary = f"[Warning] Part {idx} summary unavailable due to API error."
        elif not part_summary:
            part_summary = f"[Warning] Part {idx} summary unavailable due to API error."
        chunk_summaries.append(part_summary)
    synthesis_prompt = (
        system_prompt +
        "\n\nYou received multiple partial summaries from different parts of the same transcript. "
        "Please synthesize them into a single coherent final summary without repeating sections, respecting the required output format."
    )
    combined_text = "\n\n--- PARTIAL SUMMARY ---\n\n".join(chunk_summaries)
    final_summary = _call_groq_chat_with_retry(synthesis_prompt, combined_text)
    return final_summary

def _call_groq_chat_with_retry(system_prompt, user_text, max_tokens=800, temperature=0.2):
    global client, summarize_api_lock, last_summary_call_time
    if not client:
        return None
    MAX_RETRIES = 5; BASE_DELAY = 1.5
    for attempt in range(1, MAX_RETRIES+1):
        try:
            # Simple rate-limit between summary calls
            with summarize_api_lock:
                now = time.monotonic(); elapsed = now - last_summary_call_time
                if elapsed < MIN_TIME_BETWEEN_SUMMARY_CALLS_SECONDS:
                    time.sleep(MIN_TIME_BETWEEN_SUMMARY_CALLS_SECONDS - elapsed)
                last_summary_call_time = time.monotonic()
            # If the message is very large, proactively return split requirement
            if _estimate_prompt_tokens(user_text) + max_tokens > 10000:
                return "__SPLIT_REQUIRED__"
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role":"system","content":system_prompt},
                    {"role":"user","content":user_text}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            if completion and completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content
            return None
        except Exception as e:
            msg = str(e).lower()
            if ("413" in msg or "request too large" in msg or "rate_limit_exceeded" in msg):
                return "__SPLIT_REQUIRED__"
            if any(code in msg for code in ["429", "rate limit", "too many requests", "service unavailable", "503"]):
                if attempt < MAX_RETRIES:
                    jitter = random.uniform(0, BASE_DELAY)
                    time.sleep((BASE_DELAY * (2 ** (attempt-1))) + jitter)
                    continue
            # For other errors or final failure
            messagebox.showerror("Summary Error", f"Failed to generate summary: {e}", parent=root_gui if root_gui else None)
            return None
    return None

def _split_text_into_chunks(text, target_chunk_chars=120000, hard_max_chars=160000):
    # Try to split on paragraph boundaries first
    paragraphs = text.split('\n\n')
    chunks = []
    current = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para) + 2
        if current_len + para_len <= target_chunk_chars:
            current.append(para)
            current_len += para_len
        else:
            if current:
                chunks.append('\n\n'.join(current))
            # If a single paragraph is gigantic, hard-split it
            if para_len > hard_max_chars:
                sub = _hard_split(para, hard_max_chars)
                chunks.extend(sub)
                current = []
                current_len = 0
            else:
                current = [para]
                current_len = para_len
    if current:
        chunks.append('\n\n'.join(current))
    return chunks

def _hard_split(s, max_chars):
    return [s[i:i+max_chars] for i in range(0, len(s), max_chars)]

def _estimate_prompt_tokens(text):
    # Heuristic: ~4 chars per token for English text
    length = len(text or "")
    return max(1, length // 4)

def save_summary_text(summary_text, summary_type_label):
    ts_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = get_transcript_output_dir(); os.makedirs(out_dir, exist_ok=True)
    base_name = "interview_summary" if "interview" in (summary_type_label or "").lower() else "meeting_summary"
    out_path = os.path.join(out_dir, f"{base_name}_{ts_file}.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    return out_path

def display_text_in_live_transcript(text_widget, text_to_show, clear_first=True):
    if not text_widget:
        return
    state = text_widget.cget("state"); text_widget.config(state=tk.NORMAL)
    if clear_first:
        text_widget.delete('1.0', tk.END)
    if text_to_show:
        text_widget.insert(tk.END, text_to_show)
    text_widget.config(state=state); text_widget.see(tk.END)

def summarize_transcript(selection_text, override_transcript_text=None):
    global last_saved_transcript_path, live_transcript_widget, last_generated_summary_text
    if not last_saved_transcript_path or not os.path.exists(last_saved_transcript_path):
        if not override_transcript_text:
            messagebox.showwarning("No Transcript", "Please record/stop or upload a transcript first.", parent=root_gui if root_gui else None)
            return
    transcript_text = override_transcript_text if override_transcript_text is not None else read_full_transcript_text_from_file(last_saved_transcript_path)
    if not transcript_text.strip():
        messagebox.showwarning("Empty Transcript", "Transcript file is empty.", parent=root_gui if root_gui else None)
        return
    prompt_text = get_summary_prompt(selection_text)
    summary = generate_summary_with_groq(transcript_text, prompt_text)
    if not summary:
        return
    # Show in Live Transcript box after clearing
    display_text_in_live_transcript(live_transcript_widget, summary, clear_first=True)
    last_generated_summary_text = summary
    messagebox.showinfo("Summary Generated", "Summary has been generated and displayed in the Live Transcript box.\n\nUse 'Save Summary As...' to save it to a file.", parent=root_gui if root_gui else None)

def upload_transcript_and_preview(text_widget):
    global last_saved_transcript_path
    init_dir = os.path.join(os.path.expanduser("~"), "Documents")
    try:
        os.makedirs(init_dir, exist_ok=True)
    except Exception:
        pass
    file_path = filedialog.askopenfilename(initialdir=init_dir, title="Open Transcript", filetypes=[("Text", "*.txt"), ("All Files", "*.*")])
    if not file_path:
        return
    last_saved_transcript_path = file_path
    content = read_full_transcript_text_from_file(file_path)
    display_text_in_live_transcript(text_widget, content, clear_first=True)

def save_live_transcript_as(text_widget):
    state = text_widget.cget("state"); text_widget.config(state=tk.NORMAL)
    content = text_widget.get('1.0', tk.END)
    text_widget.config(state=state)
    if not content.strip():
        messagebox.showwarning("Nothing to Save", "Live Transcript is empty.", parent=root_gui if root_gui else None)
        return
    init_dir = os.path.join(os.path.expanduser("~"), "Documents")
    try:
        os.makedirs(init_dir, exist_ok=True)
    except Exception:
        pass
    ts_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = filedialog.asksaveasfilename(initialdir=init_dir, title="Save As", defaultextension=".txt", initialfile=f"summary_{ts_file}.txt", filetypes=[("Text", "*.txt")])
    if not save_path:
        return
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(content)
        messagebox.showinfo("Saved", f"Saved to:\n{save_path}", parent=root_gui if root_gui else None)
    except Exception as e:
        messagebox.showerror("Error", f"Could not save: {e}", parent=root_gui if root_gui else None)


# --- (get_mic_device_dshow, transcribe_audio_chunk, capture_mic_audio_thread_func_rt,
#      system_audio_realtime_callback_rt, record_system_audio_pyaudiowpatch_realtime_thread_func_rt,
#      audio_processor_thread_func_rt, update_transcript_display_rt,
#      start_recording_realtime, stop_recording_realtime functions remain the same as your last working version
#      that included the "last batch fix" and "timestamp order fix") ---

def get_mic_device_dshow():
    if not FFMPEG_CMD: return None
    try:
        cmd = [FFMPEG_CMD, '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy']
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, check=True, creationflags=creation_flags)
        output_text = result.stderr; audio_device_names = []
        for line in output_text.splitlines():
            if "(audio)" in line.lower():
                match = re.search(r'"([^"]+)"', line)
                if match: audio_device_names.append(match.group(1))
        selected_mic_device = None
        if audio_device_names:
            selected_mic_device = next((d for d in audio_device_names if "external microphone" in d.lower()), None)
            if not selected_mic_device: selected_mic_device = next((d for d in audio_device_names if "microphone" in d.lower()), None)
            if not selected_mic_device:
                 selected_mic_device = next((d for d in audio_device_names if any(k in d.lower() for k in ["mic","input","capture"]) and not any(ex in d.lower() for ex in ["speaker","stereo mix","loopback","cable","output"])), None)
            if not selected_mic_device and audio_device_names: selected_mic_device = audio_device_names[0]
        return selected_mic_device
    except Exception: return None

def transcribe_audio_chunk(audio_data_bytes, channels, rate, sampwidth, speaker_label, attempt=1):
    global client, last_api_call_time_rt, api_call_lock_rt
    if not client: return [] 
    if not audio_data_bytes: return []
    MAX_RETRIES = 4; BASE_RETRY_DELAY = 1.5
    with api_call_lock_rt:
        current_time = time.monotonic(); time_since_last_call = current_time - last_api_call_time_rt
        if time_since_last_call < MIN_TIME_BETWEEN_API_CALLS_SECONDS: time.sleep(MIN_TIME_BETWEEN_API_CALLS_SECONDS - time_since_last_call)
        last_api_call_time_rt = time.monotonic()
    wav_buffer = io.BytesIO()
    try:
        with wav_writer.open(wav_buffer, 'wb') as wf: wf.setnchannels(channels); wf.setsampwidth(sampwidth); wf.setframerate(rate); wf.writeframes(audio_data_bytes)
        wav_buffer.seek(0)
        response = client.audio.transcriptions.create(file=(f"{speaker_label}_chunk.wav", wav_buffer.read()), model="whisper-large-v3", response_format="verbose_json", timestamp_granularities=["segment"])
        return [{"speaker": speaker_label, "start": s.get("start"), "end": s.get("end"), "text": s.get("text","").strip()} for s in response.segments] if response.segments else []
    except Exception as e:
        err_msg = str(e).lower()
        if ("rate limit" in err_msg or "429" in err_msg or "too many requests" in err_msg or "service unavailable" in err_msg or "503" in err_msg) and attempt < MAX_RETRIES:
            delay = BASE_RETRY_DELAY * (2**(attempt-1)) + random.uniform(0, BASE_RETRY_DELAY * 0.5); time.sleep(delay)
            return transcribe_audio_chunk(audio_data_bytes, channels, rate, sampwidth, speaker_label, attempt + 1)
        elif "insufficient_quota" in err_msg: messagebox.showerror("API Quota Error", "Insufficient API quota.", parent=root_gui)
        elif "request entity too large" in err_msg or "413" in err_msg: messagebox.showwarning("Transcription Error", f"Audio chunk for {speaker_label} too large. Skipped.", parent=root_gui)
        elif attempt == 1: messagebox.showerror("Transcription Error", f"Failed to transcribe for {speaker_label}: {str(e)[:100]}", parent=root_gui)
        return []

def capture_mic_audio_thread_func_rt(mic_device_name):
    global ffmpeg_mic_process_rt, stop_mic_capture_event_rt, mic_audio_chunk_queue_rt, current_mic_time_offset_rt 
    if not FFMPEG_CMD: return
    bytes_per_chunk_mic = int(AUDIO_SAMPLE_RATE*AUDIO_CHANNELS_MIC*AUDIO_SAMPLE_WIDTH_MIC*AUDIO_CHUNK_DURATION_SECONDS)
    cmd = [ FFMPEG_CMD, "-f", "dshow" if os.name == 'nt' else "avfoundation", "-i", f"audio={mic_device_name}" if os.name == 'nt' else mic_device_name, "-f", "s16le", "-ar", str(AUDIO_SAMPLE_RATE), "-ac", str(AUDIO_CHANNELS_MIC), "-" ]
    try:
        ffmpeg_mic_process_rt = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        audio_buffer_mic = bytearray()
        while not stop_mic_capture_event_rt.is_set():
            raw_audio = ffmpeg_mic_process_rt.stdout.read(AUDIO_SAMPLE_RATE*AUDIO_CHANNELS_MIC*AUDIO_SAMPLE_WIDTH_MIC//20) # 0.05s
            if not raw_audio and ffmpeg_mic_process_rt.poll() is not None: break
            if not raw_audio: time.sleep(0.01); continue
            audio_buffer_mic.extend(raw_audio)
            while len(audio_buffer_mic) >= bytes_per_chunk_mic:
                chunk = audio_buffer_mic[:bytes_per_chunk_mic]; audio_buffer_mic = audio_buffer_mic[bytes_per_chunk_mic:]
                mic_audio_chunk_queue_rt.put((bytes(chunk), current_mic_time_offset_rt)); current_mic_time_offset_rt += AUDIO_CHUNK_DURATION_SECONDS
        if audio_buffer_mic: mic_audio_chunk_queue_rt.put((bytes(audio_buffer_mic), current_mic_time_offset_rt))
    except Exception: pass # Log errors if needed
    finally:
        if ffmpeg_mic_process_rt and ffmpeg_mic_process_rt.poll() is None:
            try: ffmpeg_mic_process_rt.stdin.write(b'q\n'); ffmpeg_mic_process_rt.stdin.close(); ffmpeg_mic_process_rt.terminate(); ffmpeg_mic_process_rt.wait(timeout=1)
            except: 
                if ffmpeg_mic_process_rt.poll() is None: ffmpeg_mic_process_rt.kill()

system_audio_buffer_rt = bytearray(); system_audio_device_rate_rt, system_audio_device_channels_rt, system_audio_device_sampwidth_rt = None, None, 2
def system_audio_realtime_callback_rt(in_data, frame_count, time_info, status_flags):
    global system_audio_buffer_rt, system_audio_chunk_queue_rt, stop_system_audio_event_rt, current_system_time_offset_rt 
    if stop_system_audio_event_rt.is_set(): return (in_data, pa.paComplete)
    system_audio_buffer_rt.extend(in_data)
    if system_audio_device_rate_rt and system_audio_device_channels_rt:
        bytes_per_chunk = int(system_audio_device_rate_rt*system_audio_device_channels_rt*system_audio_device_sampwidth_rt*AUDIO_CHUNK_DURATION_SECONDS)
        while len(system_audio_buffer_rt) >= bytes_per_chunk:
            chunk = system_audio_buffer_rt[:bytes_per_chunk]; system_audio_buffer_rt = system_audio_buffer_rt[bytes_per_chunk:]
            system_audio_chunk_queue_rt.put({"data":bytes(chunk),"rate":system_audio_device_rate_rt,"channels":system_audio_device_channels_rt,"width":system_audio_device_sampwidth_rt,"abs_start_time":current_system_time_offset_rt})
            current_system_time_offset_rt += AUDIO_CHUNK_DURATION_SECONDS
    return (in_data, pa.paContinue)

def record_system_audio_pyaudiowpatch_realtime_thread_func_rt():
    global stop_system_audio_event_rt, system_audio_buffer_rt, system_audio_device_rate_rt, system_audio_device_channels_rt, system_audio_device_sampwidth_rt, current_system_time_offset_rt
    py_audio, stream = None, None; system_audio_buffer_rt.clear()
    try:
        py_audio = pa.PyAudio(); wasapi_info = py_audio.get_host_api_info_by_type(pa.paWASAPI)
        default_speakers = py_audio.get_device_info_by_index(wasapi_info["defaultOutputDevice"]); loopback_dev = None
        if default_speakers.get("isLoopbackDevice"): loopback_dev = default_speakers
        else:
            for dev in py_audio.get_loopback_device_info_generator():
                if default_speakers["name"] in dev["name"]: loopback_dev = dev; break
            if not loopback_dev: 
                for dev in py_audio.get_loopback_device_info_generator(): loopback_dev = dev; break # Fallback
        if not loopback_dev: return
        system_audio_device_rate_rt, system_audio_device_channels_rt, system_audio_device_sampwidth_rt = int(loopback_dev["defaultSampleRate"]), loopback_dev["maxInputChannels"], py_audio.get_sample_size(pa.paInt16)
        stream = py_audio.open(format=pa.paInt16, channels=system_audio_device_channels_rt, rate=system_audio_device_rate_rt, input=True, frames_per_buffer=pa.paFramesPerBufferUnspecified, input_device_index=loopback_dev["index"], stream_callback=system_audio_realtime_callback_rt)
        stream.start_stream()
        while not stop_system_audio_event_rt.is_set() and stream.is_active(): time.sleep(0.1)
        if system_audio_buffer_rt: system_audio_chunk_queue_rt.put({"data":bytes(system_audio_buffer_rt),"rate":system_audio_device_rate_rt,"channels":system_audio_device_channels_rt,"width":system_audio_device_sampwidth_rt,"abs_start_time":current_system_time_offset_rt})
    except Exception: pass # Log errors if needed
    finally:
        if stream: 
            if stream.is_active(): stream.stop_stream()
            stream.close()
        if py_audio: py_audio.terminate()

def audio_processor_thread_func_rt(audio_queue, speaker_label, is_mic_thread):
    global transcript_display_queue_rt, is_recording_realtime_flag, all_transcribed_segments_rt, all_segments_lock_rt
    while True:
        try:
            item = audio_queue.get(timeout=0.5); audio_queue.task_done(); abs_start_time = 0.0
            if is_mic_thread: data, abs_start_time = item; chans, sr, width = AUDIO_CHANNELS_MIC, AUDIO_SAMPLE_RATE, AUDIO_SAMPLE_WIDTH_MIC
            else: data, chans, sr, width, abs_start_time = item["data"],item["channels"],item["rate"],item["width"],item["abs_start_time"]
            if data:
                rel_segs = transcribe_audio_chunk(data, chans, sr, width, speaker_label)
                if rel_segs:
                    adj_segs = [{"speaker":s["speaker"],"start":abs_start_time+(s.get("start")or 0.0),"end":abs_start_time+(s.get("end")or 0.0),"text":s["text"]} for s in rel_segs]
                    if adj_segs: transcript_display_queue_rt.put(adj_segs);
                    with all_segments_lock_rt: all_transcribed_segments_rt.extend(adj_segs)
        except queue.Empty:
            if not is_recording_realtime_flag and audio_queue.empty(): break
            continue
        except Exception: # Log errors if needed
            if not is_recording_realtime_flag: break 

def update_transcript_display_rt(text_widget):
    global transcript_display_queue_rt, is_recording_realtime_flag, root_gui
    try:
        while not transcript_display_queue_rt.empty():
            batch = transcript_display_queue_rt.get_nowait(); text_to_add = ""
            for seg in batch:
                start_str = ""; 
                if seg.get("start") is not None:
                    try: secs_tot = int(float(seg["start"])); h,r=divmod(secs_tot,3600);m,s=divmod(r,60); start_str=f"[{h:02d}:{m:02d}:{s:02d}] "
                    except: start_str = f"[{seg['start']:.1f}s] "
                text_to_add += f"{start_str}{seg['speaker']}: {seg['text']}\n"
            if text_to_add:
                state = text_widget.cget("state"); text_widget.config(state=tk.NORMAL); text_widget.insert(tk.END,text_to_add); text_widget.config(state=state); text_widget.see(tk.END)
            transcript_display_queue_rt.task_done()
    except queue.Empty: pass
    except Exception: pass # Log errors if needed
    if is_recording_realtime_flag or not transcript_display_queue_rt.empty():
        if root_gui: root_gui.after(300, lambda: update_transcript_display_rt(text_widget))
    elif root_gui: root_gui.after(500, lambda: update_transcript_display_rt(text_widget))

def start_recording_realtime(status_label, start_button, stop_button, transcript_text_widget):
    global is_recording_realtime_flag, mic_capture_thread_rt, stop_mic_capture_event_rt, system_audio_capture_thread_rt, stop_system_audio_event_rt, mic_audio_processor_thread_rt, system_audio_processor_thread_rt, all_transcribed_segments_rt, mic_audio_chunk_queue_rt, system_audio_chunk_queue_rt, transcript_display_queue_rt, current_mic_time_offset_rt, current_system_time_offset_rt, client
    if not FFMPEG_CMD and os.name == 'nt': messagebox.showerror("FFmpeg Error", "FFmpeg not found."); return
    if not client: 
        if not prompt_and_set_api_key(root_gui): messagebox.showerror("API Key Error", "Groq API Key required."); return
    is_recording_realtime_flag = True; stop_mic_capture_event_rt.clear(); stop_system_audio_event_rt.clear()
    current_mic_time_offset_rt = 0.0; current_system_time_offset_rt = 0.0
    with all_segments_lock_rt: all_transcribed_segments_rt.clear()
    for q in [mic_audio_chunk_queue_rt,system_audio_chunk_queue_rt,transcript_display_queue_rt]:
        while not q.empty(): 
            try: q.get_nowait(); q.task_done()
            except queue.Empty: break
    transcript_text_widget.config(state=tk.NORMAL); transcript_text_widget.delete('1.0',tk.END); transcript_text_widget.config(state=tk.DISABLED)
    status_label.config(text="🔴 Real-time recording..."); start_button.config(state=tk.DISABLED); stop_button.config(state=tk.NORMAL)
    mic_dev = get_mic_device_dshow()
    if not mic_dev and os.name=='nt': messagebox.showerror("Mic Error","Mic not found."); is_recording_realtime_flag=False; start_button.config(state=tk.NORMAL); stop_button.config(state=tk.DISABLED); return
    elif not mic_dev: mic_dev="default"
    mic_capture_thread_rt=threading.Thread(target=capture_mic_audio_thread_func_rt,args=(mic_dev,),daemon=True); mic_capture_thread_rt.start()
    mic_audio_processor_thread_rt=threading.Thread(target=audio_processor_thread_func_rt,args=(mic_audio_chunk_queue_rt,"You (Mic)",True),daemon=True); mic_audio_processor_thread_rt.start()
    if os.name=='nt':
        system_audio_capture_thread_rt=threading.Thread(target=record_system_audio_pyaudiowpatch_realtime_thread_func_rt,daemon=True); system_audio_capture_thread_rt.start()
        system_audio_processor_thread_rt=threading.Thread(target=audio_processor_thread_func_rt,args=(system_audio_chunk_queue_rt,"Other (System)",False),daemon=True); system_audio_processor_thread_rt.start()
    update_transcript_display_rt(transcript_text_widget)

def stop_recording_realtime(status_label, progress_bar, start_button, stop_button):
    global is_recording_realtime_flag, stop_mic_capture_event_rt, stop_system_audio_event_rt, all_transcribed_segments_rt, mic_capture_thread_rt, system_audio_capture_thread_rt, mic_audio_processor_thread_rt, system_audio_processor_thread_rt, last_saved_transcript_path
    status_label.config(text="⏳ Stopping & finalizing..."); stop_button.config(state=tk.DISABLED)
    is_recording_realtime_flag=False; stop_mic_capture_event_rt.set(); stop_system_audio_event_rt.set()
    timeout_cap = AUDIO_CHUNK_DURATION_SECONDS+3
    for t,n in [(mic_capture_thread_rt,"MicCap"),(system_audio_capture_thread_rt,"SysCap")]:
        if t and t.is_alive(): t.join(timeout=timeout_cap)
    status_label.config(text="⚙️ Processing remaining chunks...")
    if root_gui: 
        for _ in range(3): root_gui.update_idletasks(); time.sleep(0.05)
    timeout_proc = (AUDIO_CHUNK_DURATION_SECONDS*2)+25
    for t,n in [(mic_audio_processor_thread_rt,"MicProc"),(system_audio_processor_thread_rt,"SysProc")]:
        if t and t.is_alive(): t.join(timeout=timeout_proc)
    status_label.config(text="⚙️ Compiling final transcript..."); progress_bar["value"]=50
    try:
        with all_segments_lock_rt:
            segs=[s for s in all_transcribed_segments_rt if s and s.get("text")]
            def get_start(s): 
                try: return float(s["start"]) if s.get("start") is not None else float('inf')
                except: return float('inf')
            sorted_segs = sorted(segs, key=get_start)
        ts_file=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sugg_file=f"rt_transcript_{ts_file}.txt"
        init_dir=os.path.join(os.path.expanduser("~"),"Documents"); os.makedirs(init_dir,exist_ok=True)
        final_path=filedialog.asksaveasfilename(initialdir=init_dir,title="Save Transcript",defaultextension=".txt",initialfile=sugg_file,filetypes=[("Text","*.txt")])
        if final_path:
            with open(final_path,"w",encoding="utf-8") as f:
                for s_item in sorted_segs:
                    s_time_str=""; 
                    if s_item.get("start") is not None:
                        try: tot_s=int(float(s_item["start"]));h,r=divmod(tot_s,3600);m,s_val=divmod(r,60);s_time_str=f"[{h:02d}:{m:02d}:{s_val:02d}] "
                        except: s_time_str=f"[{s_item['start']}] "
                    f.write(f"{s_time_str}{s_item['speaker']}: {s_item['text']}\n")
            last_saved_transcript_path = final_path
            messagebox.showinfo("Done",f"Transcript saved:\n{final_path}", parent=root_gui)
            status_label.config(text=f"✅ Saved: {os.path.basename(final_path)}")
        else: 
            messagebox.showinfo("Cancelled","Save cancelled.", parent=root_gui); 
            status_label.config(text="Finalized (save cancelled).")
    except Exception as e: messagebox.showerror("Error",f"Could not save: {e}", parent=root_gui); status_label.config(text="❌ Error saving.")
    finally: progress_bar["value"]=0; start_button.config(state=tk.NORMAL)

def setup_gui_realtime():
    global root_gui
    root_gui = tk.Tk(); root_gui.title("Meeting Transcriber (Real-Time)"); root_gui.geometry("750x650")
    menubar=tk.Menu(root_gui); settings_menu=tk.Menu(menubar,tearoff=0)
    settings_menu.add_command(label="Set Groq API Key",command=lambda:prompt_and_set_api_key(root_gui))
    menubar.add_cascade(label="Settings",menu=settings_menu); root_gui.config(menu=menubar)
    
    ffmpeg_ok, api_key_ok = True, True
    if not FFMPEG_CMD and os.name=='nt': messagebox.showerror("Startup Error - FFmpeg Missing","FFmpeg not found. Mic recording disabled.",parent=root_gui); ffmpeg_ok=False
    if not load_api_key_from_config(): api_key_ok=False # Tries to load and validate
    # If API key still not set after load_api_key_from_config (e.g. no config, or invalid stored key)
    # prompt_and_set_api_key could be called here, or rely on user doing it via menu.
    # For now, we'll show a status if API key isn't ready.
    
    main_frm=ttk.Frame(root_gui,padding="10");main_frm.pack(expand=True,fill=tk.BOTH)
    ctrl_frm=ttk.Frame(main_frm);ctrl_frm.pack(fill=tk.X,pady=5)
    ttk.Label(ctrl_frm,text="\U0001F399\ufe0f Real-Time Transcriber",font=("Arial",16)).pack(pady=5)
    prog_bar=ttk.Progressbar(ctrl_frm,length=450,mode='determinate');prog_bar.pack(pady=5)
    stat_lbl=ttk.Label(ctrl_frm,text="Welcome! Configure API Key via Settings if needed.",font=("Arial",10));stat_lbl.pack(pady=5)
    disp_frm=ttk.LabelFrame(main_frm,text="Live Transcript",padding="5");disp_frm.pack(expand=True,fill=tk.BOTH,pady=10)
    txt_widget=scrolledtext.ScrolledText(disp_frm,wrap=tk.WORD,state=tk.DISABLED,height=20,font=("Arial",9));txt_widget.pack(expand=True,fill=tk.BOTH)
    global live_transcript_widget; live_transcript_widget = txt_widget
    # Summary Controls
    summary_frm=ttk.Frame(main_frm);summary_frm.pack(fill=tk.X,pady=5)
    ttk.Label(summary_frm,text="Summary Type:").pack(side=tk.LEFT,padx=5)
    summary_var=tk.StringVar(value="Summarize Interview")
    summary_combo=ttk.Combobox(summary_frm,textvariable=summary_var,state="readonly",values=["Summarize Interview","Summarize Meeting"],width=30)
    summary_combo.pack(side=tk.LEFT,padx=5)
    def on_summarize_click():
        summarize_transcript(summary_var.get())
    summarize_btn=ttk.Button(summary_frm,text="Summarize",command=on_summarize_click)
    summarize_btn.pack(side=tk.LEFT,padx=10)
    upload_btn=ttk.Button(summary_frm,text="Upload Transcript...",command=lambda:upload_transcript_and_preview(txt_widget))
    upload_btn.pack(side=tk.LEFT,padx=10)
    clear_btn=ttk.Button(summary_frm,text="Clear",command=lambda:display_text_in_live_transcript(txt_widget, "", clear_first=True))
    clear_btn.pack(side=tk.LEFT,padx=10)
    saveas_btn=ttk.Button(summary_frm,text="Save Summary As...",command=lambda:save_live_transcript_as(txt_widget))
    saveas_btn.pack(side=tk.LEFT,padx=10)

    btn_frm=ttk.Frame(main_frm);btn_frm.pack(fill=tk.X,pady=10)
    start_btn=ttk.Button(btn_frm,text="\U0001F3AC Start Recording",command=lambda:start_recording_realtime(stat_lbl,start_btn,stop_btn,txt_widget));start_btn.pack(side=tk.LEFT,padx=10,expand=True)
    stop_btn=ttk.Button(btn_frm,text="\u23F9\ufe0f Stop & Save",command=lambda:stop_recording_realtime(stat_lbl,prog_bar,start_btn,stop_btn),state=tk.DISABLED);stop_btn.pack(side=tk.LEFT,padx=10,expand=True)

    if not ffmpeg_ok or not api_key_ok : # Check initial load/config status
        start_btn.config(state=tk.DISABLED)
        err_parts=[]
        if not ffmpeg_ok: err_parts.append("FFmpeg missing")
        if not api_key_ok: err_parts.append("API Key not (yet) configured/valid")
        stat_lbl.config(text=f"Error: {', '.join(err_parts)}. Check settings.")
    elif not client: # API key was attempted but client is still None (e.g. invalid key entered and not fixed)
        start_btn.config(state=tk.DISABLED)
        stat_lbl.config(text="Error: API Key invalid or connection issue. Please set a valid key via Settings.")
        if not GROQ_API_KEY_STORED : # If no key is even stored, guide user
             messagebox.showinfo("API Key Required", "A valid Groq API Key is required for transcription. Please set it via Settings > Set Groq API Key.", parent=root_gui)


    root_gui.mainloop()

if __name__ == "__main__":
    if os.name != 'nt': print("INFO: System audio capture is Windows-specific.")
    setup_gui_realtime()
