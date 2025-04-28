import gradio as gr
import torch
import torchaudio
import numpy as np
import os
import time
import traceback
import soundfile as sf
import torch.nn.functional as F
import matplotlib.pyplot as plt
from demucs import pretrained
from demucs import apply
import uuid
import librosa
import shutil  # Added, used for file operations like copying
import platform
import tempfile
from typing import List, Dict, Tuple, Optional, Union

# Pedalboard Check
try:
    from pedalboard import (
        Pedalboard, Compressor, Reverb, Gain, LowpassFilter,
        Distortion, Delay, Phaser, HighShelfFilter,
        LowShelfFilter,
    )
    from pedalboard.io import AudioFile
    PEDALBOARD_AVAILABLE = True
    print("Pedalboard library found.")
except ImportError as e:
    PEDALBOARD_AVAILABLE = False
    print(f"ImportError occurred: {str(e)}")
    class Pedalboard: pass
    class AudioFile: pass
    class Compressor: pass
    class Reverb: pass
    class Gain: pass
    class LowpassFilter: pass
    class Distortion: pass
    class Delay: pass
    class Phaser: pass
    class HighShelfFilter: pass
    class LowShelfFilter: pass
    print("Warning: Pedalboard library not found. Effects will be disabled. Install with: pip install pedalboard")

# Update librosa import for tempo
from librosa.feature.rhythm import tempo as librosa_tempo

# Constants
MODELS = ["mdx", "mdx_extra", "mdx_q"]
OUTPUT_DIR = "output"
SAMPLE_RATE = 44100
STEM_NAMES = ["drums", "bass", "other", "vocals"]

# Color schemes
PRIMARY_COLOR = "#FF3860"
SECONDARY_COLOR = "#3273DC"
ACCENT_COLOR = "#23D160"
DARK_COLOR = "#363636"
LIGHT_COLOR = "#F5F5F5"

# Utility Functions (unchanged except for get_unique_filename)
def get_unique_filename(base_name, folder, ext=".wav"):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as e:
            print(f"Error creating directory {folder}: {e}")
            folder = "."
    filename = f"{base_name}_{str(uuid.uuid4())[:8]}{ext}"
    return os.path.join(folder, filename)

def create_waveform_plot(audio_path, output_path=None, color=PRIMARY_COLOR):
    if not audio_path or not os.path.exists(audio_path):
        print(f"Waveform Plot Error: Audio file not found at {audio_path}")
        return None
    try:
        y, sr = sf.read(audio_path)
        if y.ndim > 1: y = np.mean(y, axis=1)
        max_duration_plot = 300
        if len(y) / sr > max_duration_plot:
            print(f"Waveform Plot Warning: Audio too long, plotting first {max_duration_plot}s.")
            y = y[:int(max_duration_plot * sr)]
        time = np.linspace(0, len(y) / sr, num=len(y))
        plt.figure(figsize=(8, 5), facecolor='#1A1A1A')  # Adjusted figsize for new column width (1.6:1 ratio)
        plt.plot(time, y, color=color, linewidth=0.5)
        plt.fill_between(time, y, alpha=0.2, color=color)
        plt.xlabel('Time (s)', color='white', fontsize=10)
        plt.ylabel('Amplitude', color='white', fontsize=10)
        plt.yticks([-1, 0, 1])
        plt.grid(color='#333333', linestyle='-', linewidth=0.3, alpha=0.3)
        plt.tight_layout(pad=0.5)
        ax = plt.gca()
        ax.set_facecolor('#1A1A1A')
        for spine in ['top', 'right', 'left', 'bottom']: ax.spines[spine].set_color('#555555')
        ax.tick_params(axis='x', colors='white', labelsize=9)
        ax.tick_params(axis='y', colors='white', labelsize=9)
        if output_path is None: output_path = get_unique_filename("waveform", OUTPUT_DIR, ext=".png")
        plt.savefig(output_path, facecolor='#1A1A1A', dpi=90)
        plt.close()
        print(f"Saved waveform plot: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating waveform plot for {audio_path}: {str(e)}")
        traceback.print_exc()
        return None

# GPU Accelerated Silence Trimming (unchanged)
def trim_silence_gpu(waveform_gpu: torch.Tensor, sr: int, threshold: float = 0.01, pad_ms: int = 250, win_ms: int = 40, hop_ms: int = 10) -> torch.Tensor:
    print(f"GPU Trim: Input shape {waveform_gpu.shape}, threshold={threshold}, pad_ms={pad_ms}")
    if waveform_gpu.dim() == 1: waveform_gpu = waveform_gpu.unsqueeze(0)
    C, N = waveform_gpu.shape
    if N == 0: return waveform_gpu
    win_length = int(win_ms / 1000 * sr)
    hop_length = int(hop_ms / 1000 * sr)
    if win_length == 0 or hop_length == 0:
        print("GPU Trim Warning: Window or hop length is zero, skipping trim.")
        return waveform_gpu
    if N >= win_length: padding_needed = (hop_length - (N - win_length) % hop_length) % hop_length
    else: padding_needed = win_length - N
    waveform_padded = F.pad(waveform_gpu, (0, padding_needed))
    frames = waveform_padded.unfold(dimension=-1, size=win_length, step=hop_length)
    frame_max_amp = frames.abs().max(dim=-1).values
    frame_energy = frame_max_amp.mean(dim=0)
    is_nonsilent = frame_energy > threshold
    nonsilent_indices = torch.nonzero(is_nonsilent).squeeze()
    if nonsilent_indices.numel() == 0:
        print("GPU Trim: No non-silent frames found.")
        return waveform_gpu
    if nonsilent_indices.dim() == 0:
        first_nonsilent_frame = nonsilent_indices.item()
        last_nonsilent_frame = nonsilent_indices.item()
    else:
        first_nonsilent_frame = nonsilent_indices.min().item()
        last_nonsilent_frame = nonsilent_indices.max().item()
    start_sample_approx = first_nonsilent_frame * hop_length
    end_sample_approx = last_nonsilent_frame * hop_length + win_length
    pad_samples = int(pad_ms / 1000 * sr)
    start_sample = max(0, start_sample_approx - pad_samples)
    end_sample = min(N, end_sample_approx + pad_samples)
    if start_sample >= end_sample:
        print("GPU Trim Warning: Calculated trim range is invalid, returning original.")
        return waveform_gpu
    print(f"GPU Trim: Detected sound from sample {start_sample} to {end_sample} (orig len {N})")
    trimmed_waveform = waveform_gpu[..., start_sample:end_sample]
    print(f"GPU Trim: Output shape {trimmed_waveform.shape}")
    return trimmed_waveform

# --- Detect_tempo function ---
def detect_tempo(y: np.ndarray, sr: int, name_for_log: str = "audio") -> Optional[float]:
    """Detects tempo directly from a NumPy array."""
    if y is None or sr is None:
        print(f"Tempo Detection Error: Invalid audio data or sample rate provided for {name_for_log}.")
        return None
    try:
        print(f"Detecting tempo for {name_for_log} (shape={y.shape}, sr={sr})...")
        # Ensure 'y' is suitable for librosa tempo (e.g., float, mono)
        if y.dtype != np.float32 and y.dtype != np.float64:
             y = y.astype(np.float32)
        if y.ndim > 1: # Convert to mono if stereo/multi-channel
             y_mono = np.mean(y, axis=-1)
        else:
             y_mono = y

        # Detect tempo using librosa
        tempo_result = librosa_tempo(y=y_mono, sr=sr, hop_length=512) # Using the imported librosa_tempo

        if tempo_result is None or len(tempo_result) == 0:
             print(f"Tempo detection failed for {name_for_log}.")
             return None

        tempo_value = float(tempo_result[0]) # Extract the primary tempo value
        print(f"Detected tempo for {name_for_log}: {tempo_value:.2f} BPM")
        return tempo_value
    except Exception as e:
        print(f"Error detecting tempo for {name_for_log}: {str(e)}")
        traceback.print_exc()
        return None
# --- End of Detect_tempo function ---

# --- Time_stretch_audio function ---
def time_stretch_audio(y: np.ndarray, sr: int, target_tempo: float, original_tempo: float, max_stretch_factor: float = 1.10, name_for_log: str = "audio") -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Time-stretches audio directly from a NumPy array."""
    if y is None or sr is None:
        print(f"Skipping time-stretch for {name_for_log}: Invalid audio data or sample rate.")
        return None, None
    try:
        original_tempo_f = float(original_tempo)
        target_tempo_f = float(target_tempo)

        if original_tempo_f <= 0 or target_tempo_f <= 0:
            print(f"Skipping time-stretch for {name_for_log}: Invalid tempo values (original={original_tempo_f}, target={target_tempo_f})")
            return None, None
        if abs(original_tempo_f - target_tempo_f) < 1.0:
            print(f"Skipping time-stretch for {name_for_log}: Tempos match ({original_tempo_f:.2f} vs {target_tempo_f:.2f})")
            return y, sr # Return original array and SR

        full_rate = target_tempo_f / original_tempo_f
        # Apply cap
        capped_rate = min(max_stretch_factor, max(1.0 / max_stretch_factor, full_rate))

        if abs(full_rate - capped_rate) > 1e-6: # Check if capping occurred
            print(f"Warning: Capping stretch rate for {name_for_log} from {full_rate:.3f} to {capped_rate:.3f} to preserve quality")
            effective_tempo = original_tempo_f * capped_rate
            print(f"Effective tempo after stretch: {effective_tempo:.2f} BPM (target was {target_tempo_f:.2f} BPM)")
        else:
            print(f"Time-stretching {name_for_log} from {original_tempo_f:.2f} BPM to {target_tempo_f:.2f} BPM (rate={capped_rate:.3f})")

        # Ensure 'y' is float for time_stretch
        if y.dtype != np.float32 and y.dtype != np.float64:
             y = y.astype(np.float32)

        # Perform time stretch directly on the array
        y_stretched = librosa.effects.time_stretch(y, rate=capped_rate)

        print(f"Time stretch successful for {name_for_log}.")
        # Time stretching preserves sample rate
        return y_stretched, sr
    except Exception as e:
        print(f"Error time-stretching {name_for_log}: {str(e)}")
        traceback.print_exc()
        return None, None
# --- End of Time_stretch_audio function ---

# --- Added align_beats function to handle beat alignment ---
def align_beats(track1_np: np.ndarray, sr1: int, track2_np: np.ndarray, sr2: int) -> tuple[np.ndarray, int]:
    """Aligns track2_np start to match the first beat of track1_np."""
    print("Attempting beat alignment using NumPy arrays...")
    try:
        if track1_np is None or track2_np is None:
            print("Alignment Error: Received None for track data.")
            return track2_np, sr2

        # Convert to mono if stereo to avoid axis issues during resampling
        if track1_np.ndim == 2:
            print("Converting Track 1 to mono for alignment")
            track1_np = np.mean(track1_np, axis=1)  # Shape: (samples,)
        if track2_np.ndim == 2:
            print("Converting Track 2 to mono for alignment")
            track2_np = np.mean(track2_np, axis=1)  # Shape: (samples,)

        # Downsampling for alignment analysis
        target_sr = 11025  # Target sample rate for alignment
        sr1_a, sr2_a = sr1, sr2

        print(f"Resampling tracks to {target_sr} Hz for alignment analysis...")
        # Resample numpy arrays
        if sr1 != target_sr:
            if len(track1_np) < 10:  # Minimum length to avoid resampling errors
                print("Track 1 too short for resampling, skipping alignment")
                track1_np_a = track1_np
                sr1_a = sr1
            else:
                track1_np_a = librosa.resample(track1_np, orig_sr=sr1, target_sr=target_sr, res_type='kaiser_fast')
                sr1_a = target_sr
        else:
            track1_np_a = track1_np

        if sr2 != target_sr:
            if len(track2_np) < 10:  # Minimum length to avoid resampling errors
                print("Track 2 too short for resampling, skipping alignment")
                track2_np_a = track2_np
                sr2_a = sr2
            else:
                track2_np_a = librosa.resample(track2_np, orig_sr=sr2, target_sr=target_sr, res_type='kaiser_fast')
                sr2_a = target_sr
        else:
            track2_np_a = track2_np

        print("Detecting onsets and beats...")
        try:
            # Detect onsets and beats
            track1_onset_env = librosa.onset.onset_strength(y=track1_np_a, sr=sr1_a, hop_length=128, n_fft=512)
            track2_onset_env = librosa.onset.onset_strength(y=track2_np_a, sr=sr2_a, hop_length=128, n_fft=512)

            # Get beat frames
            track1_beats = librosa.beat.beat_track(onset_envelope=track1_onset_env, sr=sr1_a, hop_length=128)[1]
            track2_beats = librosa.beat.beat_track(onset_envelope=track2_onset_env, sr=sr2_a, hop_length=128)[1]
        except Exception as detection_err:
            print(f"Error during onset/beat detection: {detection_err}")
            traceback.print_exc()
            print("Skipping alignment due to detection error.")
            return track2_np, sr2

        # Calculate offset based on the first detected beat
        if len(track1_beats) > 0 and len(track2_beats) > 0:
            track1_first_beat_time = librosa.frames_to_time(track1_beats[0], sr=sr1_a, hop_length=128)
            track2_first_beat_time = librosa.frames_to_time(track2_beats[0], sr=sr2_a, hop_length=128)
            print(f"Aligning beats: T1 first beat={track1_first_beat_time:.3f}s, T2 first beat={track2_first_beat_time:.3f}s")

            # Calculate offset
            time_offset_sec = track2_first_beat_time - track1_first_beat_time
            offset_samples = int(time_offset_sec * sr2)  # Use original sr2 for offset
            print(f"Calculated offset = {offset_samples} samples ({time_offset_sec:.3f} s for sr={sr2})")

            # Apply offset to track2_np
            if offset_samples > 0:  # T2 starts later -> Trim T2 start
                offset_samples = abs(offset_samples)
                print(f"Trimming start of Track 2 by {offset_samples} samples.")
                if offset_samples < track2_np.shape[0]:
                    track2_np = track2_np[offset_samples:]
                else:
                    print("Warning: Offset trim >= track length.")
                    track2_np = np.zeros_like(track2_np[:1])
            elif offset_samples < 0:  # T2 starts earlier -> Pad T2 start
                offset_samples = abs(offset_samples)
                print(f"Padding start of Track 2 by {offset_samples} samples.")
                pad_width = ((offset_samples, 0),) + ((0, 0),) * (track2_np.ndim - 1)
                track2_np = np.pad(track2_np, pad_width, mode='constant')
        else:
            print("Beat detection failed or found no beats, skipping alignment offset calculation.")

        print("Beat alignment finished.")
        return track2_np, sr2

    except Exception as e:
        print(f"Error during align_beats function: {e}")
        traceback.print_exc()
        return track2_np, sr2

# --- Replaced blend_tracks function to fix mono to stereo issue ---
def blend_tracks(track1_np, sr1, track1_vol, track1_effects_board,
                 track2_np, sr2, track2_vol, track2_effects_board,
                 output_path):
    try:
        if sr1 != sr2:
            raise ValueError(f"Sample rates must match before blending ({sr1} vs {sr2}). Call resample first.")
        sr_target = sr1

        # --- Ensure inputs are float32 ---
        if track1_np.dtype != np.float32: track1_np = track1_np.astype(np.float32)
        if track2_np.dtype != np.float32: track2_np = track2_np.astype(np.float32)

        # --- Apply effects if available ---
        if PEDALBOARD_AVAILABLE and track1_effects_board:
            print("Applying effects to Track 1...")
            # Pedalboard expects shape (samples, channels) or (samples,)
            track1_np = track1_effects_board(track1_np, sr_target)
        if PEDALBOARD_AVAILABLE and track2_effects_board:
            print("Applying effects to Track 2...")
            track2_np = track2_effects_board(track2_np, sr_target)

        # --- Apply volumes ---
        # Ensure volumes are floats to avoid potential type issues
        track1_np *= float(track1_vol)
        track2_np *= float(track2_vol)

        # --- Handle dimensions and padding ---
        # Ensure at least 1D
        if track1_np.ndim == 0: track1_np = np.array([track1_np])
        if track2_np.ndim == 0: track2_np = np.array([track2_np])

        # Get lengths and ensure 2D (samples, channels) for consistent processing
        len1 = track1_np.shape[0]
        chans1 = track1_np.shape[1] if track1_np.ndim == 2 else 1
        if track1_np.ndim == 1: track1_np = track1_np[:, np.newaxis] # Convert (N,) to (N, 1)

        len2 = track2_np.shape[0]
        chans2 = track2_np.shape[1] if track2_np.ndim == 2 else 1
        if track2_np.ndim == 1: track2_np = track2_np[:, np.newaxis] # Convert (N,) to (N, 1)

        # Determine target shape
        max_len = max(len1, len2)
        max_chans = max(chans1, chans2) # Will be 1 if both mono, 2 if either is stereo

        # --- Corrected Padding Logic ---
        # Pad Track 1 if necessary
        if len1 < max_len or chans1 < max_chans:
            padded_track1 = np.zeros((max_len, max_chans), dtype=track1_np.dtype)
            if chans1 == 1 and max_chans == 2:
                # Mono to Stereo: Copy mono to both channels
                print("Padding Track 1: Mono -> Stereo")
                padded_track1[:len1, 0] = track1_np[:len1, 0]
                padded_track1[:len1, 1] = track1_np[:len1, 0]
            else:
                # Length padding or already stereo
                 print(f"Padding Track 1: Length {len1}->{max_len}, Chans {chans1}->{max_chans}")
                 padded_track1[:len1, :chans1] = track1_np[:len1, :chans1]
            track1_np = padded_track1

        # Pad Track 2 if necessary
        if len2 < max_len or chans2 < max_chans:
            padded_track2 = np.zeros((max_len, max_chans), dtype=track2_np.dtype)
            if chans2 == 1 and max_chans == 2:
                 # Mono to Stereo: Copy mono to both channels
                 print("Padding Track 2: Mono -> Stereo")
                 padded_track2[:len2, 0] = track2_np[:len2, 0]
                 padded_track2[:len2, 1] = track2_np[:len2, 0]
            else:
                 # Length padding or already stereo
                 print(f"Padding Track 2: Length {len2}->{max_len}, Chans {chans2}->{max_chans}")
                 padded_track2[:len2, :chans2] = track2_np[:len2, :chans2]
            track2_np = padded_track2
        # --- End Corrected Padding Logic ---

        # --- Mix ---
        print(f"Mixing Track 1 ({track1_np.shape}) and Track 2 ({track2_np.shape})")
        mixed_track_np = track1_np + track2_np

        # --- Normalize and Save ---
        max_val = np.max(np.abs(mixed_track_np))
        if max_val == 0: print("Warning: Blended result is silent.")
        elif max_val > 0.98:
            print(f"Normalizing blended track (peak was {max_val:.2f})")
            mixed_track_np = mixed_track_np * (0.98 / max_val)
        else:
             print(f"Peak value: {max_val:.2f} (no normalization needed)")


        sf.write(output_path, mixed_track_np, sr_target)
        print(f"Saved blended track: {output_path}")
        return output_path, mixed_track_np, sr_target

    except Exception as e:
        print(f"Error in blend_tracks helper: {str(e)}")
        traceback.print_exc()
        return None, None, None
# --- End of updated blend_tracks function ---

# --- Updated blend_audio function ---
def blend_audio(
    track1_source, track1_upload, track1_vol, track1_low_gain, track1_high_gain,
    track1_threshold, track1_ratio, track1_attack, track1_release, track1_reverb_room, track1_reverb_decay, track1_reverb_wetdry,
    track1_delay_time, track1_delay_feedback, track1_delay_wetdry,
    track2_source, track2_upload, track2_vol, track2_low_gain, track2_high_gain,
    track2_threshold, track2_ratio, track2_attack, track2_release,
    track2_reverb_room, track2_reverb_decay, track2_reverb_wetdry,
    track2_delay_time, track2_delay_feedback, track2_delay_wetdry,
    match_tempo_chk, offset_ms,
    processed_vocals_state, processed_instrumental_state,
    processed_vocals_path, processed_instrumental_path,
    progress=gr.Progress(track_tqdm=True)
):
    print("\n--- Inside blend_audio Function (with Progress) ---")
    try:
        start_time = time.time()
        sr_for_processing = SAMPLE_RATE # Assuming SAMPLE_RATE is defined globally

        if progress is not None:
            try:
                progress(0, desc="Initializing blend... -- 0%")
            except Exception as e:
                print(f"Warning: Progress update failed: {e}")

        # --- Load Track 1 ---
        track1_np, sr1 = None, None
        # (Keep the existing logic for loading Track 1 based on track1_source)
        if track1_source == "Upload New Track":
            if not track1_upload: raise ValueError("Please upload Track 1!")
            print(f"Loading Track 1 from upload: {track1_upload}")
            # Use librosa for consistency if possible, or ensure output is numpy
            track1_np, sr1 = librosa.load(track1_upload, sr=None) # Using librosa load
        elif track1_source == "Use Processed Vocals":
            if not processed_vocals_state: raise ValueError("No processed vocals available!")
            print("Using processed vocals for Track 1")
            track1_np, sr1, _ = processed_vocals_state
        elif track1_source == "Use Processed Instrumental":
            if not processed_instrumental_state: raise ValueError("No processed instrumental available!")
            print("Using processed instrumental for Track 1")
            track1_np, sr1, _ = processed_instrumental_state
        else: raise ValueError("Invalid Track 1 Source")

        if track1_np is None: raise ValueError("Failed to load Track 1 data.")
        print(f"Track 1 loaded: shape={track1_np.shape}, sr={sr1}")

        # --- Resample Track 1 if necessary ---
        if sr1 != sr_for_processing:
             print(f"Resampling T1 {sr1}->{sr_for_processing}")
             track1_np = librosa.resample(track1_np, orig_sr=sr1, target_sr=sr_for_processing, res_type='kaiser_fast')
             sr1 = sr_for_processing

        if progress is not None:
            try:
                progress(0.1, desc="Loading tracks... -- 10%")
            except Exception as e:
                print(f"Warning: Progress update failed: {e}")

        # --- Load Track 2 ---
        track2_np, sr2 = None, None
        # (Keep the existing logic for loading Track 2 based on track2_source)
        if track2_source == "Upload New Track":
            if not track2_upload: raise ValueError("Please upload Track 2!")
            print(f"Loading Track 2 from upload: {track2_upload}")
            # Use librosa for consistency if possible, or ensure output is numpy
            track2_np, sr2 = librosa.load(track2_upload, sr=None) # Using librosa load
        elif track2_source == "Use Processed Vocals":
            if not processed_vocals_state: raise ValueError("No processed vocals available!")
            print("Using processed vocals for Track 2")
            track2_np, sr2, _ = processed_vocals_state
        elif track2_source == "Use Processed Instrumental":
            if not processed_instrumental_state: raise ValueError("No processed instrumental available!")
            print("Using processed instrumental for Track 2")
            track2_np, sr2, _ = processed_instrumental_state
        else: raise ValueError("Invalid Track 2 Source")

        if track2_np is None: raise ValueError("Failed to load Track 2 data.")
        print(f"Track 2 loaded: shape={track2_np.shape}, sr={sr2}")

        # --- Resample Track 2 if necessary ---
        if sr2 != sr_for_processing:
             print(f"Resampling T2 {sr2}->{sr_for_processing}")
             track2_np = librosa.resample(track2_np, orig_sr=sr2, target_sr=sr_for_processing, res_type='kaiser_fast')
             sr2 = sr_for_processing

        if progress is not None:
            try:
                progress(0.2, desc="Processing tracks... -- 20%") # Changed description slightly
            except Exception as e:
                print(f"Warning: Progress update failed: {e}")

        # --- Ensure Mono/Stereo Consistency (Example: Convert both to mono for alignment/effects if needed) ---
        # This depends on your downstream functions (align_beats, blend_tracks)
        # For simplicity, let's assume they handle mono/stereo appropriately or convert inside them.
        # Example: Convert track 2 to mono if needed by align_beats
        # if track2_np.ndim == 2:
        #     print("Converting Track 2 to mono for alignment")
        #     track2_np = np.mean(track2_np, axis=1) # Now shape (samples,)

        # --- Tempo Matching ---
        if match_tempo_chk:
            if progress: progress(0.3, desc="Matching tempo... -- 30%")
            print("Attempting tempo matching...")
            # (Keep existing tempo matching logic using temp files and time_stretch_audio)
            # ... (tempo matching code remains here) ...
            temp_t1_path = get_unique_filename("temp_t1", OUTPUT_DIR, ext=".wav"); sf.write(temp_t1_path, track1_np, sr1, format="WAV")
            temp_t2_path = get_unique_filename("temp_t2", OUTPUT_DIR, ext=".wav"); sf.write(temp_t2_path, track2_np, sr2, format="WAV")
            t1_tempo = detect_tempo(temp_t1_path)
            t2_tempo = detect_tempo(temp_t2_path)
            if os.path.exists(temp_t1_path): os.remove(temp_t1_path)
            if os.path.exists(temp_t2_path): os.remove(temp_t2_path)

            if progress: progress(0.4, desc="Matching tempo... -- 40%")

            if t1_tempo and t2_tempo and abs(float(t1_tempo) - float(t2_tempo)) > 1.0:
                print(f"Tempo mismatch found: T1={t1_tempo:.2f}, T2={t2_tempo:.2f}. Stretching T2...")
                temp_t2_stretch_input_path = get_unique_filename("temp_t2_stretch_input", OUTPUT_DIR)
                sf.write(temp_t2_stretch_input_path, track2_np, sr2, format="WAV") # Save track 2 before stretch

                stretched_np, stretched_sr = time_stretch_audio(temp_t2_stretch_input_path, target_tempo=t1_tempo, original_tempo=t2_tempo)
                if os.path.exists(temp_t2_stretch_input_path): os.remove(temp_t2_stretch_input_path)

                if stretched_np is not None:
                    print("Tempo stretch successful.")
                    track2_np = stretched_np
                    sr2 = stretched_sr
                    # Resample again *if* stretching changed the sample rate AND it's not the target SR
                    if sr2 != sr_for_processing:
                        print(f"Resampling T2 after stretch {sr2}->{sr_for_processing}")
                        track2_np = librosa.resample(track2_np, orig_sr=sr2, target_sr=sr_for_processing, res_type='kaiser_fast')
                        sr2 = sr_for_processing
                else:
                    print("Tempo stretch failed.")
            else:
                print("Skipping tempo matching (tempos close or detection failed).")
            print("--- TEMPO MATCHING SECTION COMPLETE ---")


        # --- Beat Alignment ---
        if progress: progress(0.5, desc="Aligning beats... -- 50%")
        # Make sure align_beats returns track2_np, sr2
        track2_np, sr2 = align_beats(track1_np, sr1, track2_np, sr2)
        print("--- BEAT ALIGNMENT SECTION COMPLETE ---")

        # --- Manual Offset ---
        if offset_ms != 0:
            if progress: progress(0.6, desc="Applying offset... -- 60%")
            print(f"Applying manual offset: {offset_ms} ms")
            offset_samples = int((offset_ms / 1000.0) * sr2)
            if offset_samples > 0: # Pad start
                pad_width = ((offset_samples, 0),) + ((0, 0),) * (track2_np.ndim - 1)
                track2_np = np.pad(track2_np, pad_width, mode='constant')
            elif offset_samples < 0: # Trim start
                 offset_samples = abs(offset_samples)
                 if offset_samples < track2_np.shape[0]:
                     track2_np = track2_np[offset_samples:]
                 else: # Offset longer than track
                     print("Warning: Offset trim exceeds track length.")
                     track2_np = np.zeros_like(track2_np[:1]) # Make it silent
            print(f"Offset applied. New T2 shape: {track2_np.shape}")

        # --- Prepare Effects ---
        if progress: progress(0.7, desc="Preparing effects... -- 70%")
        # (Keep existing effect board preparation logic)
        track1_board = None
        track2_board = None
        if PEDALBOARD_AVAILABLE:
            board1_fx = []
            # Add effects based on inputs (Gain, EQ, Comp, Reverb, Delay)
            if track1_vol != 1.0: board1_fx.append(Gain(gain_db=20 * np.log10(track1_vol)))
            if track1_low_gain != 0: board1_fx.append(LowShelfFilter(gain_db=track1_low_gain))
            if track1_high_gain != 0: board1_fx.append(HighShelfFilter(gain_db=track1_high_gain))
            if track1_threshold != -20.0 or track1_ratio != 1.0: board1_fx.append(Compressor(threshold_db=track1_threshold, ratio=track1_ratio, attack_ms=track1_attack, release_ms=track1_release))
            if track1_reverb_wetdry > 0: board1_fx.append(Reverb(room_size=track1_reverb_room, damping=track1_reverb_decay, wet_level=track1_reverb_wetdry))
            if track1_delay_wetdry > 0: board1_fx.append(Delay(delay_seconds=track1_delay_time, feedback=track1_delay_feedback, mix=track1_delay_wetdry))
            if len(board1_fx) > 0: track1_board = Pedalboard(board1_fx); print("Track 1 effects prepared.")

            board2_fx = []
            # Add effects based on inputs (Gain, EQ, Comp, Reverb, Delay)
            if track2_vol != 1.0: board2_fx.append(Gain(gain_db=20 * np.log10(track2_vol)))
            if track2_low_gain != 0: board2_fx.append(LowShelfFilter(gain_db=track2_low_gain))
            if track2_high_gain != 0: board2_fx.append(HighShelfFilter(gain_db=track2_high_gain))
            if track2_threshold != -20.0 or track2_ratio != 1.0: board2_fx.append(Compressor(threshold_db=track2_threshold, ratio=track2_ratio, attack_ms=track2_attack, release_ms=track2_release))
            if track2_reverb_wetdry > 0: board2_fx.append(Reverb(room_size=track2_reverb_room, damping=track2_reverb_decay, wet_level=track2_reverb_wetdry))
            if track2_delay_wetdry > 0: board2_fx.append(Delay(delay_seconds=track2_delay_time, feedback=track2_delay_feedback, mix=track2_delay_wetdry))
            if len(board2_fx) > 0: track2_board = Pedalboard(board2_fx); print("Track 2 effects prepared.")
        else:
            print("Pedalboard not available, skipping effects.")

        # --- Blending Tracks ---
        if progress: progress(0.9, desc="Blending tracks... -- 90%")
        blended_output_path = get_unique_filename("blend", OUTPUT_DIR, ext=".wav")
        # Ensure blend_tracks takes numpy arrays and SR, returns path, numpy array, SR
        blended_path_result, blended_audio_np, blended_sr = blend_tracks(
            track1_np=track1_np, sr1=sr1, track1_vol=1.0, track1_effects_board=track1_board, # Apply effects inside blend_tracks
            track2_np=track2_np, sr2=sr2, track2_vol=1.0, track2_effects_board=track2_board, # Apply effects inside blend_tracks
            output_path=blended_output_path
        )
        if blended_path_result is None or not os.path.exists(blended_path_result):
            raise RuntimeError(f"blend_tracks failed or did not create output file: {blended_output_path}")
        print(f"Blending successful. Output: {blended_path_result}")

        # --- Final Steps ---
        processing_time = time.time() - start_time
        if progress is not None:
            try:
                progress(1.0, desc="Completed!")
            except Exception as e:
                print(f"Warning: Progress update failed: {e}")
        print(f"Core blend_audio finished in {processing_time:.2f}s")

        # --- *** THIS IS THE CHANGED PART *** ---
        # --- SIMPLIFIED RETURN ---
        # Return the path for the audio player and the processing time string
        return blended_path_result, f"{processing_time:.2f} seconds"

    # --- Main Exception Handling ---
    except Exception as e:
        print(f"Error in blend_audio: {e}")
        traceback.print_exc()
        # Return None for the audio path and the error message for the time text
        # This helps clear the audio player and show the error in the UI
        error_message = f"Error: {e}"
        # Optionally, you could raise gr.Error(error_message) to show a popup
        # raise gr.Error(error_message)
        return None, error_message

# Global lock to prevent double processing
PROCESSING_LOCK = False

def process_audio(
    audio_file: str, task: str, model_name: str,
    drums_volume: float = 1.0, bass_volume: float = 1.0, other_volume: float = 1.0, vocals_volume: float = 1.0,
    instrumental_volume: float = 1.0, instrumental_low_gain: float = 0.0, instrumental_high_gain: float = 0.0, instrumental_reverb: float = 0.0,
    vocal_volume: float = 1.0, vocal_low_gain: float = 0.0, vocal_high_gain: float = 0.0, vocal_reverb: float = 0.0,
    trim_silence_enabled: bool = False,
    progress=None
) -> dict:
    output = {
        "original_audio": audio_file, "processed_audio": None,
        "visualizations": {}, "processing_time": 0,
        "stem_files": {"vocals": None, "instrumental": None}, "error": None
    }
    start_time = time.time()
    sr = SAMPLE_RATE

    try:
        if not audio_file or not os.path.exists(audio_file):
            raise ValueError("Input audio file not found or not provided.")

        if progress is not None:
            try:
                progress(0, desc="Initializing...")
            except Exception as e:
                print(f"Warning: Progress initialization failed: {e}")

        temp_dir = tempfile.gettempdir()
        base_name = os.path.basename(audio_file).replace(" ", "_")
        stable_path = os.path.join(temp_dir, f"stemxtract_{base_name}")
        shutil.copy(audio_file, stable_path)
        print(f"Copied file to: {stable_path}")
        if not os.path.exists(stable_path):
            raise FileNotFoundError(f"Failed to copy file to: {stable_path}")

        try:
            os.chmod(stable_path, 0o666)
            print(f"Set permissions for {stable_path}: {oct(os.stat(stable_path).st_mode)[-3:]}")
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not set permissions on {stable_path}: {e}. Continuing.")

        available_backends = torchaudio.list_audio_backends()
        print(f"Available backends: {available_backends}")
        backends_to_try = ["ffmpeg", "soundfile", "sox_io"]
        selected_backend = None
        waveform = None
        sr_orig = None
        for backend in backends_to_try:
            if backend not in available_backends:
                print(f"Backend {backend} not available.")
                continue
            try:
                # torchaudio.set_audio_backend(backend)
                print(f"Using backend: {backend}")
                waveform, sr_orig = torchaudio.load(stable_path)
                print(f"Loaded audio with sample rate: {sr_orig}, Shape: {waveform.shape}")
                if waveform.shape[-1] > 0:
                    selected_backend = backend
                    break
                print(f"Backend {backend} produced empty waveform, trying next...")
            except Exception as e:
                print(f"Backend {backend} failed: {e}")
                continue

        if waveform is None or waveform.shape[-1] == 0:
            raise ValueError("All backends failed to load audio with non-empty waveform")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA detected; using GPU acceleration.")
        elif platform.system() == "Darwin" and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("MPS detected; using Apple Silicon acceleration.")
        else:
            device = torch.device("cpu")
            print("No GPU detected; using CPU.")
        print(f"Using device: {device}")

        if progress is not None:
            try:
                progress(0.1, desc="Loading audio...")
            except Exception as e:
                print(f"Warning: Progress update failed: {e}")

        if sr_orig != sr:
            print(f"Resampling input from {sr_orig}Hz to {sr}Hz")
            waveform = torchaudio.functional.resample(waveform, sr_orig, sr)
        waveform = waveform.to(device)
        if waveform.dim() == 3 and waveform.shape[0] == 1:
            waveform_input_for_sep = waveform.squeeze(0)
        elif waveform.dim() == 2:
            waveform_input_for_sep = waveform
        else:
            raise ValueError(f"Unexpected waveform dimension: {waveform.dim()}")

        if progress is not None:
            try:
                progress(0.2, desc="Loading separation model...")
            except Exception as e:
                print(f"Warning: Progress update failed: {e}")

        # Load Demucs model
        model = pretrained.get_model(model_name)
        model.to(device)
        model.eval()

        if progress is not None:
            try:
                progress(0.3, desc="Separating stems...")
            except Exception as e:
                print(f"Warning: Progress update failed: {e}")

        # Separate stems using apply_model
        with torch.no_grad():
            estimates = apply.apply_model(model, waveform_input_for_sep.unsqueeze(0), shifts=1, split=True, overlap=0.25)[0]
        source_tensors = {
            'drums': estimates[0],
            'bass': estimates[1],
            'other': estimates[2],
            'vocals': estimates[3]
        }

        # Save vocals stem
        vocals_tensor = source_tensors.get("vocals")
        if vocals_tensor is not None:
            vocals_np = vocals_tensor.cpu().permute(1, 0).numpy()
            vocals_path = get_unique_filename("vocals", OUTPUT_DIR, ext=".wav")
            sf.write(vocals_path, vocals_np, sr, format="WAV")
            output["stem_files"]["vocals"] = (vocals_np, sr, vocals_path)
            print(f"Saved vocals stem as WAV: {vocals_path}")
        else:
            print("Warning: Vocals stem not available.")

        # Calculate and save instrumental stem
        instrumental_tensor = None
        if all(k in source_tensors for k in ['drums', 'bass', 'other']):
            instrumental_tensor = source_tensors['drums'] + source_tensors['bass'] + source_tensors['other']
            instrumental_np = instrumental_tensor.cpu().permute(1, 0).numpy()
            instrumental_path = get_unique_filename("instrumental", OUTPUT_DIR, ext=".wav")
            sf.write(instrumental_path, instrumental_np, sr, format="WAV")
            output["stem_files"]["instrumental"] = (instrumental_np, sr, instrumental_path)
            print(f"Saved instrumental stem as WAV: {instrumental_path}")
        else:
            print("Warning: Could not calculate instrumental stem.")

        if progress is not None:
            try:
                progress(0.5, desc="Processing task and applying effects...")
            except Exception as e:
                print(f"Warning: Progress update failed: {e}")
        effect_board = None
        target_tensor = None

        if task == "remove_vocals":
            target_tensor = instrumental_tensor
            if target_tensor is None:
                raise ValueError("Instrumental tensor not available.")
            if PEDALBOARD_AVAILABLE:
                board_fx = []
                if instrumental_volume != 1.0:
                    board_fx.append(Gain(gain_db=20 * np.log10(instrumental_volume)))
                if instrumental_low_gain != 0:
                    board_fx.append(LowShelfFilter(gain_db=instrumental_low_gain))
                if instrumental_high_gain != 0:
                    board_fx.append(HighShelfFilter(gain_db=instrumental_high_gain))
                if instrumental_reverb > 0:
                    board_fx.append(Reverb(room_size=instrumental_reverb))
                if len(board_fx) > 0:
                    effect_board = Pedalboard(board_fx)
        elif task == "isolate_vocals":
            target_tensor = vocals_tensor
            if target_tensor is None:
                raise ValueError("Vocals tensor not available.")
            if PEDALBOARD_AVAILABLE:
                board_fx = []
                if vocal_volume != 1.0:
                    board_fx.append(Gain(gain_db=20 * np.log10(vocal_volume)))
                if vocal_low_gain != 0:
                    board_fx.append(LowShelfFilter(gain_db=vocal_low_gain))
                if vocal_high_gain != 0:
                    board_fx.append(HighShelfFilter(gain_db=vocal_high_gain))
                if vocal_reverb > 0:
                    board_fx.append(Reverb(room_size=vocal_reverb))
                if len(board_fx) > 0:
                    effect_board = Pedalboard(board_fx)
        elif task == "mix_stems":
            target_tensor = (
                source_tensors.get('drums', torch.zeros((2, 0), device=device)) * drums_volume +
                source_tensors.get('bass', torch.zeros((2, 0), device=device)) * bass_volume +
                source_tensors.get('other', torch.zeros((2, 0), device=device)) * other_volume +
                source_tensors.get('vocals', torch.zeros((2, 0), device=device)) * vocals_volume
            )
            if target_tensor.shape[-1] == 0:
                target_tensor = torch.zeros_like(waveform_input_for_sep) * 0.0
        else:
            raise ValueError(f"Unknown task: {task}")

        processed_tensor = target_tensor
        if effect_board is not None and processed_tensor is not None and PEDALBOARD_AVAILABLE:
            print(f"Applying stem effects for task: {task}")
            try:
                target_np = processed_tensor.squeeze(0).cpu().numpy()
                if target_np.ndim == 1:
                    target_np_for_board = target_np
                elif target_np.shape[0] < target_np.shape[1]:
                    target_np_for_board = target_np.T
                else:
                    target_np_for_board = target_np
                effected_np = effect_board(target_np_for_board, sr)
                if effected_np.ndim == 1:
                    processed_tensor = torch.from_numpy(effected_np).to(device).unsqueeze(0)
                elif effected_np.shape[1] < effected_np.shape[0]:
                    processed_tensor = torch.from_numpy(effected_np.T).to(device)
                else:
                    processed_tensor = torch.from_numpy(effected_np).to(device)
            except Exception as effect_err:
                print(f"Error applying effects: {effect_err}")
                traceback.print_exc()
                processed_tensor = target_tensor
        elif processed_tensor is None:
            raise ValueError("Target tensor None before effects.")

        if processed_tensor.dim() == 1:
            processed_tensor = processed_tensor.unsqueeze(0)

        if trim_silence_enabled and processed_tensor is not None:
            if progress is not None:
                try:
                    progress(0.6, desc="Trimming silence...")
                except Exception as e:
                    print(f"Warning: Progress update failed: {e}")
            if processed_tensor.dim() == 3 and processed_tensor.shape[0] == 1:
                processed_tensor = processed_tensor.squeeze(0)
            processed_tensor = trim_silence_gpu(processed_tensor, sr)

        if progress is not None:
            try:
                progress(0.7, desc="Preparing processed audio...")
            except Exception as e:
                print(f"Warning: Progress update failed: {e}")
        if processed_tensor is not None:
            if processed_tensor.dim() == 3 and processed_tensor.shape[0] == 1:
                processed_tensor = processed_tensor.squeeze(0)
            processed_np = processed_tensor.cpu().permute(1, 0).numpy()
            processed_path = get_unique_filename("processed", OUTPUT_DIR, ext=".wav")
            sf.write(processed_path, processed_np, sr, format="WAV")
            output["processed_audio"] = (processed_np, sr, processed_path)
            print(f"Saved processed audio as wav: {processed_path}")
        else:
            output["error"] = "Processed tensor is None before final output."
            print("Error: Processed tensor is None.")

        output["processing_time"] = time.time() - start_time
        if progress is not None:
            try:
                progress(1.0, desc="Completed!")
            except Exception as e:
                print(f"Warning: Progress update failed: {e}")
        print(f"Processing finished in {output['processing_time']:.2f}s")
        return output

    except Exception as e:
        print(f"Error in process_audio main block: {str(e)}")
        traceback.print_exc()
        output["error"] = str(e)
        output["processing_time"] = time.time() - start_time if 'start_time' in locals() else 0
        return output

# Gradio Interface Helper Functions
def process_track(
    audio_file, task, model_name, apply_effects_chk,
    compression_slider, reverb_slider, distortion_slider, delay_slider, gain_slider,
    mastering_preset_dd, drums_vol, bass_vol, other_vol, vocals_vol,
    trim_silence_chk,
    progress=gr.Progress(track_tqdm=True)
):
    if not audio_file: raise gr.Error("Please upload an audio file!")
    mastering_preset_val = None if mastering_preset_dd == "None" else mastering_preset_dd
    print("\n--- Starting processing ---")
    result = process_audio(
        audio_file=audio_file, task=task, model_name=model_name, apply_effects=apply_effects_chk,
        compression_ratio=compression_slider, reverb_amount=reverb_slider, distortion_amount=distortion_slider,
        delay_amount=delay_slider, gain_db=gain_slider, mastering_preset=mastering_preset_val,
        drums_volume=drums_vol, bass_volume=bass_vol, other_volume=other_vol, vocals_volume=vocals_vol,
        trim_silence_enabled=trim_silence_chk,
        progress=progress
    )
    if result.get("error"):
        raise gr.Error(f"Processing failed: {result['error']}")
    vocals_path = result.get("stem_files", {}).get("vocals", "")
    instrumental_path = result.get("stem_files", {}).get("instrumental")
    if not instrumental_path and task == "remove_vocals": instrumental_path = result.get("processed_audio", "")
    waveform_vis_path = result.get("visualizations", {}).get("waveform")
    processed_player_audio = result.get("processed_audio")
    if not processed_player_audio or not os.path.exists(processed_player_audio): processed_player_audio = None
    print(f"--- Processing Successful ---")
    return (processed_player_audio, f"{result.get('processing_time', 0):.2f} seconds", waveform_vis_path,
            result.get("processed_audio", ""), vocals_path or "", instrumental_path or "",
            result.get("processed_audio", ""), waveform_vis_path, f"{result.get('processing_time', 0):.2f} seconds")

def update_track_visibility_factory(controls_list, audio_component, is_track1=True):
    def update_track_visibility(track_source, vocals_state, instrumental_state, vocals_path, instrumental_path):
        print(f"Updating track visibility for {'Track 1' if is_track1 else 'Track 2'}: source={track_source}")
        vocals_available = vocals_state is not None and vocals_path is not None
        instrumental_available = instrumental_state is not None and instrumental_path is not None

        audio_update_dict = {"visible": True, "value": None, "label": f"Upload Track {1 if is_track1 else 2}", "interactive": True}
        if track_source == "Upload New Track":
            pass
        elif track_source == "Use Processed Vocals" and vocals_available:
            audio_update_dict["value"] = vocals_path
            audio_update_dict["label"] = "Using Processed Vocals"
            audio_update_dict["interactive"] = False
        elif track_source == "Use Processed Instrumental" and instrumental_available:
            audio_update_dict["value"] = instrumental_path
            audio_update_dict["label"] = "Using Processed Instrumental"
            audio_update_dict["interactive"] = False
        else:
            track_source = "Upload New Track"

        choices = ["Upload New Track"]
        if vocals_available:
            choices.append("Use Processed Vocals")
        if instrumental_available:
            choices.append("Use Processed Instrumental")
        source_radio_update = gr.update(choices=choices, value=track_source)

        final_updates = [
            gr.update(**audio_update_dict),
            source_radio_update
        ] + [gr.update(visible=True, interactive=True) for _ in controls_list]
        return tuple(final_updates)
    return update_track_visibility

# Blend Audio Function with Progress
import io

def save_file(audio_data, filename_prefix):
    if isinstance(audio_data, gr.State):
        audio_data = audio_data.value
    if audio_data and isinstance(audio_data, tuple) and len(audio_data) >= 2:
        np_array, sr = audio_data[0], audio_data[1] # Explicitly get numpy array and sr
        path = get_unique_filename(filename_prefix, OUTPUT_DIR)
        sf.write(path, np_array, sr, format="WAV") # Force WAV format
        print(f"Saved {filename_prefix} to {path}")
        return path
    print(f"Invalid audio data for saving {filename_prefix}")
    return None

def save_file_vocals(audio_data):
    return save_file(audio_data, "vocals")

def save_file_instrumental(audio_data):
    return save_file(audio_data, "instrumental")

# CSS Function (unchanged)
def custom_css():
    css = f"""
    /* --- Adjusting title for Pinokio --- */
    // #app-title h1 {{ color: white !important; text-align: center; font-size: 2.5rem !important; font-weight: 600; margin-bottom: 0.5rem; }}
    /* Optional: Style the emoji span if needed, but often unnecessary with direct h1 styling */
    /* #app-title h1 .music-note {{ ... }} */
    // #app-subtitle {{ color: #e0e0e0 !important; text-align: center !important; font-size: 1.2rem !important; margin-bottom: 0.5rem; margin-top: 0.5rem; }}
    // #app-credit {{ color: #bbbbbb !important; text-align: center !important; font-size: 0.9rem !important; margin-bottom: 1rem; font-style: italic; margin-top: 0.5rem; }}
    #app-header-container {{
        text-align: center !important; /* Center everything within the container */
        margin-bottom: 1rem; /* Add some space below the header */
    }}
    #app-header-container h1 {{ /* Style the H1 (title) within the container */
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 600;
        margin-bottom: 0.5rem; /* Space below title */
    }}
    #app-header-container p:nth-of-type(1) {{ /* Style the first P (subtitle) */
        color: #e0e0e0 !important;
        font-size: 1.2rem !important;
        margin-bottom: 0.5rem; /* Space below subtitle */
        margin-top: 0; /* Reset top margin */
    }}
    #app-header-container p:nth-of-type(2) {{ /* Style the second P (credit) */
        color: #bbbbbb !important;
        font-size: 0.9rem !important;
        font-style: italic;
        margin-top: 0; /* Reset top margin */
        margin-bottom: 1rem; /* Keep space below credit */
    }}
    .section-header {{ color: white !important; font-weight: 600 !important; background: linear-gradient(90deg, {PRIMARY_COLOR}, {SECONDARY_COLOR}); padding: 6px 8px !important; border-radius: 5px; font-size: 1.1rem !important; margin-bottom: 3px !important; }}
    .results-header {{ color: white !important; font-weight: 600 !important; background: linear-gradient(90deg, {ACCENT_COLOR}, {SECONDARY_COLOR}); padding: 6px 8px !important; border-radius: 5px; font-size: 1.1rem !important; margin-bottom: 3px !important; }}
    .effects-header {{ color: white !important; font-weight: 600 !important; background: linear-gradient(90deg, {SECONDARY_COLOR}, {PRIMARY_COLOR}); padding: 6px 8px !important; border-radius: 5px; font-size: 1.1rem !important; margin-bottom: 3px !important; }}
    .mixer-header {{ color: white !important; font-weight: 600 !important; background: linear-gradient(90deg, {PRIMARY_COLOR}, {ACCENT_COLOR}); padding: 6px 8px !important; border-radius: 5px; font-size: 1.1rem !important; margin-bottom: 3px !important; }}
    .blend-header {{ color: white !important; font-weight: 600 !important; background: linear-gradient(90deg, {ACCENT_COLOR}, {PRIMARY_COLOR}); padding: 6px 8px !important; border-radius: 5px; font-size: 1.1rem !important; margin-bottom: 3px !important; }}
    .tab-header {{ font-weight: 700 !important; font-size: 1.1rem !important; color: white !important; background: #444444 !important; padding: 5px 10px !important; border-radius: 5px 5px 0 0 !important; }}
    .gradio-container {{ background-color: #2a2a2a !important; padding: 8px !important; }}
    .gr-button-primary {{ background: {PRIMARY_COLOR} !important; border-color: {PRIMARY_COLOR} !important; color: white !important; }}
    .gr-button-secondary {{ background-color: {SECONDARY_COLOR} !important; border-color: {SECONDARY_COLOR} !important; color: white !important; }}
    .gradio-accordion {{ background-color: #393939 !important; border: none !important; border-radius: 5px !important; margin-bottom: 5px !important; }}
    .gradio-accordion > .label-wrap {{ background-color: #4a4a4a !important; border-radius: 5px !important; padding: 4px 6px !important;}}
    .block-info {{ background-color: {DARK_COLOR} !important; border-radius: 5px !important; padding: 4px 8px !important; }}
    .audio-upload-container label span {{ font-weight: bold !important; color: {ACCENT_COLOR} !important; font-size: 1.1rem !important;}}
    .note-text {{ font-style: italic; color: #bbbbbb !important; font-size: 0.9rem !important; margin-bottom: 3px !important; }}
    footer {{ display: none !important; }}
    .gr-form {{ background-color: #333333 !important; border-radius: 6px !important; padding: 10px !important; margin: 3px 0 !important; border: none !important; box-shadow: none !important; }}
    .gr-box {{ background-color: #333333 !important; border-radius: 6px !important; margin: 3px 0 !important; padding: 8px !important; border: none !important; box-shadow: none !important; }}
    .gr-button {{ border-radius: 5px !important; font-weight: bold !important; padding: 6px 12px !important; }}
    .gr-slider {{ margin-top: 3px !important; margin-bottom: 3px !important; }}
    .gr-tabs {{ background-color: #2a2a2a !important; border-bottom: 2px solid #555555 !important; margin-bottom: 5px !important; padding: 0 !important; }}
    .gr-tab {{ background-color: #2a2a2a !important; border: 1px solid #555555 !important; border-bottom: none !important; border-radius: 5px 5px 0 0 !important; margin-right: 3px !important; padding: 6px 12px !important; color: white !important; font-weight: 700 !important; font-size: 1.1rem !important; }}
    .gr-tab-selected {{ background-color: #444444 !important; border-bottom: 2px solid #444444 !important; position: relative; top: 2px; }}
    #stem-downloads {{ animation: highlight 2s ease-in-out; }}
    @keyframes highlight {{ 0% {{ border: 2px solid {ACCENT_COLOR}; }} 50% {{ border: 2px solid transparent; }} 100% {{ border: 2px solid {ACCENT_COLOR}; }} }}
    #reset-blending-button {{
        background: linear-gradient(90deg, {ACCENT_COLOR}, {PRIMARY_COLOR}) !important;
        color: white !important;
        border: none !important;
        padding: 6px 12px !important;
        font-size: 1em !important;
        font-weight: bold !important;
        border-radius: 5px !important;
        transition: background 0.3s ease !important;
    }}
    #reset-blending-button:hover {{
        background: linear-gradient(90deg, {ACCENT_COLOR}, {SECONDARY_COLOR}) !important;
    }}
    """
    return css

# Gradio UI Definition Function with Full Single-Page Layout
def create_interface():
    with gr.Blocks(title="StemXtract", css=custom_css()) as app:
        # State variables (Keep as is)
        processed_audio_state = gr.State(None)
        extracted_vocals_state = gr.State(None)
        extracted_instrumental_state = gr.State(None)
        processed_vocals_path = gr.State(None)
        processed_instrumental_path = gr.State(None)
        final_output_audio = gr.State(None)
        final_output_time = gr.State("")

        # Wrap title/subtitle/credit in a Column and give IT an ID
        with gr.Column(elem_id="app-header-container"): # New wrapping Column with ID
             gr.Markdown("# 🎵 StemXtract") # Remove elem_id here
             gr.Markdown("Split, Blend, Mix! Separate stems with AI, align tracks automatically, and create unique audio effortlessly") # Remove elem_id here
             gr.Markdown("_Powered by Demucs, Gradio UI Built by TheAwakeOne_") # Remove elem_id here
        
        # How to Use Section (Accordion at the Top)
        with gr.Accordion("How to Use", open=False):
            gr.Markdown(
                """
                1.  **Upload Audio**:
                    * In the "Upload Audio" section, drag and drop an audio file or click to browse.
                    * Supported formats: MP3, WAV, FLAC, AAC (max 50MB).

                2.  **Process Audio**:
                    * In the "AI Stem Separation" section, select an AI model and task (e.g., "remove_vocals" to isolate the instrumental).
                    * Optionally adjust settings in the "Stem Effects" accordion.
                    * Click "Process Audio" to separate the stems.

                3.  **Blend Tracks (Optional)**:
                    * In the "Track Blending" section, choose sources for Track 1 and Track 2 (e.g., upload a new track or use processed vocals).
                    * Adjust volume and blending controls (e.g., match tempo, manual offset).
		            * Uncheck Match Track 1 Tempo if vocals or instrumental are too fast or too slow and set Manual Offset (best between 150-350).
                    * Best if isolated vocals are used in track 2 but it works either way.
                    * Optionally, apply effects in the "Track Effects" accordion.
                    * Click "Blend Tracks" to mix the tracks.

                4.  **Download Stems**:
                    * In the "Final Output" section, listen to the processed or blended audio.
                    * Under "Stem Downloads", click "Download Vocals" or "Download Instrumental" to save the separated stems.

                ---
                ### Acknowledgements

                This application utilizes the powerful Demucs library for music source separation. Please cite the relevant papers if you use results from this tool in your work (https://github.com/adefossez/demucs):

                ```bibtex
                @inproceedings{rouard2022hybrid,
                  title={Hybrid Transformers for Music Source Separation},
                  author={Rouard, Simon and Massa, Francisco and D{\'e}fossez, Alexandre},
                  booktitle={ICASSP 23},
                  year={2023}
                }
                ```
                ```bibtex
                @inproceedings{defossez2021hybrid,
                  title={Hybrid Spectrogram and Waveform Source Separation},
                  author={D{\'e}fossez, Alexandre},
                  booktitle={Proceedings of the ISMIR 2021 Workshop on Music Source Separation},
                  year={2021}
                }
                ```
                Demucs is released under the MIT license.
                """
            ) # End gr.Markdown

        # Main Layout: Three columns
        with gr.Row():
            # Left Column: Process Audio
            with gr.Column(scale=1):
                # Upload Audio Section
                with gr.Group():
                    gr.HTML('<div class="section-header">Upload Audio</div>')
                    gr.Markdown("Drag & drop your audio file here or browse files.", elem_classes=["note-text"])
                    audio_input = gr.Audio(type="filepath", label="Upload Audio File")
                    gr.Markdown("Supported formats: MP3, WAV, FLAC, AAC. Max file size: 50MB", elem_classes=["note-text"])

                # AI Stem Separation Section
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.HTML('<div class="section-header">AI Stem Separation</div>')
                        gr.Markdown("Select a model, task, and output format for stem separation.", elem_classes=["note-text"])
                        model_dropdown = gr.Dropdown(choices=MODELS, label="AI Model Selection", value="mdx_extra")
                        task_dropdown = gr.Dropdown(choices=["remove_vocals", "isolate_vocals", "mix_stems"], label="Task", value="remove_vocals")
                        
                        trim_silence_chk = gr.Checkbox(label="Trim Silence", value=False)
                        gr.Markdown("Note: For blending, use WAV files or processed stems to avoid format issues.", elem_classes=["note-text"])

                # Stem Effects Section
                with gr.Accordion("Stem Effects", open=False) as stem_effects_accordion:
                    gr.HTML('<div class="effects-header">Stem Effects</div>')
                    gr.Markdown("Apply effects to the output of the AI separation process.", elem_classes=["note-text"])
                    # Container for 'mix_stems' controls
                    with gr.Group(visible=(task_dropdown.value == "mix_stems")) as mix_stems_controls:
                        gr.Markdown("#### Mix Stem Volumes", elem_classes=["note-text"])
                        drums_vol = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Drums Volume")
                        bass_vol = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Bass Volume")
                        other_vol = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Other Volume")
                        vocals_vol = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Vocals Volume")
                    # Container for 'remove_vocals' (Instrumental) controls
                    with gr.Group(visible=(task_dropdown.value == "remove_vocals")) as instrumental_effects_controls:
                        gr.Markdown("#### Instrumental Effects", elem_classes=["note-text"])
                        instrumental_volume = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Volume")
                        instrumental_low_gain = gr.Slider(-12.0, 12.0, value=0.0, step=0.5, label="Low Gain (dB)")
                        instrumental_high_gain = gr.Slider(-12.0, 12.0, value=0.0, step=0.5, label="High Gain (dB)")
                        instrumental_reverb = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Reverb Amount")
                    # Container for 'isolate_vocals' controls
                    with gr.Group(visible=(task_dropdown.value == "isolate_vocals")) as vocal_effects_controls:
                        gr.Markdown("#### Vocal Effects", elem_classes=["note-text"])
                        vocal_volume = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Volume")
                        vocal_low_gain = gr.Slider(-12.0, 12.0, value=0.0, step=0.5, label="Low Gain (dB)")
                        vocal_high_gain = gr.Slider(-12.0, 12.0, value=0.0, step=0.5, label="High Gain (dB)")
                        vocal_reverb = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Reverb Amount")

                # Process Audio Button
                process_btn = gr.Button("Process Audio", variant="primary", size="lg")

            # Middle Column: Track Blending
            with gr.Column(scale=1):
                # Track Blending Setup Section
                with gr.Group():
                    gr.HTML('<div class="blend-header">Track Blending</div>')
                    reset_blend_button = gr.Button("Reset Blending", variant="secondary", elem_id="reset-blending-button")
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### Track 1", elem_classes=["note-text"])
                            track1_source = gr.Radio(["Upload New Track", "Use Processed Vocals", "Use Processed Instrumental"], label="Track 1 Source", value="Upload New Track", info="Choose the source for Track 1.")
                            track1_upload = gr.Audio(type="filepath", label="Upload Track 1", visible=True)
                            track1_placeholder = gr.Markdown("Using Processed Track", visible=False, elem_classes=["note-text"])
                            track1_volume = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Volume")
                        with gr.Column(scale=1):
                            gr.Markdown("#### Track 2", elem_classes=["note-text"])
                            track2_source = gr.Radio(["Upload New Track", "Use Processed Vocals", "Use Processed Instrumental"], label="Track 2 Source", value="Upload New Track", info="Choose the source for Track 2.")
                            track2_upload = gr.Audio(type="filepath", label="Upload Track 2", visible=True)
                            track2_placeholder = gr.Markdown("Using Processed Track", visible=False, elem_classes=["note-text"])
                            track2_volume = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Volume")

                    # Blending Controls Section
                    with gr.Group():
                        gr.HTML('<div class="blend-header">Blending Controls</div>')
                        match_tempo = gr.Checkbox(label="Match Track 1 Tempo", value=True, info="Time-stretch T2 to match T1")
                        offset_ms = gr.Slider(-500, 500, value=0, step=10, label="Manual Offset (ms)", info="Adjust T2 start relative to T1")
                        blend_progress = gr.Progress(track_tqdm=True) # Ensure it's defined before use

                    # --- Track Effects Accordion - Ensure sliders are defined HERE ---
                    with gr.Accordion("Track Effects", open=False) as track_effects_accordion: # Renamed variable for clarity
                        gr.HTML('<div class="effects-header">Track Effects</div>')
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("#### Track 1 Effects", elem_classes=["note-text"])
                                track1_low_gain = gr.Slider(-12.0, 12.0, value=0.0, step=0.5, label="Low Gain")
                                track1_high_gain = gr.Slider(-12.0, 12.0, value=0.0, step=0.5, label="High Gain")
                                track1_threshold = gr.Slider(-60.0, 0.0, value=-20.0, step=1, label="Comp Threshold") # DEFINED HERE
                                track1_ratio = gr.Slider(1.0, 10.0, value=1.0, step=0.1, label="Comp Ratio") # DEFINED HERE
                                track1_attack = gr.Slider(1.0, 100.0, value=5.0, step=1, label="Comp Attack (ms)") # DEFINED HERE
                                track1_release = gr.Slider(10.0, 500.0, value=50.0, step=10, label="Comp Release (ms)") # DEFINED HERE
                                track1_reverb_room = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Reverb Room") # DEFINED HERE
                                track1_reverb_decay = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Reverb Damping") # DEFINED HERE
                                track1_reverb_wetdry = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Reverb Amount") # DEFINED HERE
                                track1_delay_time = gr.Slider(0.01, 2.0, value=0.25, step=0.01, label="Delay Time (s)") # DEFINED HERE
                                track1_delay_feedback = gr.Slider(0.0, 0.9, value=0.3, step=0.01, label="Delay Feedback") # DEFINED HERE
                                track1_delay_wetdry = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Delay Amount") # DEFINED HERE
                            with gr.Column(scale=1):
                                gr.Markdown("#### Track 2 Effects", elem_classes=["note-text"])
                                track2_low_gain = gr.Slider(-12.0, 12.0, value=0.0, step=0.5, label="Low Gain")
                                track2_high_gain = gr.Slider(-12.0, 12.0, value=0.0, step=0.5, label="High Gain")
                                track2_threshold = gr.Slider(-60.0, 0.0, value=-20.0, step=1, label="Comp Threshold") # DEFINED HERE
                                track2_ratio = gr.Slider(1.0, 10.0, value=1.0, step=0.1, label="Comp Ratio") # DEFINED HERE
                                track2_attack = gr.Slider(1.0, 100.0, value=5.0, step=1, label="Comp Attack (ms)") # DEFINED HERE
                                track2_release = gr.Slider(10.0, 500.0, value=50.0, step=10, label="Comp Release (ms)") # DEFINED HERE
                                track2_reverb_room = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Reverb Room") # DEFINED HERE
                                track2_reverb_decay = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Reverb Damping") # DEFINED HERE
                                track2_reverb_wetdry = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Reverb Amount") # DEFINED HERE
                                track2_delay_time = gr.Slider(0.01, 2.0, value=0.25, step=0.01, label="Delay Time (s)") # DEFINED HERE
                                track2_delay_feedback = gr.Slider(0.0, 0.9, value=0.3, step=0.01, label="Delay Feedback") # DEFINED HERE
                                track2_delay_wetdry = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Delay Amount") # DEFINED HERE

                    # Blend Tracks Button
                    blend_btn = gr.Button("Blend Tracks", variant="primary", size="lg")

            # Right Column: Final Output
            with gr.Column(scale=1):
                with gr.Group():
                    gr.HTML('<div class="results-header">Final Output</div>')
                    final_audio_player = gr.Audio(label="Final Output")
                    final_time_text = gr.Textbox(label="Processing/Blending Time", interactive=False)
                with gr.Group(elem_id="stem-downloads"):
                    gr.HTML('<div class="results-header">Stem Downloads</div>')
                    gr.Markdown("Download the separated stems from your last process.", elem_classes=["note-text"])
                    with gr.Row():
                        download_vocals_btn = gr.Button("Download Vocals", variant="secondary", interactive=False)
                        download_instrumental_btn = gr.Button("Download Instrumental", variant="secondary", interactive=False)

        # --- Event Handlers ---
        # (Define helper functions FIRST)

        def update_stem_effects_visibility(selected_task):
            is_mix = selected_task == "mix_stems"; is_instrumental = selected_task == "remove_vocals"; is_vocals = selected_task == "isolate_vocals" #
            return { mix_stems_controls: gr.update(visible=is_mix), instrumental_effects_controls: gr.update(visible=is_instrumental), vocal_effects_controls: gr.update(visible=is_vocals), stem_effects_accordion: gr.update(visible=True) } #

        def process_track_wrapper(
            audio_file, task, model_name, drums_vol, bass_vol, other_vol, vocals_vol,
            instrumental_volume, instrumental_low_gain, instrumental_high_gain, instrumental_reverb,
            vocal_volume, vocal_low_gain, vocal_high_gain, vocal_reverb, trim_silence_chk,
            progress=gr.Progress(track_tqdm=True)
        ):
            if not audio_file:
                raise gr.Error("Please upload an audio file!")
            print("\n--- Starting processing ---")
            result = process_audio(
                audio_file=audio_file, task=task, model_name=model_name,
                drums_volume=drums_vol, bass_volume=bass_vol, other_volume=other_vol, vocals_volume=vocals_vol,
                instrumental_volume=instrumental_volume, instrumental_low_gain=instrumental_low_gain,
                instrumental_high_gain=instrumental_high_gain, instrumental_reverb=instrumental_reverb,
                vocal_volume=vocal_volume, vocal_low_gain=vocal_low_gain,
                vocal_high_gain=vocal_high_gain, vocal_reverb=vocal_reverb,
                trim_silence_enabled=trim_silence_chk,
                progress=progress
            )
            if result.get("error"):
                raise gr.Error(f"Processing failed: {result['error']}")
            processed_data = result.get("processed_audio")
            vocals_data = result.get("stem_files", {}).get("vocals")
            instrumental_data = result.get("stem_files", {}).get("instrumental")
            audio_player_output = processed_data[2] if processed_data else None  # Use path
            vocals_available = vocals_data is not None
            instrumental_available = instrumental_data is not None
            print(f"--- Processing Successful ---")
            return (
                audio_player_output, f"{result.get('processing_time', 0):.2f}s",
                processed_data, vocals_data, instrumental_data,
                vocals_data[2] if vocals_data else None,  # Use path
                instrumental_data[2] if instrumental_data else None,
                audio_player_output, f"{result.get('processing_time', 0):.2f}s",
                gr.update(interactive=vocals_available),
                gr.update(interactive=instrumental_available)
            )

        def reset_blending_state(): return None, "", "Upload New Track", None, "Upload New Track", None #

        # --- CORRECTED update_track_visibility_factory (Updates gr.Audio) ---
        def update_track_visibility_factory(
            controls_list, # List of effect sliders for this track
            audio_component, # e.g., track1_upload (The component to update)
            is_track1=True # Keep track identifier
            ):
            def update_track_visibility(track_source, vocals_state, instrumental_state, vocals_path, instrumental_path):
                """Handles visibility and value of the main Audio component."""
                # Determine availability
                vocals_available = vocals_state is not None
                instrumental_available = instrumental_state is not None

                # Default update values
                audio_update_dict = {"visible": True, "value": None, "label": f"Upload Track {1 if is_track1 else 2}", "interactive": True}
                source_file_path = None # Path for value if using processed

                # Logic based on selected source
                if track_source == "Upload New Track":
                    # Keep defaults: visible, interactive, no value
                    pass
                elif track_source == "Use Processed Vocals":
                    if vocals_available and vocals_path:
                        source_file_path = vocals_path
                        audio_update_dict["value"] = source_file_path
                        audio_update_dict["label"] = "Using Processed Vocals"
                        audio_update_dict["interactive"] = False # Make non-interactive
                    else:
                        # Fallback if state/path missing
                        track_source = "Upload New Track" # Reset source selection
                        # Keep defaults (upload state)
                        pass
                elif track_source == "Use Processed Instrumental":
                    if instrumental_available and instrumental_path:
                        source_file_path = instrumental_path
                        audio_update_dict["value"] = source_file_path
                        audio_update_dict["label"] = "Using Processed Instrumental"
                        audio_update_dict["interactive"] = False # Make non-interactive
                    else:
                        # Fallback if state/path missing
                        track_source = "Upload New Track" # Reset source selection
                        # Keep defaults (upload state)
                        pass
                else: # Should not happen, but fallback to upload
                    track_source = "Upload New Track"

                # Update radio choices based on availability
                choices = ["Upload New Track"]
                if vocals_available: choices.append("Use Processed Vocals")
                if instrumental_available: choices.append("Use Processed Instrumental")
                source_radio_update = gr.update(choices=choices, value=track_source)

                # Prepare list of updates for return
                # Order: audio_component, source_radio, effect_sliders...
                final_updates = [
                    gr.update(**audio_update_dict), # Update for the gr.Audio component
                    source_radio_update # Update for the gr.Radio component
                ]
                # Add updates for effect sliders (just make them visible/interactive)
                final_updates.extend([gr.update(visible=True, interactive=True) for _ in controls_list])

                return tuple(final_updates) # Return as tuple

            return update_track_visibility

        # ---Blend_audio_helper function ---
        # --- Modify blend_audio_helper ---
        def blend_audio_helper(
            # --- Arguments ---
            track1_source, track1_upload, track1_vol, track1_low_gain, track1_high_gain,
            track1_threshold, track1_ratio, track1_attack, track1_release, track1_reverb_room, track1_reverb_decay, track1_reverb_wetdry,
            track1_delay_time, track1_delay_feedback, track1_delay_wetdry,
            track2_source, track2_upload, track2_vol, track2_low_gain, track2_high_gain,
            track2_threshold, track2_ratio, track2_attack, track2_release,
            track2_reverb_room, track2_reverb_decay, track2_reverb_wetdry,
            track2_delay_time, track2_delay_feedback, track2_delay_wetdry,
            match_tempo_chk, offset_ms,
            processed_vocals_state, processed_instrumental_state
            # --- REMOVE progress=None from this line ---
        ):
            print("\n--- Inside Core blend_audio Function ---")
            # --- REMOVE ALL 'if progress:' BLOCKS and 'progress.tqdm' LOOPS ---
            try:
                start_time = time.time()
                sr_for_processing = SAMPLE_RATE
                track1_np, sr1 = None, None
                track2_np, sr2 = None, None
                blended_path_result, blended_audio_np, blended_sr = None, None, None
                track1_board, track2_board = None, None

                # --- Stage 1: Loading Track 1 ---
                # (Keep only the core loading logic)
                print("Loading Track 1...")
                if track1_source == "Upload New Track":
                    if not track1_upload: raise ValueError("Upload Track 1!")
                    track1_np, sr1 = librosa.load(track1_upload, sr=None)
                elif track1_source == "Use Processed Vocals":
                    if not processed_vocals_state: raise ValueError("No processed vocals!")
                    track1_np, sr1, _ = processed_vocals_state
                elif track1_source == "Use Processed Instrumental":
                    if not processed_instrumental_state: raise ValueError("No processed instrumental!")
                    track1_np, sr1, _ = processed_instrumental_state
                else: raise ValueError("Invalid Track 1 Source")

                # --- Stage 2: Loading Track 2 ---
                # (Keep only the core loading logic)
                print("Loading Track 2...")
                if track2_source == "Upload New Track":
                    if not track2_upload: raise ValueError("Upload Track 2!")
                    track2_np, sr2 = librosa.load(track2_upload, sr=None)
                elif track2_source == "Use Processed Vocals":
                    if not processed_vocals_state: raise ValueError("No processed vocals!")
                    track2_np, sr2, _ = processed_vocals_state
                elif track2_source == "Use Processed Instrumental":
                    if not processed_instrumental_state: raise ValueError("No processed instrumental!")
                    track2_np, sr2, _ = processed_instrumental_state
                else: raise ValueError("Invalid Track 2 Source")

                if track1_np is None or track2_np is None: raise ValueError("Failed to load track data.")

                # Resample
                if sr1 != sr_for_processing: print(f"Resampling T1 {sr1}->{sr_for_processing}"); track1_np = librosa.resample(track1_np, orig_sr=sr1, target_sr=sr_for_processing, res_type='kaiser_fast'); sr1 = sr_for_processing
                if sr2 != sr_for_processing: print(f"Resampling T2 {sr2}->{sr_for_processing}"); track2_np = librosa.resample(track2_np, orig_sr=sr2, target_sr=sr_for_processing, res_type='kaiser_fast'); sr2 = sr_for_processing
                print(f"Track 1 shape after load/resample: {track1_np.shape}, sr={sr1}")
                print(f"Track 2 shape after load/resample: {track2_np.shape}, sr={sr2}")

                # --- Stage 3: Tempo Matching ---
                if match_tempo_chk:
                        print("Attempting tempo matching...")
                        # Detect tempo directly from numpy arrays
                        t1_tempo = detect_tempo(y=track1_np, sr=sr1, name_for_log="Track 1")
                        t2_tempo = detect_tempo(y=track2_np, sr=sr2, name_for_log="Track 2")

                        # Check if tempo detection was successful and tempos differ significantly
                        if t1_tempo is not None and t2_tempo is not None and abs(t1_tempo - t2_tempo) > 1.0:
                            print(f"Tempo mismatch found: T1={t1_tempo:.2f}, T2={t2_tempo:.2f}. Stretching T2...")

                            # Perform time stretch directly on track 2's numpy array
                            stretched_np, stretched_sr = time_stretch_audio(
                                y=track2_np,
                                sr=sr2,
                                target_tempo=t1_tempo,
                                original_tempo=t2_tempo,
                                name_for_log="Track 2"
                                # max_stretch_factor is handled inside the function now
                            )

                            if stretched_np is not None:
                                print("Tempo stretch successful.")
                                track2_np = stretched_np
                                # Sample rate (sr2) should remain the same after librosa time_stretch
                                # So no resampling needed here unless time_stretch function changes SR (unlikely)
                                # sr2 = stretched_sr # This line is likely unnecessary
                            else:
                                print("Tempo stretch failed, using original Track 2.")
                        elif t1_tempo is None or t2_tempo is None:
                            print("Skipping tempo matching because tempo detection failed for one or both tracks.")
                        else:
                            print(f"Skipping tempo matching (Tempos close: T1={t1_tempo:.2f}, T2={t2_tempo:.2f})")
                        print("--- TEMPO MATCHING SECTION COMPLETE ---")
                else:
                        print("Tempo matching disabled by checkbox.")

                # --- Stage 4: Beat Alignment ---
                # (Keep the core alignment logic, remove progress loops)
                print("Aligning beats...")
                track2_np, sr2 = align_beats(track1_np, sr1, track2_np, sr2)
                print("--- BEAT ALIGNMENT SECTION COMPLETE ---")

                # --- Stage 5: Offset & Effects ---
                # (Keep the core offset/effects logic, remove progress loops)
                print("Applying offset & effects...")
                if offset_ms != 0:
                    # (Keep offset logic)
                    offset_samples = int((offset_ms / 1000.0) * sr2)
                    if offset_samples > 0: pad_width = ((offset_samples, 0),) + ((0, 0),) * (track2_np.ndim - 1); track2_np = np.pad(track2_np, pad_width, mode='constant')
                    elif offset_samples < 0:
                        offset_samples = abs(offset_samples)
                        if offset_samples < track2_np.shape[0]: track2_np = track2_np[offset_samples:]
                        else: track2_np = np.zeros_like(track2_np[:1])
                    print(f"Offset applied. New T2 shape: {track2_np.shape}")

                if PEDALBOARD_AVAILABLE:
                    # (Keep effect board preparation logic)
                    board1_fx = []; # ... add effects based on inputs ...
                    if track1_vol != 1.0: board1_fx.append(Gain(gain_db=20 * np.log10(track1_vol)))
                    if track1_low_gain != 0: board1_fx.append(LowShelfFilter(gain_db=track1_low_gain))
                    # ... (add other track 1 effects) ...
                    if len(board1_fx) > 0: track1_board = Pedalboard(board1_fx)

                    board2_fx = []; # ... add effects based on inputs ...
                    if track2_vol != 1.0: board2_fx.append(Gain(gain_db=20 * np.log10(track2_vol)))
                    if track2_low_gain != 0: board2_fx.append(LowShelfFilter(gain_db=track2_low_gain))
                    # ... (add other track 2 effects) ...
                    if len(board2_fx) > 0: track2_board = Pedalboard(board2_fx)
                    print("Effects prepared (if any).")
                else: print("Pedalboard not available, skipping effects.")

                # --- Stage 6: Blending Tracks ---
                # (Keep the core blending logic, remove progress loops)
                print("Blending tracks...")
                blended_output_path = get_unique_filename("blend", OUTPUT_DIR, ext=".wav")
                blended_path_result, blended_audio_np, blended_sr = blend_tracks(
                    track1_np=track1_np, sr1=sr1, track1_vol=1.0, track1_effects_board=track1_board,
                    track2_np=track2_np, sr2=sr2, track2_vol=1.0, track2_effects_board=track2_board,
                    output_path=blended_output_path
                )
                if blended_path_result is None: raise RuntimeError("blend_tracks failed.")
                print(f"Blending successful. Output: {blended_path_result}")

                # --- Final Steps ---
                processing_time = time.time() - start_time
                print(f"Core blend_audio_helper finished in {processing_time:.2f}s")
                return (blended_sr, blended_audio_np, blended_path_result), f"{processing_time:.2f} seconds"

            except Exception as e:
                print(f"Error in blend_audio_helper: {e}")
                traceback.print_exc()
                raise e # Re-raise
            
        # --- Start: Replace your existing blend_audio function with this ---
        def blend_audio(track1_source, track1_upload, track1_vol, track1_low_gain, track1_high_gain,
                        track1_threshold, track1_ratio, track1_attack, track1_release, track1_reverb_room, track1_reverb_decay, track1_reverb_wetdry,
                        track1_delay_time, track1_delay_feedback, track1_delay_wetdry,
                        track2_source, track2_upload, track2_vol, track2_low_gain, track2_high_gain,
                        track2_threshold, track2_ratio, track2_attack, track2_release,
                        track2_reverb_room, track2_reverb_decay, track2_reverb_wetdry,
                        track2_delay_time, track2_delay_feedback, track2_delay_wetdry,
                        match_tempo_chk, offset_ms,
                        processed_vocals_state, processed_instrumental_state,
                        processed_vocals_path, processed_instrumental_path
                        ): # <--- Colon ensures correct syntax
            # --- Code indented correctly below ---
            print("\n--- Inside blend_audio Function ---")
            try:
                # Call the helper function which now does all the main work
                (blended_sr, blended_audio_np, blended_path_result), processing_time_str = blend_audio_helper(
                    track1_source, track1_upload, track1_vol, track1_low_gain, track1_high_gain,
                    track1_threshold, track1_ratio, track1_attack, track1_release, track1_reverb_room, track1_reverb_decay, track1_reverb_wetdry,
                    track1_delay_time, track1_delay_feedback, track1_delay_wetdry,
                    track2_source, track2_upload, track2_vol, track2_low_gain, track2_high_gain,
                    track2_threshold, track2_ratio, track2_attack, track2_release,
                    track2_reverb_room, track2_reverb_decay, track2_reverb_wetdry,
                    track2_delay_time, track2_delay_feedback, track2_delay_wetdry,
                    match_tempo_chk, offset_ms,
                    processed_vocals_state, processed_instrumental_state
                    # No progress argument is passed to the helper
                )

                # Directly return the results from the helper function
                # The helper returns (tuple(sr, np, path), time_str)
                if blended_path_result and os.path.exists(blended_path_result):
                    print(f"Blend successful. Returning path: {blended_path_result}, time: {processing_time_str}")
                    # Return path for audio player, time string for textbox
                    return blended_path_result, processing_time_str
                else:
                    # This case means the helper failed to produce the file path
                    print("Blending helper failed to return a valid output path.")
                    raise RuntimeError("Blending failed to produce a valid output file in helper.")

            except Exception as e:
                # Catch any errors from the helper and report them in the UI
                print(f"Error caught in blend_audio: {e}")
                traceback.print_exc()
                error_message = f"Error: {str(e)}"
                # Return None for the audio player path, and the error message for the time textbox
                return None, error_message
            
        # --- Event Listener Connections ---

        # Task dropdown visibility control
        task_dropdown.change( fn=update_stem_effects_visibility, inputs=[task_dropdown], outputs=[mix_stems_controls, instrumental_effects_controls, vocal_effects_controls, stem_effects_accordion] )

        # Process button click
        process_btn.click(
            fn=process_track_wrapper,
            inputs=[
                audio_input, task_dropdown, model_dropdown,
                drums_vol, bass_vol, other_vol, vocals_vol,
                instrumental_volume, instrumental_low_gain, instrumental_high_gain, instrumental_reverb,
                vocal_volume, vocal_low_gain, vocal_high_gain, vocal_reverb,
                trim_silence_chk
            ],
            outputs=[
                final_audio_player, final_time_text, processed_audio_state,
                extracted_vocals_state, extracted_instrumental_state,
                processed_vocals_path, processed_instrumental_path,
                final_output_audio, final_output_time,
                download_vocals_btn, download_instrumental_btn
            ]
        )

        # Reset button click
        reset_blend_button.click( fn=reset_blending_state, inputs=None, outputs=[final_audio_player, final_time_text, track1_source, track1_upload, track2_source, track2_upload] )

        # Define track control lists
        track1_controls_list = [ track1_volume, track1_low_gain, track1_high_gain, track1_threshold, track1_ratio, track1_attack, track1_release, track1_reverb_room, track1_reverb_decay, track1_reverb_wetdry, track1_delay_time, track1_delay_feedback, track1_delay_wetdry ]
        track2_controls_list = [ track2_volume, track2_low_gain, track2_high_gain, track2_threshold, track2_ratio, track2_attack, track2_release, # Included track2_release here
                                  track2_reverb_room, track2_reverb_decay, track2_reverb_wetdry, track2_delay_time, track2_delay_feedback, track2_delay_wetdry ]

        # --- CORRECTED .change() calls (No Plot/Placeholder outputs) ---
        track1_source.change(
            fn=update_track_visibility_factory( # Pass only audio component
                track1_controls_list, track1_upload, is_track1=True
            ),
            inputs=[track1_source, extracted_vocals_state, extracted_instrumental_state, processed_vocals_path, processed_instrumental_path],
            outputs=[ # Order must match return tuple: audio_comp, source_radio, sliders...
                track1_upload, # The gr.Audio component to update
                track1_source, # The gr.Radio component
            ] + track1_controls_list # Append effect sliders
        )
        track2_source.change(
            fn=update_track_visibility_factory( # Pass only audio component
                track2_controls_list, track2_upload, is_track1=False
            ),
            inputs=[track2_source, extracted_vocals_state, extracted_instrumental_state, processed_vocals_path, processed_instrumental_path],
            outputs=[ # Order must match return tuple
                track2_upload, # The gr.Audio component
                track2_source, # The gr.Radio component
            ] + track2_controls_list # Append effect sliders
        )

        # Blend button click (ensure inputs list length matches wrapper expectation = 35)
        blend_btn.click(
            fn=blend_audio,
            inputs=[
                track1_source, track1_upload, track1_volume, track1_low_gain, track1_high_gain,
                track1_threshold, track1_ratio, track1_attack, track1_release, track1_reverb_room, track1_reverb_decay, track1_reverb_wetdry,
                track1_delay_time, track1_delay_feedback, track1_delay_wetdry,
                track2_source, track2_upload, track2_volume, track2_low_gain, track2_high_gain,
                track2_threshold, track2_ratio, track2_attack, track2_release, # Included track2_release
                track2_reverb_room, track2_reverb_decay, track2_reverb_wetdry,
                track2_delay_time, track2_delay_feedback, track2_delay_wetdry,
                match_tempo, offset_ms,
                extracted_vocals_state, extracted_instrumental_state,
                processed_vocals_path, processed_instrumental_path,
            ],
            outputs=[final_audio_player, final_time_text],
            
        )

        # Download buttons
        download_vocals_btn.click(
            fn=save_file_vocals, inputs=[extracted_vocals_state], outputs=[gr.File(label="Download Vocals")]
        )
        download_instrumental_btn.click(
            fn=save_file_instrumental, inputs=[extracted_instrumental_state], outputs=[gr.File(label="Download Instrumental")]
        )

        # Return the app instance
        return app

if __name__ == "__main__":
    # Create output dir if needed
    if not os.path.exists(OUTPUT_DIR):
        try: os.makedirs(OUTPUT_DIR); print(f"Created output directory: {OUTPUT_DIR}") #
        except OSError as e: print(f"Error creating output directory {OUTPUT_DIR}: {e}") #
    # Ensure helper functions are defined if they were global
    # (e.g., blend_tracks, align_beats, detect_tempo, time_stretch_audio etc.)
    print("Creating Gradio interface...") #
    app_instance = create_interface()
    print("Launching Gradio interface...") #
    # Consider adding share=False explicitly if not needed
    app_instance.launch(debug=True)  # Removed share=False to match original maybe? Add if needed.