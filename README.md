StemXtract
🎵 StemXtract is an audio processing tool designed to separate, mix, and master audio tracks with ease. This app allows you to isolate vocals and instrumentals, blend tracks with tempo matching and beat alignment, apply effects, and download separated stems—all through an intuitive web interface powered by Gradio.

Features
AI Stem Separation: Isolate vocals and instrumentals from audio files using AI models (e.g., mdx_extra).
Stem Effects: Add some minimal effects to your process audio.
Track Blending: Mix two tracks with automatic tempo matching and beat alignment for seamless blending.
Track Effects: Apply effects (compression, reverb, distortion, delay, gain).
Stem Downloads: Download separated vocals and instrumentals as WAV files.
User-Friendly Interface: Built with Gradio for an interactive web-based experience, including a "How to Use" guide.
Trim Silence: Optionally remove silence from the start and end of processed tracks.
Customizable Mixing: Adjust volume levels, apply manual offsets, and fine-tune effects for each track.

# Installation
Follow these steps to set up and run the **StemXtract** project, which provides a Gradio-based interface for audio stem separation using AI model Demucs.

## Prerequisites

- **Python**: Version 3.8 or higher (tested up to 3.12; 3.13 may work but is not officially supported). Python 3.9 or 3.10 is recommended for optimal compatibility.

- **Operating System**: Windows, macOS, or Linux.

- **Optional (for GPU acceleration)**: NVIDIA GPU with CUDA 12.8 support, plus the latest NVIDIA drivers and CUDA Toolkit ([download here](https://developer.nvidia.com/cuda-downloads)).

- **Optional (FFmpeg)**: For audio processing with Demucs. Install it based on your platform.

Notes
PyTorch and Demucs: These are installed separately due to specific index requirements for CUDA support. Ensure your Python version and PyTorch version are compatible.

Setup
Clone the Repository:

```bash
git clone https://github.com/your-username/StemXtract.git
cd StemXtract

Create a Virtual Environment (recommended to isolate dependencies):

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install PyTorch (required for Demucs and AI stem separation) and demucs seperately:

- For GPU (NVIDIA CUDA 12.8, Windows/Linux):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

For CPU (Windows, macOS, Linux, or non-GPU systems):

```bash
pip install torch torchvision torchaudio

Install Demucs (for AI stem separation):

```bash
pip install demucs==4.0.1

# Install depencies
pip install -r requirements.txt

Run the App: Launch the app using the main script:

```bash
python stemxtract.py

This will start a local web server, and Gradio will provide a URL (e.g., http://127.0.0.1:7860) to access the app in your browser.

Usage
1. Upload Audio:
In the "Upload Audio" section, drag and drop an audio file or click to browse.
Supported formats: MP3, WAV, FLAC, AAC (max file size: 50MB).

2. Process Audio:
In the "AI Stem Separation" section, select an AI model (e.g., mdx_extra) and a task:
remove_vocals: Isolate the instrumental.
isolate_vocals: Isolate the vocals.
mix_stems: Apply stem effects to the original track.
Optionally, enable "Trim Silence" to remove silence from the start and end.
Click "Process Audio" to separate the stems.

3. Blend Tracks (Optional):
In the "Track Blending" section, choose sources for Track 1 and Track 2:
"Upload New Track": Upload a new audio file.
"Use Processed Vocals": Use the vocals from the processed audio.
"Use Processed Instrumental": Use the instrumental from the processed audio.
Adjust volume levels and blending controls:
"Match Track 1 Tempo": Time-stretch Track 2 to match Track 1’s tempo.
"Manual Offset (ms)": Adjust Track 2’s start relative to Track 1.
Click "Blend Tracks" to mix the tracks.

4. Download Stems:
In the "Final Output" section, listen to the processed or blended audio.
Under "Stem Downloads," click "Download Vocals" or "Download Instrumental" to save the separated stems as WAV files.

Example Workflow
Upload a song (e.g., song.mp3) in the "Upload Audio" section.
Select the mdx_extra model and the remove_vocals task, then click "Process Audio."
In the "Track Blending" section, set Track 1 to "Upload New Track" (e.g., upload an instrumental track) and Track 2 to "Use Processed Vocals."
Adjust the volume levels and enable "Match Track 1 Tempo," then click "Blend Tracks."
Listen to the blended output in the "Final Output" section and download the separated vocals using the "Download Vocals" button.

Troubleshooting
Blending Takes Too Long: Ensure your system has sufficient memory and CPU resources. The app downscales audio to 22050 Hz during tempo matching and beat alignment to reduce memory usage.
No Vocals/Instrumental Available: If the "Use Processed Vocals" or "Use Processed Instrumental" options are not available, ensure you’ve processed an audio file with a task that generates the desired stem (remove_vocals for vocals, isolate_vocals for both).
Errors During Processing: Check the console logs for detailed error messages. Ensure all dependencies (e.g., resampy, pedalboard) are installed.

License
This project is licensed under the MIT License. See the  file for details.

Acknowledgments
UI Design: @TheAwakeOne619

This application utilizes the powerful Demucs library for music source separation. Please cite the relevant papers if you use results from this tool in your work (https://github.com/adefossez/demucs):

@inproceedings{rouard2022hybrid,
  title={Hybrid Transformers for Music Source Separation},
  author={Rouard, Simon and Massa, Francisco and D{'e}fossez, Alexandre},
  booktitle={ICASSP 23},
  year={2023}
}

@inproceedings{defossez2021hybrid,
  title={Hybrid Spectrogram and Waveform Source Separation},
  author={D{'e}fossez, Alexandre},
  booktitle={Proceedings of the ISMIR 2021 Workshop on Music Source Separation},
  year={2021}
}

