# StemXtract: AI Stem Separation & Blending

<p align="center">
  <img src="icon.png" alt="StemXtract Logo" width="150"/>
</p>

StemXtract is a user-friendly application built with Gradio that allows you to:
1.  Separate audio tracks into vocal and instrumental stems using AI models (Demucs).
2.  Blend different audio tracks (e.g., vocals from one song with an instrumental from another).
3.  Adjust volume levels, tempo, and timing for creative mixing.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/TheAwaken1/StemXtract.git](https://github.com/TheAwaken1/StemXtract.git)
    cd StemXtract
    ```

2.  **Create and Activate a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    ```
    * On Windows:
        ```bash
        venv\Scripts\activate
        ```
    * On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install PyTorch:** (Required for Demucs) Choose the appropriate command for your system.

    * **For GPU (NVIDIA CUDA):**
        *Find the correct command for your specific CUDA version on the [PyTorch website](https://pytorch.org/get-started/locally/).*
        *Example for CUDA 11.8:*
        ```bash
        pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
        ```
        *Example for CUDA 12.1 or later:*
        ```bash
        pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
        ```

    * **For CPU Only** (Windows, macOS, Linux):
        ```bash
        pip install torch torchvision torchaudio
        ```

4.  **Install Demucs:**
    ```bash
    pip install demucs==4.0.1
    ```

5.  **Install Other Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Launch the app using the main script:

```bash
python stemxtract.py
```

This will start a local web server. Look for a URL in the terminal output (e.g., Running on local URL: http://127.0.0.1:7860) and open it in your web browser.

Usage
Upload Audio:
In the "Upload Audio" section, drag and drop an audio file or click to browse.
Supported formats: MP3, WAV, FLAC, AAC. (Max file size may be configured, e.g., 50MB).

Process Audio (AI Stem Separation):
In the "AI Stem Separation" section, select an AI model (e.g., mdx_extra).
Select a task:
remove_vocals: Isolate the instrumental.
isolate_vocals: Isolate the vocals (may produce both).
mix_stems: (If applicable) Apply effects or remix stems based on the model.

Optionally, enable "Trim Silence" to remove silence from the start and end.
Click "Process Audio".

Blend Tracks (Optional):
In the "Track Blending" section, choose sources for Track 1 and Track 2:
"Upload New Track": Upload a new audio file.
"Use Processed Vocals": Use the vocals from the last processed audio.
"Use Processed Instrumental": Use the instrumental from the last processed audio.
Adjust volume levels.
Adjust blending controls:
"Match Track 1 Tempo": Time-stretch Track 2 to match Track 1’s tempo.
"Manual Offset (ms)": Adjust Track 2’s start time relative to Track 1.
Click "Blend Tracks".

Download Stems / Output:
In the "Final Output" section, listen to the processed or blended audio.
Under "Stem Downloads," click "Download Vocals" or "Download Instrumental" to save the separated stems (usually as WAV files).

Example WorkflowUpload a song (e.g., song.mp3) in the "Upload Audio" section.
Select the mdx_extra model and the remove_vocals task, then click "Process Audio."
Wait for processing. Vocals and Instrumental should become available.
In the "Track Blending" section:
Set Track 1 to "Upload New Track" and upload a different instrumental track (e.g., new_instrumental.wav).
Set Track 2 to "Use Processed Vocals".
Adjust volume levels and enable "Match Track 1 Tempo".
Click "Blend Tracks".
Listen to the blended result in the "Final Output" section.
Download the original separated vocals using the "Download Vocals" button if needed.

TroubleshootingBlending Takes Too Long: Ensure sufficient system memory (RAM) and CPU resources. Tempo matching can be resource-intensive. The app might downscale audio (e.g., to 22050 Hz) during tempo matching to manage memory.
No Vocals/Instrumental Available for Blending: Make sure you successfully ran a processing task (remove_vocals or isolate_vocals) first.
Errors During Processing: Check the console/terminal where you ran python stemxtract.py for detailed error messages. Ensure all dependencies (e.g., resampy, pedalboard, PyTorch, Demucs) are correctly installed in your virtual environment.

LicenseThis project is licensed under the MIT License. See the LICENSE file for details.

AcknowledgmentsUI 

Design: @TheAwakeOne619

This application utilizes the powerful Demucs library for music source separation. Please cite the relevant papers if you use results from this tool in your work: https://github.com/adefossez/demucs

```bibtex
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
