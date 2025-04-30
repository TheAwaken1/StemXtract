module.exports = {
  "run": [
    // Step 1: Remove existing app directory
    {
      "method": "shell.run",
      "params": {
        "message": "{{platform === 'win32' ? 'rmdir /s /q app || echo App directory not found' : 'rm -rf app || echo App directory not found'}}"
      }
    },
    // Step 2: Clone repository
    {
      "method": "shell.run",
      "params": {
        "message": "git clone -b main https://github.com/TheAwaken1/StemXtract.git app"
      }
    },
    // Step 3: Install PyTorch and related packages (includes venv creation)
    {
      "method": "script.start",
      "params": {
        "uri": "torch.js",
        "params": {
          "venv": "env",                // Edit this to customize the venv folder path
          "path": "app",                // Edit this to customize the path to start the shell from
          // xformers: true   // uncomment this line if your project requires xformers
          // triton: true   // uncomment this line if your project requires triton
          "sageattention2": true   // uncomment this line if your project requires sageattention
        }
      }
    },
    // Step 4: Install core Python dependencies
    {
      "method": "shell.run",
      "params": {
        "venv": "env",                // Edit this to customize the venv folder path
        "path": "app",                // Edit this to customize the path to start the shell from
        "message": [
          "uv pip install demucs==4.0.1",
          "uv pip install gradio==5.27.0",
          "uv pip install resampy==0.4.3" 
        ]
      }
    },
    // Step 5: Install additional Python dependencies
    {
      "method": "shell.run",
      "params": {
        "venv": "env",                // Edit this to customize the venv folder path
        "path": "app",                // Edit this to customize the path to start the shell from
        "message": [
          "uv pip install gradio devicetorch",
          "uv pip install -r requirements.txt"
        ]
      }
    },
    // Step 6: Install system dependencies (ffmpeg, libsndfile, cmake, gfortran)
    {
      "when": "{{platform === 'darwin'}}",
      "method": "shell.run",
      "params": {
        "message": [
          // Install ffmpeg, libsndfile, cmake, and gcc (includes gfortran) via Homebrew, with Conda fallback for ffmpeg
          "brew install ffmpeg libsndfile cmake gcc || conda install ffmpeg -c conda-forge --yes || echo 'System dependencies (ffmpeg, libsndfile, cmake, gcc) installation failed. Please install manually with: brew install ffmpeg libsndfile cmake gcc'"
        ]
      }
    },
    {
      "when": "{{platform === 'linux'}}",
      "method": "shell.run",
      "params": {
        "message": [
          // Try apt-get for Debian/Ubuntu, yum for CentOS, with Conda fallback for ffmpeg
          "sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1-dev cmake gfortran || sudo yum install -y ffmpeg libsndfile-devel cmake gcc-gfortran || conda install ffmpeg -c conda-forge --yes || echo 'System dependencies (ffmpeg, libsndfile, cmake, gfortran) installation failed. Please install manually with: sudo apt-get install ffmpeg libsndfile1-dev cmake gfortran or sudo yum install ffmpeg libsndfile-devel cmake gcc-gfortran'"
        ]
      }
    },
    {
      "when": "{{platform === 'win32'}}",
      "method": "shell.run",
      "params": {
        "message": [
          // Windows only needs ffmpeg, as libsndfile, cmake, and gfortran are handled by Conda or not required
          "conda install ffmpeg -c conda-forge --yes || echo 'FFmpeg installation failed. Please install manually with: conda install ffmpeg'"
        ]
      }
    },
    // Step 7: Notify user
    {
      "method": "notify",
      "params": {
        "html": "Installation complete. Click the 'start' tab to launch StemXtract!"
      }
    }
  ]
};
