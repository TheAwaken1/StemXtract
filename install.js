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
    // Edit this step with your custom install commands
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
    {
      "when": "{{platform === 'darwin'}}",
      "method": "shell.run",
      "params": {
        "message": [
          // Try Homebrew first, fallback to Conda
          "brew install ffmpeg || conda install ffmpeg -c conda-forge --yes || echo 'FFmpeg installation failed, please install manually'"
        ]
      }
    },
    {
      "when": "{{platform === 'linux'}}",
      "method": "shell.run",
      "params": {
        "message": [
          // Try apt-get for Debian/Ubuntu, yum for CentOS, fallback to Conda
          "sudo apt-get update && sudo apt-get install -y ffmpeg || sudo yum install -y ffmpeg || conda install ffmpeg -c conda-forge --yes || echo 'FFmpeg installation failed, please install manually'"
        ]
      }
    },
    {
      "method": "notify",
      "params": {
        "html": "Installation complete. Click the 'start' tab to launch StemXtract!"
      }
    }
  ]
};
