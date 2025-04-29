module.exports = {
  run: [
    {
      "method": "shell.run",
      "params": {
        "message": "rmdir /s /q app || echo App directory not found"
      }
    },
    {
      "method": "shell.run",
      "params": {
        "message": "git clone -b main https://github.com/TheAwaken1/StemXtract.git app"
      }
    },
    {
      "method": "shell.run",
      "params": {
        "uri": "torch.js",
        "params": {
          "venv": "env",
          "path": "app"
        }
      }
    },
    {
      "method": "shell.run",
      "params": {
        "venv": "env",
        "path": "app",
        "message": [
          "uv pip install demucs==4.0.1",
          "uv pip install gradio==5.27.0",
          "uv pip install resampy==0.4.3"
        ]
      }
    },
    {
      "method": "shell.run",
      "params": {
        "venv": "env",
        "path": "app",
        "message": [
          "uv pip install gradio devicetorch",
          "uv pip install -r requirements.txt"
        ]
      }
    },
    {
      "method": "shell.run",
      "params": {
        "message": [
          "conda install ffmpeg -c conda-forge --yes"
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
}
