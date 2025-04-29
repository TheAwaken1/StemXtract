module.exports = {
  "run": [
    {
      "method": "shell.run",
      "params": {
        "message": "rmdir /s /q app || echo App directory not found"
      }
    },
    {
      "method": "shell.run",
      "params": {
        "message": "git clone -b pinokio-integration https://github.com/TheAwaken1/Intelligent-Excel-Analyzer.git app"
      }
    },
    {
      "method": "shell.run",
      "params": {
        "message": "python -m venv env"
      }
    },
    {
      "method": "script.start",
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
        "message": "pip install -r requirements.txt"
      }
    },
    {
      "when": "{{gpu === 'nvidia'}}",
      "method": "shell.run",
      "params": {
        "venv": "env",
        "path": "app",
        "message": "pip install bitsandbytes"
      }
    }
  ]
}
