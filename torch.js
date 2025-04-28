module.exports = {
  run: [
    // NVIDIA 50-series Windows
    {
      "when": "{{platform === 'win32' && gpu === 'nvidia' && kernel.gpu_model && /50\\d{2,}/.test(kernel.gpu_model) }}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128",
          "{{args && args.triton ? 'uv pip install -U --pre triton-windows' : ''}}",
          "{{args && args.sageattention ? 'uv pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1+cu128torch2.7.0-cp310-cp310-win_amd64.whl' : ''}}"
        ]
      },
      "next": null
    },
    // NVIDIA 50-series Linux
    {
      "when": "{{platform === 'linux' && gpu === 'nvidia' && kernel.gpu_model && /50\\d{2,}/.test(kernel.gpu_model) }}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
        ]
      },
      "next": null
    },
    // Windows NVIDIA (non-50-series)
    {
      "when": "{{platform === 'win32' && gpu === 'nvidia' && !(kernel.gpu_model && /50\\d{2,}/.test(kernel.gpu_model))}}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "uv pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html"
        ]
      }
    },
    // Windows AMD
    {
      "when": "{{platform === 'win32' && gpu === 'amd'}}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "uv pip install torch-directml==2.0.0 torchaudio==2.0.1 torchvision==0.15.1 numpy==1.26.4"
        ]
      }
    },
    // Windows CPU
    {
      "when": "{{platform === 'win32' && (gpu !== 'nvidia' && gpu !== 'amd')}}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "uv pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 numpy==1.26.4 --index-url https://download.pytorch.org/whl/cpu"
        ]
      }
    },
    // Mac
    {
      "when": "{{platform === 'darwin'}}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "uv pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 numpy==1.26.4"
        ]
      }
    },
    // Linux NVIDIA (non-50-series)
    {
      "when": "{{platform === 'linux' && gpu === 'nvidia' && !(kernel.gpu_model && /50\\d{2,}/.test(kernel.gpu_model))}}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "uv pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html"
        ]
      }
    },
    // Linux ROCm (AMD)
    {
      "when": "{{platform === 'linux' && gpu === 'amd'}}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "uv pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/rocm5.6"
        ]
      }
    },
    // Linux CPU
    {
      "when": "{{platform === 'linux' && (gpu !== 'amd' && gpu !== 'nvidia')}}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "uv pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu"
        ]
      }
    }
  ]
};