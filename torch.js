module.exports = {
  run: [
    // NVIDIA 50-series Windows
    {
      "when": "{{platform === 'win32' && gpu === 'nvidia' && kernel.gpus && kernel.gpus.find(x => / 50.+/.test(x.model))}}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128",
          "{{args && args.triton ? 'uv pip install -U --pre triton-windows' : ''}}",
          "{{args && args.sageattention ? 'uv pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1+cu128torch2.7.0-cp310-cp310-win_amd64.whl' : ''}}"
        ]
      },
      "next": null
    },
    // NVIDIA 50-series Linux
    {
      "when": "{{platform === 'linux' && gpu === 'nvidia' && kernel.gpus && kernel.gpus.find(x => / 50.+/.test(x.model))}}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128",
          "{{args && args.triton ? 'uv pip install -U --pre triton' : ''}}",
          "{{args && args.sageattention ? 'uv pip install git+https://github.com/thu-ml/SageAttention.git@2.1.1' : ''}}"
        ]
      },
      "next": null
    },
    // Windows NVIDIA (non-50-series)
    {
      "when": "{{platform === 'win32' && gpu === 'nvidia' && !(kernel.gpus && kernel.gpus.find(x => / 50.+/.test(x.model)))}}",
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
          "uv pip install torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.1+cpu numpy==1.26.4 --index-url https://download.pytorch.org/whl/cpu"
        ]
      }
    },
    // Mac (Updated for MPS and compatibility)
    {
      "when": "{{platform === 'darwin'}}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          // Use torch 2.3.1 for better MPS support and compatibility with torchaudio
          "uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 numpy==1.26.4"
        ]
      }
    },
    // Linux NVIDIA (non-50-series)
    {
      "when": "{{platform === 'linux' && gpu === 'nvidia' && !(kernel.gpus && kernel.gpus.find(x => / 50.+/.test(x.model)))}}",
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
          "uv pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 numpy==1.26.4 --index-url https://download.pytorch.org/whl/rocm5.6"
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
          "uv pip install torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.1+cpu numpy==1.26.4 --index-url https://download.pytorch.org/whl/cpu"
        ]
      }
    }
  ]
};
