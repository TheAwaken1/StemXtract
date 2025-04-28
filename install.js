module.exports = {
  run: [
    // Edit this step to customize the git repository to use
    {
      method: "shell.run",
      params: {
        message: [
            // Clone the specific branch into the 'app' directory
      "git clone -b pinokio-integration https://github.com/TheAwaken1/StemXtract.git app"
        ]
      }
    },
    
    // Delete this step if your project does not use torch
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",                // Edit this to customize the venv folder path
          path: "app",                // Edit this to customize the path to start the shell from
          // xformers: true   // uncomment this line if your project requires xformers
          // triton: true   // uncomment this line if your project requires triton
          sageattention2: true   // uncomment this line if your project requires sageattention
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",                // Edit this to customize the venv folder path
        path: "app",                // Edit this to customize the path to start the shell from
        message: [
          "uv pip install demucs==4.0.1",
          "uv pip install gradio==5.27.0",
          "uv pip install resampy==0.4.3" 
        ]
      }
    },
    // Edit this step with your custom install commands
    {
      method: "shell.run",
      params: {
        venv: "env",                // Edit this to customize the venv folder path
        path: "app",                // Edit this to customize the path to start the shell from
        message: [
          "uv pip install gradio devicetorch",
          "uv pip install -r requirements.txt"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",                // Edit this to customize the venv folder path
        path: "app",                // Edit this to customize the path to start the shell from
        message: [
          "conda install ffmpeg -c conda-forge --yes"
        ]
      }
    },
  ]
}
