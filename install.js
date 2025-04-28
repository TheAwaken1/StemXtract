module.exports = {
  run: [
    // Edit this step to customize the git repository to use
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/TheAwaken1/StemXtract.git app",
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
          // sageattention: true   // uncomment this line if your project requires sageattention
        }
      }
    },
    // Edit this step with your custom install commands
    // Step 3: Install Demucs and other specific dependencies (if needed beyond requirements.txt)
    // Ensure 'path' points to the 'app' directory
    {
      method: "shell.run",
      params: {
        venv: "env", // Use the same venv
        path: "app", // Run pip install inside the 'app' directory
        message: [
          // Use uv pip or pip based on your Pinokio setup
          "uv pip install demucs==4.0.1",
          "uv pip install gradio==5.27.0", // Ensure this version is compatible with main branch
          "uv pip install resampy==0.4.3" // Ensure this version is compatible
          // Add any other specific dependencies needed by the main branch but not in requirements.txt
        ]
      }
    },

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
    // Step 5: Install ffmpeg (if needed by the main branch)
    // This step might not need venv or path if it's a system-level conda install
    {
      method: "shell.run",
      params: {
        // venv: "env", // Usually not needed for conda system installs
        // path: "app", // Usually not needed for conda system installs
        message: [
          "conda install ffmpeg -c conda-forge --yes"
        ]
      }
    },

    // Step 6: Notify user
    {
      method: "notify",
      params: {
        html: "Installation complete. Click the 'start' tab to launch StemXtract!"
      }
    }
  ]
}
