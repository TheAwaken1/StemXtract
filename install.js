module.exports = {
  run: [
    // Step 1: Clone the default (main) branch of the repository into the 'app' directory
    {
      method: "shell.run",
      params: {
        message: [
          // Removed the '-b pinokio-integration' flag to clone the default branch
          // Clone into a directory named 'app'
          "git clone https://github.com/TheAwaken1/StemXtract.git app"
        ]
        // Optional: Define 'path' if the clone needs to happen relative to a different base directory
        // path: "." // Example: clone in the root script directory
      }
    },

    // Step 2: Install Torch (using torch.js)
    // Make sure the 'path' here correctly points to where the code was cloned ('app')
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env", // Virtual environment directory name
          path: "app", // Run torch.js relative to the cloned 'app' directory
          // sageattention2: true // Uncomment if needed by your main branch
        }
      }
    },

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

    // Step 4: Install dependencies from requirements.txt (from the main branch)
    // Ensure 'path' points to the 'app' directory
    {
      method: "shell.run",
      params: {
        venv: "env", // Use the same venv
        path: "app", // Run pip install inside the 'app' directory
        message: [
          // Assuming requirements.txt exists in the main branch
          "uv pip install -r requirements.txt"
          // Note: The previous "pip install gradio devicetorch" might be redundant
          // if these are already in requirements.txt. Clean up as needed.
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
