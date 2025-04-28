module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "echo Update not implemented for local setup",
        path: "app"
      }
    }
    // {
    //   method: "shell.run",
    //   "params": {
    //     venv: "env",
    //     path: "app",
    //     message": "pip install -r requirements.txt"
    //   }
    // }
  ]
};