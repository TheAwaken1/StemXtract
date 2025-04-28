module.exports = {
  "daemon": true,
  "run": [
    {
      "method": "shell.run",
      "params": {
        "venv": "env",
        "path": "app",
        "message": "python stemxtract.py", // Corrected script name
        "on": [{
          // The regular expression pattern to monitor.
          // When this pattern occurs in the shell terminal, the shell will return,
          // and the script will go onto the next step.
          "event": "/http:\\/\\/\\S+/",   

          // "done": true will move to the next step while keeping the shell alive.
          // "kill": true will move to the next step after killing the shell.
          "done": true
        }]
      }
    },
    {
      "method": "local.set",
      "params": {
        "url": "{{input.event[0]}}"
      }
    },
    {
      "method": "browser.open",
      "params": {
        "uri": "{{input.event[0]}}"
      }
    }
  ]
}