module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    {
      method: "shell.run",
      params: {
        path: "app",
        message: "git pull"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt",
          // yt-dlp is unpinned in requirements.txt, so a plain -r install won't
          // upgrade an already-installed copy. YouTube frequently breaks older
          // yt-dlp releases (HTTP 403 on download), so force the latest here.
          "uv pip install yt-dlp -U"
        ]
      }
    }
  ]
};