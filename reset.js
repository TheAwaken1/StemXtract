module.exports = {
  run: [
    {
      method: "fs.rm",
      params: { path: "env" }
    },
    {
      method: "script.start",
      params: { uri: "install.js" }
    }
  ]
};