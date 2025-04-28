const path = require('path');
module.exports = {
  version: "1.0", // Or your application's version
  title: "🎵 StemXtract",
  description: "Your AI Audio Playground! ✨ Effortlessly split any track into stems with StemXtract. Create perfect instrumentals, isolate vocals, or craft unique mashups by blending tracks. Features smart tempo/beat matching, simple effects, and easy mixing controls.",
  icon: "icon.png", // Replace with your application's icon file
  autorun: {
    script: "start.js"
  },
  menu: async (kernel, info) => {
    let installed = info.exists("app/env"); // Assuming you're using a virtual environment named "env"
    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js"),
      link: info.running("link.js")
    };

    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js"
      }];
    } else if (installed) {
      if (running.start) {
        let local = info.local("start.js");
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Web UI",
            href: local.url
          }, {
            icon: "fa-solid fa-terminal",
            text: "Terminal",
            href: "start.js"
          }];
        } else {
          return [{
            default: true,
            icon: "fa-solid fa-terminal",
            text: "Terminal",
            href: "start.js"
          }];
        }
      } else if (running.update) {
        return [{
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Updating",
          href: "update.js"
        }];
      } else if (running.reset) {
        return [{
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Resetting",
          href: "reset.js"
        }];
      } else if (running.link) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Deduplicating",
          href: "link.js",
        }];
      } else {
        return [{
          icon: "fa-solid fa-power-off",
          text: "Start",
          href: "start.js"
        }, {
          icon: "fa-regular fa-folder-open",
          text: "View Outputs",
          href: "app/output", // Adjust if your output directory is different
          fs: true
        }, {
          icon: "fa-solid fa-plug",
          text: "Update",
          href: "update.js"
        }, {
          icon: "fa-solid fa-plug",
          text: "Reinstall",
          href: "install.js"
        }, {
          icon: "fa-regular fa-circle-xmark",
          text: "Reset",
          href: "reset.js"
        }];
      }
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js"
      }];
    }
  }
};