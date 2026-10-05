# Theta Plugins & Components

Standalone source repository for plugins and modular environments authored for [Theta-IDE](https://github.com/CameronEgb/Theta-IDE) and distributed through the [Theta Community Hub](https://github.com/CameronEgb/theta-hub).

> **Note on Architecture**: The `Theta-IDE` repository intentionally ships with **zero** bundled community plugins and only one core plugin (`documentation`). All community plugins are maintained independently in this repository and installed on-demand by users via the Community Hub.

---

## 📦 Components in This Repository

### 1. UI Plugins (`plugins/`)
* **`obsidian/` (Obsidian Notes)**: Embedded Obsidian vault browser, research markdown notebook, daily logs, and experiment run reference linking.
* **`doom-sim/` (DOOM Simulator Panel)**: Authentic 1993 id Software DOOM engine interactive web renderer running directly inside the IDE.
* **`antigravity/` (Antigravity Agent)**: Interactive AI pair programming and assistant panel integrated into Theta-IDE.

### 2. Environments (`envs/`)
* **`doom-env/` (DOOM RL Environment)**: Vectorized Gymnasium-compatible environment for tactical 2.5D Doom reinforcement learning with custom valuation and action discretization.

---

## 🛠️ Repository Structure

```
theta-plugins/
├── plugins/
│   ├── obsidian/          # Obsidian markdown vault browser
│   │   ├── plugin.json
│   │   ├── __init__.py
│   │   ├── panel.py
│   │   └── obsidian.svg
│   ├── doom-sim/          # Doom simulation panel
│   │   ├── plugin.json
│   │   ├── __init__.py
│   │   ├── panel.py
│   │   ├── doom.html
│   │   └── doom.svg
│   └── antigravity/       # Antigravity agent assistant
│       ├── plugin.json
│       ├── __init__.py
│       ├── panel.py
│       └── antigravity.svg
├── envs/
│   └── doom-env/          # Doom RL environment & wrappers
│       ├── env.py
│       ├── env_vectorized.py
│       ├── actions.py
│       ├── mlp.py
│       ├── valuation.py
│       └── doom.yaml
└── scripts/
    └── package_plugins.py # Packaging helper to build zips with SHA-256
```

---

## 🚀 Building & Packaging for Theta Hub

To compile release archives and verify SHA-256 hashes for `theta-hub`:

```bash
python scripts/package_plugins.py
```

This generates `.zip` archives ready to be published as GitHub Release assets or placed in `theta-hub/packages/`.
