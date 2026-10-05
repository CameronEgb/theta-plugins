"""DOOM Plugin for ThetaIDE: Real 1993 id Software DOOM."""
from __future__ import annotations
from pathlib import Path
import shutil
from typing import Optional

from frontend.plugins.base import Plugin, PluginManifest
from frontend.plugins.context import PluginContext
from frontend.plugins.doom.panel import DoomPanel


class DoomPlugin(Plugin):
    """Launches the authentic 1993 id Software DOOM engine inside Theta IDE and provides the DOOM RL environment."""

    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self.context: Optional[PluginContext] = None
        self.panel: Optional[DoomPanel] = None

    def activate(self, context: PluginContext) -> None:
        self.context = context
        self.panel = DoomPanel(context=context)

        # Synchronize packaged RL simulator environment into workspace if needed
        self._sync_env(context)

        # Register tab into the primary IDE sidebar
        context.add_sidebar_tab(
            tab_id="doom",
            widget=self.panel,
            title="DOOM",
            icon_name="doom",
            short_label="DOOM",
        )
        context.log("DOOM plugin activated: Can it run DOOM? YES!")

    def _sync_env(self, context: PluginContext) -> None:
        """Ensure the DOOM RL simulator is available among workspace environments."""
        try:
            workspace = context.get_workspace_dir()
            target_env = workspace / "in" / "envs" / "doom"
            target_config_dir = workspace / "in" / "config" / "env"
            target_config_file = target_config_dir / "doom.yaml"

            plugin_dir = Path(__file__).resolve().parent
            bundled_env = plugin_dir / "env"
            bundled_config = plugin_dir / "doom.yaml"

            if bundled_env.exists() and not target_env.exists():
                shutil.copytree(bundled_env, target_env)
                context.log("Installed DOOM RL simulator to in/envs/doom")

            if bundled_config.exists() and not target_config_file.exists():
                target_config_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(bundled_config, target_config_file)
                context.log("Installed DOOM environment config to in/config/env/doom.yaml")
        except Exception as exc:
            context.log(f"Notice: could not sync DOOM RL simulator: {exc}")

    def deactivate(self) -> None:
        if self.context and self.panel:
            self.context.remove_sidebar_tab("doom")
            self.panel.cleanup()
            self.panel = None
            self.context.log("DOOM plugin deactivated.")
            self.context = None
