"""Antigravity Agent Plugin for ThetaIDE."""
from frontend.plugins.base import Plugin, PluginManifest
from frontend.plugins.context import PluginContext
try:
    from .panel import AntigravityPanel
except (ImportError, ValueError):
    from frontend.plugins.antigravity.panel import AntigravityPanel


class AntigravityPlugin(Plugin):
    """Integrates Google Antigravity Agent into ThetaIDE."""

    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self.context: PluginContext | None = None
        self.panel: AntigravityPanel | None = None

    def activate(self, context: PluginContext) -> None:
        self.context = context
        self.panel = AntigravityPanel(context=context)

        # Register tab into the sidebar
        context.add_sidebar_tab(
            tab_id="antigravity",
            widget=self.panel,
            title="Antigravity Agent",
            icon_name="antigravity",
            short_label="Agent",
        )
        context.log("Antigravity Agent plugin activated.")

    def deactivate(self) -> None:
        if self.context and self.panel:
            self.context.remove_sidebar_tab("antigravity")
            self.panel = None
            self.context.log("Antigravity Agent plugin deactivated.")
            self.context = None
