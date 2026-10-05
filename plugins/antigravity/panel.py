"""Antigravity Panel widget for chat interface using google.antigravity SDK."""
from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, 
    QTextBrowser, QVBoxLayout, QWidget,
)

if TYPE_CHECKING:
    from frontend.plugins.context import PluginContext

import subprocess
import json

class AgentWorker(QThread):
    response_chunk = pyqtSignal(str)
    response_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)
    conversation_id_update = pyqtSignal(str)

    def __init__(self, prompt: str, conversation_id: Optional[str] = None):
        super().__init__()
        self.prompt = prompt
        self.conversation_id = conversation_id

    def run(self):
        try:
            cmd = ["agy", "--output-format", "stream-json"]
            if self.conversation_id:
                cmd.extend(["--conversation", self.conversation_id])
            
            cmd.extend(["--print", self.prompt])
            
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Read stdout line by line
            for line in proc.stdout:
                line = line.strip()
                if not line or line.startswith("warning:"):
                    continue
                try:
                    data = json.loads(line)
                    event = data.get("event")
                    if event == "init":
                        cid = data.get("conversation_id")
                        if cid:
                            self.conversation_id_update.emit(cid)
                    elif event == "step_update":
                        su = data.get("step_update", {})
                        if su.get("step_type") == "agent_response" and "text_delta" in su:
                            self.response_chunk.emit(su["text_delta"])
                    elif event == "result":
                        break
                except json.JSONDecodeError:
                    pass
            
            proc.wait()
            if proc.returncode != 0:
                err = proc.stderr.read().strip()
                if err:
                    self.error_occurred.emit(f"CLI Error: {err}")
            
            self.response_complete.emit()
            
        except Exception as e:
            self.error_occurred.emit(str(e))

class AntigravityPanel(QWidget):
    """Integrated Antigravity Agent chat interface."""

    def __init__(self, context: PluginContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.context = context
        self._build_ui()
        self.worker = None
        self.conversation_id = None

    def _build_ui(self):
        """Construct the Antigravity chat panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Header
        header_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        eyebrow = QLabel("ANTIGRAVITY AGENT")
        eyebrow.setObjectName("eyebrow")
        eyebrow.setStyleSheet("font-size: 11px; font-weight: 700; letter-spacing: 1px; color: #83a598;")
        
        title_label = QLabel("Chat with AI")
        title_label.setStyleSheet("font-size: 14px; font-weight: 600;")
        title_box.addWidget(eyebrow)
        title_box.addWidget(title_label)
        header_layout.addLayout(title_box)
        header_layout.addStretch()
        
        clear_btn = QPushButton("Clear Chat")
        clear_btn.clicked.connect(self._clear_chat)
        header_layout.addWidget(clear_btn)
        
        layout.addLayout(header_layout)

        # Chat History
        self.chat_history = QTextBrowser()
        self.chat_history.setOpenExternalLinks(True)
        # Use simple standard styling matching Theta-IDE themes roughly
        self.chat_history.setStyleSheet("background-color: transparent; border: 1px solid #333; padding: 8px; border-radius: 6px;")
        self._append_system_message("Antigravity Agent initialized. How can I help you build today?")
        layout.addWidget(self.chat_history)

        # Input Area
        input_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Ask Antigravity to write code, explain errors, or run tasks...")
        self.message_input.returnPressed.connect(self._send_message)
        input_layout.addWidget(self.message_input)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._send_message)
        input_layout.addWidget(self.send_btn)

        layout.addLayout(input_layout)

    def _append_system_message(self, text: str):
        self.chat_history.append(f"<b>[System]</b> {text}")

    def _append_user_message(self, text: str):
        self.chat_history.append(f"<br/><b style='color: #4daafc;'>[User]</b> {text}")

    def _append_agent_message_start(self):
        self.chat_history.append(f"<br/><b style='color: #4caf50;'>[Antigravity]</b> ")
        # Ensure it moves to the end so we can append text cleanly
        self.chat_history.moveCursor(self.chat_history.textCursor().MoveOperation.End)

    def _append_agent_chunk(self, text: str):
        # insert plaintext at the end
        self.chat_history.insertPlainText(text)
        self.chat_history.moveCursor(self.chat_history.textCursor().MoveOperation.End)

    def _send_message(self):
        text = self.message_input.text().strip()
        if not text:
            return
            
        self.message_input.clear()
        self.message_input.setEnabled(False)
        self.send_btn.setEnabled(False)
        
        self._append_user_message(text)
        self._append_agent_message_start()
        
        self.worker = AgentWorker(text, self.conversation_id)
        self.worker.response_chunk.connect(self._append_agent_chunk)
        self.worker.response_complete.connect(self._on_agent_complete)
        self.worker.error_occurred.connect(self._on_agent_error)
        self.worker.conversation_id_update.connect(self._on_conversation_id_update)
        self.worker.start()

    def _on_conversation_id_update(self, cid: str):
        self.conversation_id = cid

    def _on_agent_complete(self):
        self.message_input.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.message_input.setFocus()

    def _on_agent_error(self, err: str):
        self._append_system_message(f"Error: {err}")
        self.message_input.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.message_input.setFocus()

    def _clear_chat(self):
        self.chat_history.clear()
        self.conversation_id = None
        self._append_system_message("Chat history cleared. Started a new conversation.")
