"""Non-blocking Windows-compatible text-to-speech for live detection."""

from __future__ import annotations

import queue
import threading


class Speaker:
    """Queues every speech request so camera inference is never blocked by playback."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        # Do not cap this queue.  A bounded queue silently drops later snapshot
        # announcements while the previous one is still being spoken.
        self._messages: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._closed = False
        if enabled:
            try:
                import pyttsx3  # noqa: F401
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
            except ImportError:
                self.enabled = False
                print("Speech disabled: install pyttsx3 with `pip install -r requirements.txt`.")

    def say(self, message: str) -> None:
        if not self.enabled or self._closed:
            return
        self._messages.put_nowait(message)

    def _run(self) -> None:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        while True:
            message = self._messages.get()
            if message is None:
                return
            engine.say(message)
            engine.runAndWait()

    def close(self) -> None:
        if self.enabled and not self._closed:
            self._closed = True
            self._messages.put_nowait(None)
