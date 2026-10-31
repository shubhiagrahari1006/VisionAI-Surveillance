"""
VisionAI - Automated Launcher with Windows Asyncio EventLoop Protection
Protects against ConnectionResetError [WinError 10054] when browser refreshes or disconnects.
"""

import sys

# Windows asyncio ProactorEventLoop WinError 10054 ConnectionResetError patch
if sys.platform == "win32":
    try:
        import asyncio.proactor_events

        _orig_call_connection_lost = (
            asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost
        )

        def _silent_call_connection_lost(self, exc):
            try:
                _orig_call_connection_lost(self, exc)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
                try:
                    if self._sock is not None:
                        self._sock.close()
                        self._sock = None
                except Exception:
                    pass
                self._called_connection_lost = True

        asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost = (
            _silent_call_connection_lost
        )
    except Exception:
        pass

from streamlit.web.cli import main

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app/main.py"]
    main()
