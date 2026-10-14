"""Keep PyQt6 bound to the Qt DLLs bundled inside the one-file executable."""

import sys

if sys.platform.startswith("win") and hasattr(sys, "_MEIPASS"):
    from qt_runtime import configure_qt_dll_search_path

    configure_qt_dll_search_path()
