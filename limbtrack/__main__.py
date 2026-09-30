import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    from PySide6.QtWidgets import QApplication, QMessageBox

    from .gui import MainWindow
    from .protocol import load_protocol

    app = QApplication(sys.argv)
    try:
        tests = load_protocol(ROOT / "protocol.json")
        win = MainWindow(tests, ROOT)
    except Exception:  # noqa: BLE001 - pythonw has no console, so show the problem
        msg = traceback.format_exc()
        (ROOT / "last_error.log").write_text(msg, encoding="utf-8")
        QMessageBox.critical(None, "Could not start", msg)
        return 1
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
