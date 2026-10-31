from i18n import language, set_language, tr


def test_russian_is_the_default_language() -> None:
    assert language() == "ru"


def test_english_translation_is_available_and_russian_can_be_restored() -> None:
    original = language()
    try:
        set_language("en")
        assert tr("Настройки процесса") == "Process settings"
        set_language("ru")
        assert tr("Настройки процесса") == "Настройки процесса"
    finally:
        set_language(original)


def test_switching_back_and_forth_retranslates_every_label() -> None:
    """Live switching RU → EN → RU → EN leaves no text in the other language."""
    import os
    import re

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QAbstractButton, QApplication, QLabel

    import SpinSlicer
    from i18n import _EN
    from ui_kit import NavButton

    app = QApplication.instance() or QApplication([])
    set_language("ru")
    window = SpinSlicer.CALSlicerMainWindow()
    window.show()
    cyrillic = re.compile("[А-Яа-яЁё]")
    english = {v for v in _EN.values() if re.search("[A-Za-z]{3}", v)}

    def shown() -> list[str]:
        out = []
        for widget in window.findChildren(QLabel) + window.findChildren(QAbstractButton):
            if widget.isVisible() and widget.text() and "<" not in widget.text():
                out.append(widget.text())
        for nav in window.findChildren(NavButton):          # paints its own text
            out += [t for t in (tr(nav._title), tr(nav._sub)) if t and t != "—"]
        return out

    def switch(lang: str) -> None:
        window.language_combo.setCurrentIndex(window.language_combo.findData(lang))
        app.processEvents()

    try:
        for page in ("setup", "model", "light", "print"):
            window._select_page(page)
            for lang in ("en", "ru", "en", "ru"):
                switch(lang)
                texts = shown()
                if lang == "en":
                    assert not [t for t in texts if cyrillic.search(t) and t != "Русский"], (page, lang)
                else:
                    assert not [t for t in texts if t in english and t != "English"], (page, lang)
    finally:
        window.close()
        set_language("ru")
