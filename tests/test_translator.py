import sys
import pytest
from unittest.mock import MagicMock


# liblouis is available only on Linux (apt: python3-louis).
# On Windows dev machines, we mock it to test our wrapper logic.
# On CI / the deployed app, use the real library.

def test_translate_grade1_a(monkeypatch):
    """'⠁' (dot 1) = Grade 1 'a'."""
    mock_louis = MagicMock()
    mock_louis.backTranslateString.return_value = "a"
    monkeypatch.setitem(sys.modules, 'louis', mock_louis)

    from importlib import reload
    import pipeline.translator as tr
    reload(tr)

    result = tr.translate('⠁', grade='grade1')
    assert result == 'a'
    mock_louis.backTranslateString.assert_called_once_with(
        ['braille-patterns.cti', 'en-ueb-g1.ctb'], '⠁'
    )


def test_translate_grade2_default(monkeypatch):
    mock_louis = MagicMock()
    mock_louis.backTranslateString.return_value = "the"
    monkeypatch.setitem(sys.modules, 'louis', mock_louis)

    from importlib import reload
    import pipeline.translator as tr
    reload(tr)

    result = tr.translate('⠮')  # grade='grade2' is default
    assert result == 'the'
    mock_louis.backTranslateString.assert_called_once_with(
        ['braille-patterns.cti', 'en-ueb-g2.ctb'], '⠮'
    )


def test_translate_empty_string_returns_empty(monkeypatch):
    mock_louis = MagicMock()
    monkeypatch.setitem(sys.modules, 'louis', mock_louis)

    from importlib import reload
    import pipeline.translator as tr
    reload(tr)

    result = tr.translate('   ')
    assert result == ''
    mock_louis.backTranslateString.assert_not_called()


def test_translate_nemeth_table(monkeypatch):
    mock_louis = MagicMock()
    mock_louis.backTranslateString.return_value = "3"
    monkeypatch.setitem(sys.modules, 'louis', mock_louis)

    from importlib import reload
    import pipeline.translator as tr
    reload(tr)

    tr.translate('⠃', grade='nemeth')
    mock_louis.backTranslateString.assert_called_once_with(
        ['braille-patterns.cti', 'nemeth.ctb'], '⠃'
    )


def test_translate_computer_table(monkeypatch):
    mock_louis = MagicMock()
    mock_louis.backTranslateString.return_value = "b"
    monkeypatch.setitem(sys.modules, 'louis', mock_louis)

    from importlib import reload
    import pipeline.translator as tr
    reload(tr)

    tr.translate('⠃', grade='computer')
    mock_louis.backTranslateString.assert_called_once_with(
        ['braille-patterns.cti', 'en-us-comp8-ext.utb'], '⠃'
    )


def test_translate_handles_exception_gracefully(monkeypatch):
    """When backTranslateString raises, individual chars are tried; failures → [?]."""
    def side_effect(tables, text):
        if len(text) > 1:
            raise RuntimeError("bulk fail")
        if text == '⠁':
            return 'a'
        raise RuntimeError("char fail")

    mock_louis = MagicMock()
    mock_louis.backTranslateString.side_effect = side_effect
    monkeypatch.setitem(sys.modules, 'louis', mock_louis)

    from importlib import reload
    import pipeline.translator as tr
    reload(tr)

    result = tr.translate('⠁⠃', grade='grade1')
    assert 'a' in result
    assert '[?]' in result
