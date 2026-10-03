import pytest

from secops.data import schema as s


def test_raw_columns_count_and_boundaries() -> None:
    assert len(s.RAW_COLUMNS) == 91
    assert s.RAW_COLUMNS[0] == "id"
    assert s.RAW_COLUMNS[-2:] == ["Label", "Attempted Category"]


def test_feature_columns_exclude_identifiers_and_labels() -> None:
    assert len(s.FEATURE_COLS) == 82
    for c in s.ID_COLS + s.META_COLS + [s.LABEL_COL, s.ATTEMPTED_COL]:
        assert c not in s.FEATURE_COLS
    assert "Protocol" in s.FEATURE_COLS
    assert "Total TCP Flow Time" in s.FEATURE_COLS
    assert s.PORT_COL not in s.FEATURE_COLS


def test_base_label_strips_attempted_suffix() -> None:
    assert s.base_label("DoS Hulk - Attempted") == "DoS Hulk"
    assert s.base_label("DoS Hulk") == "DoS Hulk"
    assert s.is_attempted_label("Botnet - Attempted")
    assert not s.is_attempted_label("Botnet")


def test_family_of_every_known_label() -> None:
    assert s.family_of("BENIGN") == "benign"
    assert s.family_of("FTP-Patator") == "brute_force"
    assert s.family_of("Infiltration - Portscan") == "port_scan"
    assert s.family_of("Heartbleed") == "rare_exploit"
    assert s.family_of("Web Attack - XSS - Attempted") == "web_attack"


def test_family_of_unknown_label_raises() -> None:
    with pytest.raises(KeyError):
        s.family_of("Not A Label")


def test_family_classifier_classes_exclude_rare() -> None:
    assert "rare_exploit" not in s.FAMILY_CLASSIFIER_CLASSES
    assert set(s.FAMILY_CLASSIFIER_CLASSES) == {
        "brute_force",
        "dos",
        "ddos",
        "port_scan",
        "web_attack",
        "botnet",
    }
