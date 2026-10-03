"""Column layout, label vocabulary and family grouping for DistriNet-improved CIC-IDS-2017.

Measured on the 2023-04-27 release of CICIDS2017_improved.zip (see data/README.md).
"""

from __future__ import annotations

DAY_ORDER: list[str] = ["monday", "tuesday", "wednesday", "thursday", "friday"]
DAY_FILES: dict[str, str] = {d: f"{d}.csv" for d in DAY_ORDER}

ID_COLS: list[str] = ["id", "Flow ID"]
META_COLS: list[str] = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"]
LABEL_COL = "Label"
ATTEMPTED_COL = "Attempted Category"
PORT_COL = "Dst Port"
BENIGN_LABEL = "BENIGN"
ATTEMPTED_SUFFIX = " - Attempted"

# fmt: off
RAW_COLUMNS: list[str] = [
    "id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp",
    "Flow Duration", "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet",
    "Total Length of Bwd Packet", "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd RST Flags",
    "Bwd RST Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Packet Length Min", "Packet Length Max", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWR Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Fwd Segment Size Avg", "Bwd Segment Size Avg",
    "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg", "Bwd Bytes/Bulk Avg",
    "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "FWD Init Win Bytes", "Bwd Init Win Bytes",
    "Fwd Act Data Pkts", "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max",
    "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "ICMP Code", "ICMP Type",
    "Total TCP Flow Time", "Label", "Attempted Category",
]
# fmt: on

_NON_FEATURE = set(ID_COLS) | set(META_COLS) | {LABEL_COL, ATTEMPTED_COL}
FEATURE_COLS: list[str] = [c for c in RAW_COLUMNS if c not in _NON_FEATURE]

FAMILY_MAP: dict[str, str] = {
    "FTP-Patator": "brute_force",
    "SSH-Patator": "brute_force",
    "DoS Hulk": "dos",
    "DoS GoldenEye": "dos",
    "DoS Slowloris": "dos",
    "DoS Slowhttptest": "dos",
    "DDoS": "ddos",
    "Portscan": "port_scan",
    "Infiltration - Portscan": "port_scan",
    "Web Attack - Brute Force": "web_attack",
    "Web Attack - XSS": "web_attack",
    "Web Attack - SQL Injection": "web_attack",
    "Botnet": "botnet",
    "Heartbleed": "rare_exploit",
    "Infiltration": "rare_exploit",
}
FAMILIES: list[str] = sorted(set(FAMILY_MAP.values()))
FAMILY_CLASSIFIER_CLASSES: list[str] = [f for f in FAMILIES if f != "rare_exploit"]


def is_attempted_label(label: str) -> bool:
    return label.endswith(ATTEMPTED_SUFFIX)


def base_label(label: str) -> str:
    return label[: -len(ATTEMPTED_SUFFIX)] if is_attempted_label(label) else label


def family_of(label: str) -> str:
    """Family for a raw label. Attempted labels map to their base family. Raises KeyError."""
    base = base_label(label)
    if base == BENIGN_LABEL:
        return "benign"
    return FAMILY_MAP[base]
