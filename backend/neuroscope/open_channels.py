"""Explicit input contracts for SPaRCNet and YASA; no missing-channel filling."""
import re

ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6", "A1": "M1", "A2": "M2"}
PAIRS = [("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
         ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
         ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
         ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2")]


def normalized(name):
    name = re.sub(r"^EEG\s+", "", name.strip(), flags=re.I).upper().replace(" ", "")
    return "-".join(ALIASES.get(p, p) for p in name.split("-"))


def seizure_derivations(names):
    from .signals import channel_map
    mapped, refs, _ = channel_map(names)
    result = []
    for a, b in PAIRS:
        direct = [n for n in names if normalized(n) == f"{a}-{b}" or
                  re.fullmatch(re.escape(f"{a}-{b}") + r"-\d+", normalized(n))]
        reverse = [n for n in names if normalized(n) == f"{b}-{a}"]
        if direct:
            result.append({"positive": direct[0], "negative": None, "sign": 1,
                           "duplicates": direct[1:]})
        elif reverse:
            result.append({"positive": reverse[0], "negative": None, "sign": -1, "duplicates": reverse[1:]})
        elif a in mapped and b in mapped and refs[a] == refs[b]:
            result.append({"positive": mapped[a], "negative": mapped[b], "sign": 1, "duplicates": []})
        else:
            raise ValueError(f"SPaRCNet needs {a}-{b}, or both electrodes with the same reference")
    return result


def sleep_channels(names, rates=None):
    rates = rates or {n: 100 for n in names}
    # Explicitly support the open Sleep-EDF montage, retaining its caveat in provenance.
    preferred = ["C4-M1", "C3-M2", "C3-M1", "C4-M2", "C4-FPZ", "C3-FPZ",
                 "C3", "C4", "C3-REF", "C4-REF", "FPZ-CZ"]
    eeg = next((n for target in preferred for n in names if normalized(n) == target and rates.get(n, 0) >= 80), None)
    if eeg is None:
        raise ValueError("YASA needs a central EEG derivation (e.g. C3-M2) or Fpz-Cz, sampled at least 80 Hz")
    eog = next((n for n in names if ("EOG" in n.upper() or normalized(n).startswith(("E1-", "LOC-"))) and rates.get(n, 0) >= 80), None)
    emg = next((n for n in names if "EMG" in n.upper() and rates.get(n, 0) >= 80), None)
    return {"eeg": eeg, "eog": eog, "emg": emg}


def open_capabilities(names, rates, duration):
    result = {"abnormal": {"supported": False, "reason": "Broad normal/abnormal classification is unavailable. Other is not normal.", "channels": []}}
    try:
        pairs = seizure_derivations(names)
        chosen = list(dict.fromkeys(n for p in pairs for n in [p["positive"], p["negative"], *p["duplicates"]] if n))
        if any(rates.get(n, 0) < 100 for n in chosen):
            raise ValueError("SPaRCNet needs EEG sampled at least 100 Hz")
        if duration < 10:
            raise ValueError("SPaRCNet needs at least 10 seconds")
        result["seizure"] = {"supported": True, "reason": "SPaRCNet · 16 bipolar derivations · 10-second windows", "channels": chosen}
    except ValueError as exc:
        result["seizure"] = {"supported": False, "reason": str(exc), "channels": []}
    try:
        channels = sleep_channels(names, rates)
        if duration < 300:
            raise ValueError("Sleep staging needs at least 5 minutes; overnight recordings are recommended")
        result["sleep"] = {"supported": True, "reason": f"YASA · EEG {channels['eeg']} · 30-second epochs; verify reference suitability", "channels": [v for v in channels.values() if v]}
    except ValueError as exc:
        result["sleep"] = {"supported": False, "reason": str(exc), "channels": []}
    return result
