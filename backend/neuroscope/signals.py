"""Recording validation and bounded waveform access; input references are never guessed."""
import hashlib
import os
import re
import time
from pathlib import Path
import numpy as np
import mne
from .config import EEG_CHANNELS, SLEEP_CHANNELS

ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def channel_map(names):
    mapped, references, problems = {}, {}, []
    for original in names:
        clean = re.sub(r"^EEG\s*", "", original.strip(), flags=re.I).upper().strip()
        parts = clean.split("-")
        electrode = ALIASES.get(parts[0], parts[0])
        if electrode not in EEG_CHANNELS:
            continue
        reference = parts[1] if len(parts) == 2 else "UNSPECIFIED" if len(parts) == 1 else "INVALID"
        # Do not reinterpret bipolar channels as monopolar electrodes.
        if reference in EEG_CHANNELS or reference in ALIASES or reference == "INVALID":
            problems.append(f"{original}: bipolar/ambiguous reference")
            continue
        if reference not in {"REF", "LE", "AVG", "AR", "A1", "A2", "M1", "M2", "UNSPECIFIED"}:
            problems.append(f"{original}: unsupported reference")
            continue
        if electrode in mapped:
            problems.append(f"Duplicate electrode {electrode}")
            mapped[electrode] = None
        else:
            mapped[electrode] = original
            references[electrode] = reference
    return {k: v for k, v in mapped.items() if v}, references, problems


def capabilities(names):
    mapped, refs, problems = channel_map(names)
    result = {}
    for head, required in [("abnormal", EEG_CHANNELS), ("seizure", EEG_CHANNELS), ("sleep", SLEEP_CHANNELS)]:
        missing = [c for c in required if c not in mapped]
        selected_refs = {refs.get(c) for c in required if c in mapped}
        supported = not missing and len(selected_refs) == 1
        # Six PSG derivations are supported by the upstream sleep preprocessing.
        mastoid = {"F3": "M2", "C3": "M2", "O1": "M2", "F4": "M1", "C4": "M1", "O2": "M1"}
        if head == "sleep" and not missing and all(refs[c].replace("A", "M") == mastoid[c] for c in required):
            supported = True
        reason = "Compatible channel set" if supported else (
            "Missing compatible electrodes: " + ", ".join(missing) if missing else "Mixed or unsupported reference montage")
        result[head] = {"supported": supported, "reason": reason, "channels": [mapped[c] for c in required if c in mapped]}
    return result, mapped, problems


def inspect_edf(path: Path, recording_id: str, display_name: str):
    with path.open("rb") as f:
        header = f.read(256)
    if len(header) < 256:
        raise ValueError("File is too short to be an EDF recording")
    if b"EDF+D" in header[192:236]:
        raise ValueError("Discontinuous EDF+D is not supported yet. Export continuous segments separately to preserve timestamps.")
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    try:
        duration = float(raw.n_times / raw.info["sfreq"])
        if duration < 10:
            raise ValueError("At least 10 seconds of EEG is required")
        if duration > 7 * 24 * 3600 or raw.info["sfreq"] > 10000 or len(raw.ch_names) > 256:
            raise ValueError("Recording exceeds supported duration, sampling rate, or channel count")
        if any("boundary" in a.lower() for a in raw.annotations.description):
            raise ValueError("Recording contains discontinuities; import continuous segments separately")
        caps, mapping, problems = capabilities(raw.ch_names)
        n = int(header[252:256])
        with path.open('rb') as header_file:
            header_file.seek(256)
            signal_header = header_file.read(256 * n)
        record_seconds = float(header[244:252])
        # EDF+ annotation channels remain in the file header but MNE excludes them.
        signal_indices = [i for i in range(n) if signal_header[16*i:16*(i+1)].strip() not in
                          (b'EDF Annotations', b'BDF Annotations')]
        if len(signal_indices) != len(raw.ch_names):
            raise ValueError('EDF channel header does not match the decoded signals')
        rates = {name: int(signal_header[216*n+8*i:216*n+8*(i+1)]) / record_seconds
                 for name, i in zip(raw.ch_names, signal_indices)}
        # Low-rate input cannot supply the spectrum expected by the model.
        if raw.info["sfreq"] < 128 or not float(raw.info["sfreq"]).is_integer():
            for cap in caps.values():
                cap.update(supported=False, reason="MORGOTH requires an integer sampling rate of at least 128 Hz in this adapter")
        if os.getenv('NEUROSCOPE_ENGINE', 'open') == 'open':
            from .open_channels import open_capabilities
            caps = open_capabilities(raw.ch_names, rates, duration)
            problems = []
            if caps['sleep']['supported']:
                problems.append('YASA is trained on full-night PSG. Check EEG reference and recording suitability before interpreting sleep stages.')
            if any(re.search(r'-\d+$', name) for name in raw.ch_names):
                problems.append('Duplicate channel labels detected. Analysis verifies that duplicate derivations agree.')
        return {"id": recording_id, "name": display_name, "kind": "edf", "duration_s": duration,
                "sample_rate": float(raw.info["sfreq"]), "n_samples": int(raw.n_times), "channels": raw.ch_names,
                "mapping": mapping, "capabilities": caps, "warnings": problems, "channel_rates": rates,
                "created": time.time(), "sha256": sha256_file(path), "time_basis": "recording-relative",
                "source": "Uploaded recording", "path": str(path)}
    finally:
        raw.close()


def read_signal(recording, start_s, end_s, channels=None):
    if recording["kind"] == "synthetic":
        return demo_waveform(start_s, end_s, channels or EEG_CHANNELS, recording["sample_rate"])
    raw = mne.io.read_raw_edf(recording["path"], preload=False, verbose="ERROR")
    try:
        fs = raw.info["sfreq"]
        start = max(0, int(round(start_s * fs)))
        stop = min(raw.n_times, int(round(end_s * fs)))
        picks = channels or raw.ch_names
        data = raw.get_data(picks=picks, start=start, stop=stop, units="uV")
        return np.arange(start, stop) / fs, data, picks
    finally:
        raw.close()


def waveform(recording, start_s, end_s, channels, points=1400, montage="recorded"):
    times, values, names = read_signal(recording, start_s, end_s, channels)
    if montage == "average":
        # Display re-reference only; scores are never changed.
        values = values - np.nanmean(values, axis=0, keepdims=True)
    if not len(times):
        return {"times": [], "channels": [], "units": "µV", "montage": montage}
    stride = max(1, int(np.ceil(len(times) / points)))
    starts = np.arange(0, len(times), stride)
    lows, highs = [], []
    for row in values:
        lows.append(np.minimum.reduceat(row, starts))
        highs.append(np.maximum.reduceat(row, starts))
    def clean(array):
        return [float(x) if np.isfinite(x) else None for x in array]
    return {"times": times[starts].tolist(), "channels": [
        {"name": n, "min": clean(lo), "max": clean(hi)} for n, lo, hi in zip(names, lows, highs)],
        "units": "µV", "montage": montage, "start_s": float(times[0]), "end_s": end_s,
        "sample_rate": recording["sample_rate"], "decimated": stride > 1}


DEMO_EVENTS = [(194, 219, "abnormal"), (728, 772, "seizure"), (1408, 1442, "LPD"),
               (2024, 2059, "abnormal"), (2903, 2951, "seizure")]
DEMO_GAP = (1760, 1780)


def demo_recording():
    caps, mapping, _ = capabilities(EEG_CHANNELS)
    return {"id": "synthetic-demo-v1", "name": "Overnight study · demonstration", "kind": "synthetic",
            "duration_s": 3600, "sample_rate": 200, "n_samples": 720000, "channels": EEG_CHANNELS,
            "mapping": mapping, "capabilities": caps, "warnings": ["Synthetic signals and scripted predictions. Not patient data or MORGOTH output."],
            "created": time.time(), "sha256": "synthetic-demo-v1", "time_basis": "recording-relative",
            "source": "Synthetic demonstration"}


def demo_waveform(start_s, end_s, channels, fs=200):
    t = np.arange(round(start_s * fs), round(end_s * fs)) / fs
    traces = []
    for name in channels:
        c = EEG_CHANNELS.index(name) + 1
        signal = 9 * np.sin(2 * np.pi * (8 + c * .13) * t + c) + 5 * np.sin(2 * np.pi * 2.1 * t + c * .7)
        signal += 3 * np.sin(2 * np.pi * 18.71 * t + c * 2) + 1.3 * np.sin(2 * np.pi * 37.3 * t)
        for start, end, kind in DEMO_EVENTS:
            envelope = np.clip((t - start) / 3, 0, 1) * np.clip((end - t) / 3, 0, 1)
            signal += envelope * (48 if kind == "seizure" else 25) * np.sin(2 * np.pi * (3 + c * .02) * t + c * .2)
        signal[(t >= DEMO_GAP[0]) & (t < DEMO_GAP[1])] = np.nan
        traces.append(signal)
    return t, np.array(traces), channels
