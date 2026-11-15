"""Run in the MORGOTH Python environment. No weights or upstream code are bundled.

Uses upstream preprocessing and returns native window outputs plus a validity mask.
Deliberately bypasses the CLI's default-filling and interpolation of invalid rows.
"""
import json
import sys
from pathlib import Path


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    sys.path.insert(0, cfg["source_dir"])
    import numpy as np
    import torch
    import utils
    import task_model  # registers model architecture
    import backbone  # registers morgoth_backbone_base
    from timm.models import create_model
    from torch.utils.data import DataLoader

    head = cfg["head"]
    task = {"abnormal": "NORMAL", "seizure": "IIIC", "sleep": "SLEEPPSG"}[head]
    n_classes = {"abnormal": 1, "seizure": 6, "sleep": 5}[head]
    montage = utils.sleep_common_average_montage() if head == "sleep" else utils.common_average_montage()
    device = torch.device(cfg["device"])
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu or cuda; other devices have not been verified")
    try:
        ds = utils.ContinuousToSnippetDataset(
            type=task, path_signal=cfg["input"], given_fs=cfg["sample_rate"], has_formatted_channel=True,
            has_avg=False, allow_missing_channels=False, montage=montage,
            transform=utils.Compose([utils.Clipping(), utils.Scaling()]), step=1, step_in_point=False,
        )
    except utils.SnippetsError as exc:
        n = int(exc.result_segment_shapes)
        np.savez_compressed(cfg["output"], starts=np.arange(n, dtype=float),
                            scores=np.full((n, 2 if head == "abnormal" else n_classes), np.nan), valid=np.zeros(n, dtype=bool))
        return
    valid, spans, shapes = ds.get_valid_indices()
    valid = np.array(valid, dtype=bool)
    # Reject unexplained timing differences instead of stretching scores over time.
    if len(ds) != int(valid.sum()) or any(b - a + 1 != n for (a, b), n in zip(spans, shapes)):
        raise ValueError("Upstream window mapping requires interpolation; unsupported to avoid false timestamps")
    model = create_model("morgoth_backbone_base", pretrained=False, num_classes=n_classes,
                         drop_rate=0, drop_path_rate=.1, attn_drop_rate=0, drop_block_rate=None,
                         use_mean_pooling=True, init_scale=.001, use_rel_pos_bias=False,
                         use_abs_pos_emb=True, init_values=.1, qkv_bias=True)
    # Only load explicitly registered, trusted, checksum-verified local checkpoints.
    checkpoint = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    indices = utils.morgoth_get_input_chans(montage.get_channel_names())
    output = []
    with torch.inference_mode():
        for signals, _ in DataLoader(ds, batch_size=32, shuffle=False, num_workers=0):
            signals = (signals.float().to(device) / 100).reshape(signals.shape[0], signals.shape[1], 10, 200)
            logits = model(signals, input_chans=indices)
            probs = torch.sigmoid(logits) if n_classes == 1 else torch.softmax(logits, dim=-1)
            output.append(probs.float().cpu().numpy())
    probs = np.concatenate(output)
    if head == "abnormal":
        p = probs[:, 0]
        if cfg["positive_class"] == "normal":
            p = 1 - p
        elif cfg["positive_class"] != "abnormal":
            raise ValueError("Unverified binary class meaning")
        probs = np.column_stack([1 - p, p])
    scores = np.full((len(valid), probs.shape[1]), np.nan)
    scores[valid] = probs
    # Partially NaN inputs must never become confident outputs.
    valid &= np.isfinite(scores).all(axis=1)
    scores[~valid] = np.nan
    np.savez_compressed(cfg["output"], starts=np.arange(len(valid), dtype=float), scores=scores, valid=valid)


if __name__ == "__main__":
    main()
