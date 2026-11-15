"""Compare native-window probability matrices, excluding invalid windows.

Use upstream native probabilities BEFORE its resizing/default-filling CSV path.
Both NPZ files must contain starts, scores, valid with NeuroScope class order.
"""
import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference"); p.add_argument("candidate")
    p.add_argument("--atol", type=float, default=1e-5)
    args = p.parse_args()
    with np.load(args.reference, allow_pickle=False) as a, np.load(args.candidate, allow_pickle=False) as b:
        np.testing.assert_allclose(a["starts"], b["starts"], atol=1e-9, rtol=0)
        np.testing.assert_array_equal(a["valid"], b["valid"])
        valid = a["valid"].astype(bool)
        if not valid.any():
            raise ValueError("No valid windows to compare")
        np.testing.assert_allclose(a["scores"][valid], b["scores"][valid], atol=args.atol, rtol=0)
        error = np.max(np.abs(a["scores"][valid] - b["scores"][valid]))
        print(f"Parity passed for {valid.sum()} windows. Maximum absolute score error: {error:.8g}")


if __name__ == "__main__":
    main()
