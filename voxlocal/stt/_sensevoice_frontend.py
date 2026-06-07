from __future__ import annotations

import kaldi_native_fbank as knf
import numpy as np


def extract_features(waveform: np.ndarray, cmvn_file: str) -> np.ndarray:
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.frame_length_ms = 25.0
    opts.mel_opts.num_bins = 80
    opts.energy_floor = 0
    opts.frame_opts.snip_edges = True

    cmvn = _load_cmvn(cmvn_file)
    waveform = waveform * (1 << 15)
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(16000, waveform.tolist())
    n = fb.num_frames_ready
    mat = np.empty([n, 80], dtype=np.float32)
    for i in range(n):
        mat[i, :] = fb.get_frame(i)

    mat = _apply_lfr(mat, 7, 6)
    T, dim = mat.shape
    m = np.tile(cmvn[0:1, :dim], (T, 1))
    v = np.tile(cmvn[1:2, :dim], (T, 1))
    return (mat + m) * v


def _load_cmvn(cmvn_file: str) -> np.ndarray:
    with open(cmvn_file) as f:
        lines = f.readlines()
    ml, vl = [], []
    for i, line in enumerate(lines):
        p = line.split()
        if p[0] == "<AddShift>" and i + 1 < len(lines):
            parts = lines[i + 1].split()
            if parts[0] == "<LearnRateCoef>":
                ml = list(parts[3:-1])
        elif p[0] == "<Rescale>" and i + 1 < len(lines):
            parts = lines[i + 1].split()
            if parts[0] == "<LearnRateCoef>":
                vl = list(parts[3:-1])
    return np.array([np.array(ml, dtype=np.float64), np.array(vl, dtype=np.float64)])


def _apply_lfr(inputs: np.ndarray, m: int, n: int) -> np.ndarray:
    T = inputs.shape[0]
    Tl = int(np.ceil(T / n))
    inp = np.vstack((np.tile(inputs[0], ((m - 1) // 2, 1)), inputs))
    T = T + (m - 1) // 2
    out = []
    for i in range(Tl):
        if m <= T - i * n:
            out.append(inp[i * n : i * n + m].reshape(1, -1))
        else:
            pad = m - (T - i * n)
            frame = inp[i * n :].reshape(-1)
            for _ in range(pad):
                frame = np.hstack((frame, inp[-1]))
            out.append(frame)
    return np.vstack(out).astype(np.float32)
