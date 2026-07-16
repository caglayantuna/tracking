"""
larva_head_tail.py
==================
Head/tail assignment for larva spine sequences.

Pipeline
--------
1. parse_file / build_contour_lookup  — I/O helpers
2. stabilize_temporal                 — Fix frame-to-frame point-order flips
3. score_curvature_all                — Geometry cue: head is the pointed end
4. score_motion_multiscale            — Motion cue: head leads movement
5. score_trajectory_global            — Whole-track travel direction
6. combine_and_anchor                 — Merge signals + global polarity anchor
7. viterbi_head_sequence              — HMM decoding + short-run cleanup
8. process_spine_file                 — Top-level entry point
"""

import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_file(filepath: str) -> dict:
    """
    Parse a LarvaHub .outline or .spine file.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    dict
        {label_int: [(date_str, time_float, pts_ndarray), ...]} sorted by time.
    """
    data = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            label = int(parts[1])
            t     = float(parts[2])
            pts   = np.array(parts[3:], dtype=float).reshape(-1, 2)
            data[label].append((parts[0], t, pts))
    for k in data:
        data[k].sort(key=lambda e: e[1])
    return dict(data)


def build_contour_lookup(outline_data: dict) -> dict:
    """
    Build {label: {round(t, 3): contour_pts}} for O(1) frame lookup.

    Parameters
    ----------
    outline_data : dict
        Output of parse_file for a .outline file.

    Returns
    -------
    dict
        {label_int: {time_rounded: contour_ndarray}}
    """
    return {
        lbl: {round(t, 3): pts for _, t, pts in entries}
        for lbl, entries in outline_data.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — temporal stabilisation
# ─────────────────────────────────────────────────────────────────────────────

def _arc_lengths(entries: list) -> np.ndarray:
    """
    Total spine arc length per frame.

    Parameters
    ----------
    entries : list of (str, float, np.ndarray)

    Returns
    -------
    np.ndarray, shape (N,)
    """
    return np.array([
        np.sum(np.linalg.norm(np.diff(e[2], axis=0), axis=1))
        for e in entries
    ])


def stabilize_temporal(entries: list) -> list:
    """
    Fix frame-to-frame point-order flips in the raw spine sequence.

    Compares each frame's endpoints to the last non-degenerate reference frame
    and reverses the point order if that reduces the total endpoint displacement.
    Skips frames where the larva is tightly curled or boundary-clipped.

    Parameters
    ----------
    entries : list of (str, float, np.ndarray)

    Returns
    -------
    list of (str, float, np.ndarray)
    """
    if not entries:
        return entries

    arcs     = _arc_lengths(entries)
    median_a = np.median(arcs)
    min_arc  = median_a * 0.4 if median_a > 1e-3 else 0.0

    out            = [entries[0]]
    last_good_pts  = entries[0][2]   # spine of last non-degenerate frame
    k = 3

    for idx in range(1, len(entries)):
        date_str, t, pts = entries[idx]

        arc = arcs[idx]
        ttt = float(np.linalg.norm(pts[-1] - pts[0]))  # tip-to-tip

        # Skip boundary-clipped and tight-curl frames — endpoint comparison is unreliable
        is_clipped = arc < min_arc
        is_curled  = (arc > 1e-3) and (ttt / arc < 0.25)
        if is_clipped or is_curled:
            out.append((date_str, t, pts))
            continue

        prev = last_good_pts
        m    = min(k, len(pts))
        p0   = pts[:m].mean(axis=0)
        p1   = pts[-m:].mean(axis=0)
        q0   = prev[:m].mean(axis=0)
        q1   = prev[-m:].mean(axis=0)
        cost_same = np.linalg.norm(p0 - q0) + np.linalg.norm(p1 - q1)
        cost_flip = np.linalg.norm(p1 - q0) + np.linalg.norm(p0 - q1)
        if cost_flip < cost_same:
            pts = pts[::-1]

        out.append((date_str, t, pts))
        last_good_pts = pts          # update reference only for good frames

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — evidence signals
# ─────────────────────────────────────────────────────────────────────────────

def _contour_curvature(contour: np.ndarray, point: np.ndarray,
                       radius: int = 5) -> float:
    """
    Exterior angle (radians) of the contour at the vertex nearest to point.
    Straight → 0; sharp corner → π.

    Parameters
    ----------
    contour : np.ndarray, shape (M, 2)
    point   : np.ndarray, shape (2,)
    radius  : int

    Returns
    -------
    float in [0, π]
    """
    dists = np.linalg.norm(contour - point, axis=1)
    idx   = int(np.argmin(dists))
    n     = len(contour)
    v1    = contour[(idx - radius) % n] - contour[idx]
    v2    = contour[(idx + radius) % n] - contour[idx]
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.pi - np.arccos(np.clip(cos_a, -1.0, 1.0))


def _local_body_width(contour: np.ndarray, point: np.ndarray,
                      arc_frac: float = 0.08) -> float:
    """
    Approximate local body width at the contour point nearest to point.
    Measured as the chord between two vertices walked arc_frac * len(contour)
    steps in each direction.

    Parameters
    ----------
    contour  : np.ndarray, shape (M, 2)
    point    : np.ndarray, shape (2,)
    arc_frac : float

    Returns
    -------
    float
    """
    dists = np.linalg.norm(contour - point, axis=1)
    idx   = int(np.argmin(dists))
    n     = len(contour)
    step  = max(2, int(n * arc_frac))
    p1    = contour[(idx + step) % n]
    p2    = contour[(idx - step) % n]
    return float(np.linalg.norm(p1 - p2))


def score_body_width_all(entries: list,
                         contour_lut: dict,
                         label: int) -> np.ndarray:
    """
    Per-frame score based on body width at each spine tip.
    Head is narrower: positive score → pts[-1] is head, negative → pts[0] is head.
    Returns zeros for frames where the outline is unavailable.

    Parameters
    ----------
    entries     : list of (str, float, np.ndarray)
    contour_lut : dict
    label       : int

    Returns
    -------
    np.ndarray, shape (N,)
    """
    n      = len(entries)
    scores = np.zeros(n)
    lut    = contour_lut.get(label, {})

    for i, (_, t, pts) in enumerate(entries):
        ctr = lut.get(round(t, 3))
        if ctr is None or len(ctr) < 15:
            continue
        w0 = _local_body_width(ctr, pts[0])
        w1 = _local_body_width(ctr, pts[-1])
        scores[i] = w0 - w1          # positive when pts[-1] end is narrower

    return scores


def score_curvature_all(entries: list,
                        contour_lut: dict,
                        label: int) -> np.ndarray:
    """
    Per-frame score based on outline curvature at each spine tip.
    Head is more pointed (higher curvature): positive score → pts[-1] is head.
    Returns zeros for frames where the outline is unavailable.

    Parameters
    ----------
    entries     : list of (str, float, np.ndarray)
    contour_lut : dict
    label       : int

    Returns
    -------
    np.ndarray, shape (N,)
    """
    n      = len(entries)
    scores = np.zeros(n)
    lut    = contour_lut.get(label, {})

    for i, (_, t, pts) in enumerate(entries):
        ctr = lut.get(round(t, 3))
        if ctr is None or len(ctr) < 11:
            continue
        radius = max(3, len(ctr) // 20)   # ~5 % of contour, at least 3
        c0 = _contour_curvature(ctr, pts[0],  radius)
        c1 = _contour_curvature(ctr, pts[-1], radius)
        scores[i] = c1 - c0          # positive when pts[-1] end is sharper

    return scores


def score_motion_multiscale(entries: list,
                             scales: tuple = (1, 3, 8, 30)) -> np.ndarray:
    """
    Per-frame score based on which spine tip leads the body's movement.
    Computed at multiple temporal scales (1, 3, 8, 30 frames) and summed.
    Positive score → pts[-1] is head.

    Parameters
    ----------
    entries : list of (str, float, np.ndarray)
    scales  : tuple of int

    Returns
    -------
    np.ndarray, shape (N,)
    """
    n      = len(entries)
    scores = np.zeros(n)

    for k in scales:
        weight = 1.0 / k          # downweight longer-range estimates slightly
        for i in range(n):
            i_lo = max(0,     i - k)
            i_hi = min(n - 1, i + k)
            if i_lo == i_hi:
                continue

            # Body displacement vector over this window
            c_lo = entries[i_lo][2].mean(axis=0)
            c_hi = entries[i_hi][2].mean(axis=0)
            d    = c_hi - c_lo
            dist = np.linalg.norm(d)
            if dist < 0.5:
                continue                # essentially stationary at this scale

            d_hat = d / dist
            pts   = entries[i][2]
            c     = pts.mean(axis=0)

            # Project each endpoint onto travel direction
            p0 = np.dot(pts[0]  - c, d_hat)
            p1 = np.dot(pts[-1] - c, d_hat)
            scores[i] += weight * (p1 - p0)

    return scores


def score_trajectory_global(entries: list) -> np.ndarray:
    """
    Per-frame score based on the overall travel direction of the whole track.
    Suppressed (returns zeros) when the path is incoherent (e.g. U-turn).
    Positive score → pts[-1] is head.

    Parameters
    ----------
    entries : list of (str, float, np.ndarray)

    Returns
    -------
    np.ndarray, shape (N,)
    """
    n       = len(entries)
    centers = np.array([e[2].mean(axis=0) for e in entries])

    # Sample ~15 local displacement vectors evenly across the track
    step = max(1, n // 15)
    local_dirs = []
    for i in range(0, n - step, step):
        d    = centers[i + step] - centers[i]
        dist = np.linalg.norm(d)
        if dist > 2.0:
            local_dirs.append(d / dist)

    if not local_dirs:
        return np.zeros(n)

    avg_dir   = np.mean(local_dirs, axis=0)
    coherence = np.linalg.norm(avg_dir)   # 1 = straight, ~0 = U-turn

    if coherence < 0.50:
        return np.zeros(n)

    global_hat = avg_dir / coherence
    path_len = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
    strength = coherence * path_len

    scores = np.zeros(n)
    for i, (_, _, pts) in enumerate(entries):
        axis   = pts[-1] - pts[0]
        ax_len = np.linalg.norm(axis)
        if ax_len < 1e-6:
            continue
        scores[i] = np.dot(axis / ax_len, global_hat) * strength

    return scores


def _spine_valid_mask(entries: list, min_arc_frac: float = 0.40) -> np.ndarray:
    """
    Boolean mask: True for frames with a non-degenerate spine.
    Frames whose arc length is below min_arc_frac * median are marked invalid
    (boundary-clipped spine).

    Parameters
    ----------
    entries      : list of (str, float, np.ndarray)
    min_arc_frac : float

    Returns
    -------
    np.ndarray of bool, shape (N,)
    """
    arcs       = _arc_lengths(entries)
    median_arc = np.median(arcs)
    if median_arc < 1e-3:
        return np.ones(len(entries), dtype=bool)
    return arcs >= min_arc_frac * median_arc


def _smooth(scores: np.ndarray, window: int) -> np.ndarray:
    """
    Symmetric variable-width box filter (no zero-padding at edges).

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
    window : int  half-width

    Returns
    -------
    np.ndarray, shape (N,)
    """
    out = np.empty_like(scores, dtype=float)
    n   = len(scores)
    for i in range(n):
        lo      = max(0, i - window)
        hi      = min(n, i + window + 1)
        out[i]  = scores[lo:hi].mean()
    return out


def combine_and_anchor(entries: list,
                       contour_lut: dict,
                       label: int,
                       w_curv: float = 3.0,
                       w_motion: float = 6.0,
                       w_global: float = 4.0,
                       w_width: float = 2.0,
                       smooth_window: int = 9) -> np.ndarray:
    """
    Combine all evidence signals into a single score per frame.

    Each signal is normalised to unit std, weighted, and summed.
    A global polarity anchor (mean score across the whole track) is added as a
    constant offset to stabilise the HMM during low-evidence segments.
    Output is smoothed and compressed to [-1, 1] via tanh.
    Positive → pts[-1] is head. Negative → pts[0] is head.

    Parameters
    ----------
    entries       : list of (str, float, np.ndarray)
    contour_lut   : dict
    label         : int
    w_curv        : float
    w_motion      : float
    w_global      : float
    w_width       : float
    smooth_window : int

    Returns
    -------
    np.ndarray, shape (N,)
    """
    s_curv   = score_curvature_all(entries, contour_lut, label)
    s_motion = score_motion_multiscale(entries)
    s_global = score_trajectory_global(entries)
    s_width  = score_body_width_all(entries, contour_lut, label)

    # Normalise each signal to unit std before weighting.
    def _norm(s):
        std = s.std()
        return s / std if std > 1e-6 else np.zeros_like(s)

    raw = (w_curv   * _norm(s_curv) +
           w_motion * _norm(s_motion) +
           w_global * _norm(s_global) +
           w_width  * _norm(s_width))

    # Zero out bad frames before computing the anchor.
    valid = _spine_valid_mask(entries)
    raw[~valid] = 0.0

    # Global polarity anchor: bias every frame toward the majority-vote polarity.
    global_vote   = float(np.mean(raw[valid]) if valid.any() else 0.0)
    anchor_offset = np.sign(global_vote) * min(abs(global_vote) * 3.0, 9.0)
    raw += anchor_offset

    # Compress to [-1, 1]; re-zero bad frames so the HMM coasts.
    scores          = np.tanh(raw / 10.0)
    scores[~valid]  = 0.0
    scores          = _smooth(scores, smooth_window)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — HMM
# ─────────────────────────────────────────────────────────────────────────────

def _remove_short_runs(states: np.ndarray, min_run: int) -> np.ndarray:
    """
    Merge any state run shorter than min_run frames into the surrounding state.
    Iterates until convergence.

    Parameters
    ----------
    states  : np.ndarray of int, shape (N,)
    min_run : int

    Returns
    -------
    np.ndarray of int, shape (N,)
    """
    states  = states.copy()
    changed = True
    while changed:
        changed = False
        i, n = 0, len(states)
        while i < n:
            j = i
            while j < n and states[j] == states[i]:
                j += 1
            if (j - i) < min_run:
                left  = states[i - 1] if i > 0 else -1
                right = states[j]     if j < n else -1
                if left == -1 and right == -1:
                    i = j          # entire track is one short run — leave it
                    continue
                if   left  == -1: replacement = right
                elif right == -1: replacement = left
                else:             replacement = left   # prefer left
                states[i:j] = replacement
                changed = True
            i = j
    return states


def viterbi_head_sequence(scores: np.ndarray,
                           flip_penalty: float = 20.0,
                           stay_bonus: float   = 3.0,
                           min_run_length: int  = 10) -> np.ndarray:
    """
    Find the globally optimal head/tail state sequence via Viterbi decoding.

    State 0 → pts[0] is head. State 1 → pts[-1] is head.
    Switching states costs flip_penalty; staying in the same state gains stay_bonus.
    Short runs below min_run_length frames are removed after decoding.

    Parameters
    ----------
    scores         : np.ndarray, shape (N,)
    flip_penalty   : float
    stay_bonus     : float
    min_run_length : int

    Returns
    -------
    np.ndarray of int, shape (N,)
    """
    n   = len(scores)
    dp  = np.zeros((n, 2))
    ptr = np.zeros((n, 2), dtype=int)

    dp[0, 0] = -scores[0]
    dp[0, 1] =  scores[0]

    for t in range(1, n):
        for s in (0, 1):
            emit = scores[t] if s == 1 else -scores[t]
            stay = dp[t - 1, s]       + stay_bonus
            flip = dp[t - 1, 1 - s]  - flip_penalty
            if stay >= flip:
                dp[t, s]  = stay + emit
                ptr[t, s] = s
            else:
                dp[t, s]  = flip + emit
                ptr[t, s] = 1 - s

    states       = np.zeros(n, dtype=int)
    states[-1]   = int(np.argmax(dp[-1]))
    for t in range(n - 2, -1, -1):
        states[t] = ptr[t + 1, states[t + 1]]

    return _remove_short_runs(states, min_run_length)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def process_spine_file(outline_file_in : str,
                       spine_file_in   : str,
                       flip_penalty    : float = 40.0,
                       stay_bonus      : float = 3.0,
                       min_run_length  : int   = 20,
                       w_curv          : float = 3.0,
                       w_motion        : float = 6.0,
                       w_global        : float = 4.0,
                       w_width         : float = 2.0,
                       smooth_window   : int   = 20) -> None:
    """
    Full pipeline: load files → correct head/tail → write output.
    Only the point order is modified; coordinate values are unchanged.

    Parameters
    ----------
    outline_file_in : str
    spine_file_in   : str
    spine_file_out  : str
    flip_penalty    : float
    stay_bonus      : float
    min_run_length  : int
    w_curv          : float
    w_motion        : float
    w_global        : float
    w_width         : float
    smooth_window   : int
    """
    outline_data = parse_file(outline_file_in)
    spine_data   = parse_file(spine_file_in)
    contour_lut  = build_contour_lookup(outline_data)

    with open(spine_file_in, "w") as fout:
        for label in sorted(spine_data):
            entries = spine_data[label]
            entries = stabilize_temporal(entries)

            scores = combine_and_anchor(entries, contour_lut, label,
                                        w_curv=w_curv,
                                        w_motion=w_motion,
                                        w_global=w_global,
                                        w_width=w_width,
                                        smooth_window=smooth_window)

            states = viterbi_head_sequence(scores,
                                           flip_penalty=flip_penalty,
                                           stay_bonus=stay_bonus,
                                           min_run_length=min_run_length)

            padded = f"{label:05d}"
            for (date_str, t, pts), s in zip(entries, states):
                if s == 1:
                    pts = pts[::-1]
                row = f"{date_str} {padded} {t:.3f}"
                for x, y in pts:
                    row += f" {x:.4f} {y:.4f}"
                fout.write(row + "\n")
