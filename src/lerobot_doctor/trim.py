"""Episode trimming — remove idle/static frames from episodes.

Detects and removes frames where the robot isn't moving (common at
the start/end of teleoperation recordings). These idle frames teach
the policy to "do nothing" which causes stuck behaviors at inference.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class VideoTrimError(RuntimeError):
    """A video cut failed or its output did not verify; the dataset copy may be
    inconsistent — rerun on a fresh copy."""


@dataclass
class TrimResult:
    episodes_trimmed: int
    frames_removed: int
    episodes_removed: int
    details: list[str]
    videos_trimmed: int = 0
    videos_reencoded: int = 0


def trim_dataset(
    root: Path,
    action_threshold: float = 0.01,
    min_active_frames: int = 10,
    trim_start: bool = True,
    trim_end: bool = True,
    remove_fully_static: bool = True,
    gripper_preroll_s: float = 0.5,
    trim_videos: bool = False,
    video_copy_slack: int = 4,
    video_crf: int = 14,
    video_cpu_used: int = 3,
    manual_cuts: dict[int, int] | None = None,
    min_remaining_frames: int = 1,
    dry_run: bool = False,
) -> TrimResult:
    """Trim idle frames from episodes.

    Args:
        root: Dataset root path
        action_threshold: Frames with action std below this are "idle"
        min_active_frames: Minimum active frames to keep an episode
        trim_start: Remove idle frames at start of episodes
        trim_end: Remove idle frames at end of episodes
        remove_fully_static: Remove episodes that are entirely static
        gripper_preroll_s: When the gripper starts moving before the arm, keep
            this many seconds of frames before the gripper transition so the
            episode begins from the resting (open-gripper) state. 0 disables it.
        trim_videos: Also cut each episode's mp4 to the kept frame range and
            rebase timestamps to start at 0. When the trim start can be snapped
            to a video keyframe within ``video_copy_slack`` frames (always true
            for LeRobot's default g=2 encoding), the cut is a lossless stream
            copy — no re-encode, bit-identical frames. Otherwise the segment is
            re-encoded with libaom-av1 (still AV1) at ``video_crf``.
        video_copy_slack: Max extra leading frames to keep in order to align the
            trim start with a keyframe and stay lossless (default 4).
        video_crf: libaom-av1 CRF for the re-encode fallback (default 14,
            visually transparent).
        video_cpu_used: libaom-av1 speed/quality knob for the fallback
            (0 = slowest/best, default 3).
        manual_cuts: Manual mode (used by ``cut_dataset``): {episode_index:
            (start, end)} — keep frames [start, end), end=None meaning the
            episode's end. Episodes not listed pass through untouched;
            activity detection, static removal and gripper pre-roll are all
            skipped.
        min_remaining_frames: Manual mode only — refuse a cut that would leave
            an episode shorter than this.
        dry_run: Only report, don't modify
    """
    result = TrimResult(episodes_trimmed=0, frames_removed=0, episodes_removed=0, details=[])

    data_dir = root / "data"
    if not data_dir.exists():
        return result

    # Refuse to write-trim non-v2.x datasets before touching any file. v3.0 packs
    # multiple episodes per parquet/mp4 and stores rich per-episode metadata
    # (dataset_from/to_index, per-camera video from/to_timestamp, per-episode
    # stats) that this tool does not rebuild. Trim the v2.1 dataset, then convert
    # to v3.0. Dry-run is always allowed (read-only) so v3.0 can still be previewed.
    info_path = root / "meta" / "info.json"
    info = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
        except Exception:
            info = {}
    codebase_version = str(info.get("codebase_version", ""))
    if (not dry_run or trim_videos) and codebase_version and not codebase_version.startswith("v2"):
        raise NotImplementedError(
            f"trim: in-place trimming (and --trim-videos) supports v2.x datasets only "
            f"(found codebase_version={codebase_version!r}): v3.0 packs many episodes per "
            f"parquet/mp4 with per-episode byte/timestamp offsets that this tool does not "
            f"rebuild. Trim the v2.1 version, then convert it to v3.0 with the official "
            f"LeRobot converter."
        )

    # Gripper pre-roll setup: identify the gripper action dimension(s) by name so
    # we can detect when the gripper moves before the arm.
    fps = float(info.get("fps") or 30)
    preroll_frames = int(round(gripper_preroll_s * fps)) if trim_start else 0
    action_dim_names = (info.get("features", {}).get("action", {}) or {}).get("names")
    _GRIP_TOKENS = ("gripper", "knuckle", "finger", "robotiq", "hand", "claw")

    def _is_gripper(name) -> bool:
        return any(tok in str(name).lower() for tok in _GRIP_TOKENS)

    parquet_files = sorted(data_dir.rglob("*.parquet"))

    for pf in parquet_files:
        try:
            table = pq.read_table(pf)
            if "episode_index" not in table.column_names:
                continue

            # Find action columns (not needed in manual-cut mode)
            action_cols = [c for c in table.column_names if c.startswith("action")]
            if not action_cols and manual_cuts is None:
                continue

            ep_col = table.column("episode_index").to_pylist()
            episodes = {}
            for i, ep in enumerate(ep_col):
                if ep not in episodes:
                    episodes[ep] = []
                episodes[ep].append(i)

            rows_to_keep = []
            pending_cuts = []      # (ep_idx, mode, {video_key: path}, start, n_kept, gop, pix_fmt)
            removed_eps = []
            for ep_idx in sorted(episodes.keys()):
                row_indices = episodes[ep_idx]

                if manual_cuts is not None:
                    # Manual cut mode: only listed episodes are touched, and
                    # the cut range is taken as-is (no activity detection, no
                    # static removal, no gripper pre-roll).
                    if ep_idx not in manual_cuts:
                        rows_to_keep.extend(row_indices)
                        continue
                    start, end_req = manual_cuts[ep_idx]
                    end = len(row_indices) if end_req is None else min(end_req, len(row_indices))
                    if end - start < max(min_remaining_frames, 1):
                        raise ValueError(
                            f"cut: episode {ep_idx} has {len(row_indices)} frames; "
                            f"keeping [{start}:{end}) leaves {max(end - start, 0)} "
                            f"(minimum {max(min_remaining_frames, 1)})"
                        )
                else:
                    # Get action values for this episode.
                    # Action columns may be stored either as multiple scalar
                    # columns (one feature each) or as a single vector-valued
                    # column where each row is a list/array of shape [D]
                    # (the standard LeRobot packed layout). Handle both.
                    action_parts = []
                    col_names = []  # name per column of action_matrix (for gripper detection)
                    for col in action_cols:
                        try:
                            vals = np.array([table.column(col)[i].as_py() for i in row_indices], dtype=np.float64)
                        except (ValueError, TypeError):
                            continue
                        if vals.ndim == 1:
                            # scalar-per-frame column -> one feature dimension
                            action_parts.append(vals.reshape(-1, 1))
                            col_names.append(col)
                        elif vals.ndim == 2:
                            # vector-per-frame column (e.g. one "action" list of shape [D])
                            action_parts.append(vals)
                            if col == "action" and action_dim_names and len(action_dim_names) == vals.shape[1]:
                                col_names.extend(action_dim_names)
                            else:
                                col_names.extend(f"{col}[{k}]" for k in range(vals.shape[1]))
                        # ignore higher-dim / ragged columns

                    if not action_parts:
                        rows_to_keep.extend(row_indices)
                        continue

                    action_matrix = np.hstack(action_parts) if len(action_parts) > 1 else action_parts[0]

                    # Compute per-frame "activity" as action change magnitude
                    if len(action_matrix) < 2:
                        rows_to_keep.extend(row_indices)
                        continue

                    diffs = np.abs(np.diff(action_matrix, axis=0))
                    activity = np.concatenate([[0], diffs.mean(axis=1)])
                    # Frame 0 has no predecessor, so its activity is undefined;
                    # let it inherit frame 1's. Otherwise the first frame always
                    # counts as idle and every re-run of trim shaves one more
                    # leading frame off already-trimmed episodes (systematic
                    # 1-frame creep).
                    activity[0] = activity[1]
                    is_active = activity > action_threshold

                    # Find first and last active frames
                    active_indices = np.where(is_active)[0]

                    if len(active_indices) < min_active_frames:
                        if remove_fully_static:
                            result.episodes_removed += 1
                            result.frames_removed += len(row_indices)
                            removed_eps.append(ep_idx)
                            result.details.append(f"Episode {ep_idx}: removed (fully static)")
                            continue
                        else:
                            rows_to_keep.extend(row_indices)
                            continue

                    start = active_indices[0] if trim_start else 0
                    end = active_indices[-1] + 1 if trim_end else len(row_indices)

                    # Gripper pre-roll: if the gripper begins moving before the
                    # arm (e.g. closing the gripper to nudge an object), keep a
                    # short window of frames before that transition so the
                    # episode starts from the resting/open-gripper state and the
                    # open->close transition is captured, instead of starting
                    # mid-close.
                    if (trim_start and preroll_frames > 0
                            and len(col_names) == action_matrix.shape[1]):
                        gripper_mask = np.array([_is_gripper(n) for n in col_names], dtype=bool)
                        if gripper_mask.any() and not gripper_mask.all():
                            arm_activity = np.concatenate([[0.0], diffs[:, ~gripper_mask].mean(axis=1)])
                            arm_active = np.where(arm_activity > action_threshold)[0]
                            gripper_leads = (len(arm_active) == 0) or (start < arm_active[0])
                            if gripper_leads:
                                start = max(0, start - preroll_frames)

                # Video trim planning: snap the trim start down to the nearest
                # frame that is a keyframe in every camera so the mp4 can be cut
                # with a lossless stream copy (LeRobot encodes with g=2, so the
                # snap costs at most 1 extra frame). If no keyframe is within
                # video_copy_slack frames, fall back to an AV1 re-encode of the
                # kept segment. The (possibly snapped) start is used for the
                # parquet rows too, so rows and video frames stay 1:1 aligned.
                video_note = ""
                if trim_videos and (start > 0 or end < len(row_indices)):
                    ep_videos = _episode_video_paths(root, info, ep_idx)
                    if manual_cuts is not None and len(ep_videos) < len(
                            [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]):
                        raise VideoTrimError(
                            f"cut: episode {ep_idx}: expected video files not found on disk"
                        )
                    if ep_videos:
                        mode, start, gop, pix_fmt = _plan_video_trim(
                            ep_videos, start, len(row_indices), video_copy_slack, ep_idx
                        )
                        if start < end:  # may still be a real trim after snapping
                            pending_cuts.append(
                                (ep_idx, mode, ep_videos, start, end - start, gop, pix_fmt)
                            )
                            video_note = (" [videos: lossless copy]" if mode == "copy"
                                          else " [videos: av1 re-encode]")

                trimmed_indices = row_indices[start:end]
                n_removed = len(row_indices) - len(trimmed_indices)

                if n_removed > 0:
                    result.episodes_trimmed += 1
                    result.frames_removed += n_removed
                    if manual_cuts is not None:
                        n_head, n_tail = start, len(row_indices) - end
                        parts = ([f"first {n_head}"] if n_head else []) + \
                                ([f"last {n_tail}"] if n_tail else [])
                        what = f"cut {' + '.join(parts)} frames"
                    else:
                        what = f"trimmed {n_removed} idle frames"
                    result.details.append(
                        f"Episode {ep_idx}: {what} "
                        f"({len(row_indices)} → {len(trimmed_indices)}){video_note}"
                    )
                elif pending_cuts and pending_cuts[-1][0] == ep_idx:
                    pending_cuts.pop()  # snap ate the whole trim; nothing to cut

                rows_to_keep.extend(trimmed_indices)

            # Write trimmed table
            if len(rows_to_keep) < len(ep_col) and not dry_run:
                if not rows_to_keep:
                    # Every episode in this file was removed: drop the file
                    # instead of leaving an empty parquet behind.
                    pf.unlink()
                else:
                    trimmed_table = table.take(rows_to_keep)
                    # Reindex frames within each episode
                    trimmed_table = _reindex_frames(trimmed_table)
                    if trim_videos:
                        # Videos are being cut to start at the kept range, so
                        # rebase timestamps to start at 0 per episode to match.
                        trimmed_table = _rebase_timestamps(trimmed_table)
                    pq.write_table(trimmed_table, pf)

            if not dry_run and trim_videos:
                for ep_idx, mode, ep_videos, start, n_kept, gop, pix_fmt in pending_cuts:
                    for vkey, vpath in ep_videos.items():
                        _cut_video_file(vpath, start, n_kept, fps, mode,
                                        video_crf, video_cpu_used, gop, pix_fmt)
                    result.videos_trimmed += len(ep_videos)
                    if mode == "reencode":
                        result.videos_reencoded += len(ep_videos)
                # Removed episodes no longer have rows: drop their videos too.
                for ep_idx in removed_eps:
                    for vpath in _episode_video_paths(root, info, ep_idx).values():
                        vpath.unlink()

        except VideoTrimError:
            raise
        except Exception as e:
            result.details.append(f"Error processing {pf.name}: {e}")

    # Update metadata
    if not dry_run and (result.frames_removed > 0 or result.episodes_removed > 0):
        _update_metadata_after_trim(root)

    return result


def cut_dataset(
    root: Path,
    cuts: dict[int, str],
    dry_run: bool = False,
    force: bool = False,
    video_copy_slack: int = 4,
    video_crf: int = 14,
    video_cpu_used: int = 3,
) -> TrimResult:
    """Manually cut frames off the start and/or end of specific episodes.

    ``cuts`` maps episode_index -> cut spec, where each point is a frame
    count ("120") or seconds with an "s" suffix ("3.5s", resolved via the
    dataset fps):

    - "START":       drop the first START frames (keep [START, end))
    - "START:END":   keep frames [START, END)
    - ":END":        drop everything from END onward (keep [0, END))

    Cut points are relative to the dataset's *current* state (what you see
    when playing it back). Videos are always cut too — lossless
    keyframe-aligned stream copy when possible (the start may move up to
    ``video_copy_slack`` frames earlier), AV1 re-encode otherwise (always for
    B-frame streams) — and timestamps are rebased to 0.

    Unless ``force`` is set, a cut that would leave an episode shorter than
    2 seconds is refused. All validation happens before any file is touched.
    """
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise ValueError("cut: meta/info.json not found — not a LeRobot dataset root")
    info = json.loads(info_path.read_text())
    fps = float(info.get("fps") or 30)

    def to_frames(ep, point):
        p = point.strip().lower()
        try:
            return int(round(float(p[:-1]) * fps)) if p.endswith("s") else int(p)
        except ValueError:
            raise ValueError(
                f"cut: bad cut point {point!r} for episode {ep} — use frames "
                f"('120') or seconds ('3.5s')"
            )

    resolved: dict[int, tuple[int, int | None]] = {}
    for ep, spec in cuts.items():
        start_s, sep, end_s = str(spec).strip().partition(":")
        start = to_frames(ep, start_s) if start_s.strip() else 0
        end = to_frames(ep, end_s) if sep and end_s.strip() else None
        if start == 0 and end is None:
            raise ValueError(f"cut: episode {ep}: spec {spec!r} cuts nothing")
        if start < 0 or (end is not None and end <= 0):
            raise ValueError(f"cut: episode {ep}: invalid cut range in {spec!r}")
        resolved[int(ep)] = (start, end)

    # Validate every cut against episode lengths before touching any file.
    ep_jsonl = root / "meta" / "episodes.jsonl"
    if not ep_jsonl.exists():
        raise ValueError("cut: meta/episodes.jsonl not found (a v2.x dataset is required)")
    lengths = {}
    for line in ep_jsonl.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            lengths[d["episode_index"]] = d["length"]
    min_remaining = 1 if force else int(round(2 * fps))
    for ep, (start, end) in resolved.items():
        if ep not in lengths:
            raise ValueError(f"cut: episode {ep} not found in dataset "
                             f"(has episodes {min(lengths)}..{max(lengths)})")
        eff_end = lengths[ep] if end is None else min(end, lengths[ep])
        remaining = eff_end - start
        if remaining < min_remaining:
            raise ValueError(
                f"cut: episode {ep} has {lengths[ep]} frames; keeping "
                f"[{start}:{eff_end}) leaves {max(remaining, 0)} frames — less than "
                + ("1 frame" if force else "2 seconds (use --force to override)")
            )

    return trim_dataset(
        root,
        manual_cuts=resolved,
        min_remaining_frames=min_remaining,
        trim_videos=True,
        video_copy_slack=video_copy_slack,
        video_crf=video_crf,
        video_cpu_used=video_cpu_used,
        dry_run=dry_run,
    )


def _reindex_frames(table: pa.Table) -> pa.Table:
    """Rebuild frame_index to be 0-based per episode."""
    if "frame_index" not in table.column_names:
        return table

    ep_col = table.column("episode_index").to_pylist()
    new_frames = []
    current_ep = None
    counter = 0

    for ep in ep_col:
        if ep != current_ep:
            current_ep = ep
            counter = 0
        new_frames.append(counter)
        counter += 1

    return table.set_column(
        table.column_names.index("frame_index"),
        "frame_index",
        pa.array(new_frames)
    )


def _rebase_timestamps(table: pa.Table) -> pa.Table:
    """Shift each episode's timestamps so they start at 0 (matches cut videos)."""
    if "timestamp" not in table.column_names:
        return table
    ts = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
    ep = np.asarray(table.column("episode_index").to_pylist())
    # Episodes are stored contiguously; subtract each episode's first timestamp.
    boundaries = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    for b, e in zip(boundaries, np.r_[boundaries[1:], len(ep)]):
        ts[b:e] -= ts[b]
    col_idx = table.column_names.index("timestamp")
    orig_type = table.schema.field("timestamp").type
    return table.set_column(col_idx, "timestamp", pa.array(ts).cast(orig_type))


def _episode_video_paths(root: Path, info: dict, ep_idx: int) -> dict[str, Path]:
    """Map video_key -> mp4 path for one episode (v2.x per-episode layout)."""
    tmpl = info.get("video_path")
    if not tmpl:
        return {}
    chunks_size = int(info.get("chunks_size") or 1000)
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    out = {}
    for key in video_keys:
        try:
            p = root / tmpl.format(episode_chunk=ep_idx // chunks_size,
                                   video_key=key, episode_index=ep_idx)
        except (KeyError, IndexError):
            raise VideoTrimError(
                f"video_path template {tmpl!r} is not a v2.x per-episode layout "
                f"(expected placeholders episode_chunk/video_key/episode_index) — "
                f"the dataset metadata looks corrupted or v3.0"
            )
        if p.exists():
            out[key] = p
    return out


def _probe_video_packets(path: Path) -> tuple[int, list[int], bool]:
    """Return (n_packets, keyframe indices in presentation order, reordered).

    ``reordered`` is True when packets are stored out of presentation order
    (B-frames): stream-copy cuts are then unsafe (a cut at the end could drop
    packets that later frames reference), so callers must re-encode instead.
    """
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts,flags", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise VideoTrimError(f"ffprobe failed on {path}: {proc.stderr.strip()}")
    pkts = []  # (pts, is_key)
    for line in proc.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2 or not parts[0].lstrip("-").isdigit():
            continue
        pkts.append((int(parts[0]), "K" in parts[1]))
    pts_in_file_order = [p for p, _ in pkts]
    reordered = any(b < a for a, b in zip(pts_in_file_order, pts_in_file_order[1:]))
    pkts.sort(key=lambda t: t[0])  # presentation order
    keyframes = [i for i, (_, k) in enumerate(pkts) if k]
    return len(pkts), keyframes, reordered


def _plan_video_trim(ep_videos: dict[str, Path], start: int, ep_len: int,
                     copy_slack: int, ep_idx: int):
    """Decide how to cut this episode's videos.

    Returns (mode, start, gop, pix_fmt) where mode is "copy" (lossless stream
    copy from a keyframe common to all cameras, possibly moving `start` down by
    at most `copy_slack` frames) or "reencode" (frame-exact libaom-av1
    re-encode at the original `start`).
    """
    snapped = []
    gop = None
    any_reordered = False
    for key, path in ep_videos.items():
        n_pkts, keyframes, reordered = _probe_video_packets(path)
        any_reordered = any_reordered or reordered
        if n_pkts != ep_len:
            raise VideoTrimError(
                f"episode {ep_idx}: video {path.name} ({key}) has {n_pkts} frames "
                f"but the parquet has {ep_len} rows — refusing to cut misaligned video"
            )
        if not keyframes or keyframes[0] != 0:
            raise VideoTrimError(f"episode {ep_idx}: {path.name} does not start on a keyframe")
        kf_at_or_before = max(i for i in keyframes if i <= start)
        snapped.append(kf_at_or_before)
        if len(keyframes) > 1:
            spacing = min(b - a for a, b in zip(keyframes, keyframes[1:]))
            gop = spacing if gop is None else min(gop, spacing)
    common_start = min(snapped)
    gop = gop or 2
    # B-frame streams store packets out of display order; stream-copy cuts can
    # sever references, so only re-encoding is frame-safe there.
    if not any_reordered and start - common_start <= copy_slack:
        return "copy", common_start, gop, None
    return "reencode", start, gop, None


def _probe_pix_fmt(path: Path) -> str:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    fmt = proc.stdout.strip().splitlines()[0].strip() if proc.stdout.strip() else ""
    return fmt or "yuv420p"


def _cut_video_file(path: Path, start: int, n_frames: int, fps: float, mode: str,
                    crf: int, cpu_used: int, gop: int, pix_fmt: str | None):
    """Cut `path` in place to frames [start, start+n_frames), first frame at pts 0.

    copy: lossless stream copy starting at the keyframe at `start` (seek point
    is placed half a frame past it so ffmpeg's keyframe-at-or-before seek lands
    exactly on it); packet count bounded by -frames:v.
    reencode: frame-exact select() + libaom-av1 (AV1 kept, high quality).
    Output is written to a temp file, verified (frame count and first pts == 0),
    then atomically moved over the original.
    """
    tmp = path.with_name(path.stem + ".trim_tmp.mp4")
    if mode == "copy":
        cmd = ["ffmpeg", "-v", "error", "-y"]
        if start > 0:
            cmd += ["-ss", f"{(start + 0.5) / fps:.6f}"]
        cmd += ["-i", str(path), "-map", "0:v:0", "-c", "copy",
                "-frames:v", str(n_frames), "-avoid_negative_ts", "make_zero", str(tmp)]
    else:
        pix_fmt = pix_fmt or _probe_pix_fmt(path)
        vf = (f"select='between(n,{start},{start + n_frames - 1})',"
              f"setpts=N/FRAME_RATE/TB")
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path),
               "-vf", vf, "-vsync", "0", "-an",
               "-c:v", "libaom-av1", "-crf", str(crf), "-b:v", "0",
               "-cpu-used", str(cpu_used), "-row-mt", "1",
               "-g", str(gop), "-pix_fmt", pix_fmt, str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise VideoTrimError(f"ffmpeg cut failed on {path.name}: {proc.stderr.strip()}")

    # Verify before replacing the original.
    n_out, _, _ = _probe_video_packets(tmp)
    first_pts = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1",
         "-show_entries", "packet=pts", "-of", "csv=p=0", str(tmp)],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    first_pts_val = int(first_pts[0].split(",")[0]) if first_pts else -1
    if n_out != n_frames or first_pts_val != 0:
        tmp.unlink(missing_ok=True)
        raise VideoTrimError(
            f"cut verification failed on {path.name}: got {n_out} frames "
            f"(expected {n_frames}), first pts {first_pts_val} (expected 0)"
        )
    tmp.replace(path)


def _update_metadata_after_trim(root: Path):
    """Update all metadata to match the trimmed data (v2.x layout).

    Rebuilds, in a version-correct way:
      - info.json total_frames / total_episodes (via fix._fix_metadata)
      - the global ``index`` column (contiguous 0..N-1) and ``frame_index``
        (0-based per episode) across the data files
      - meta/episodes.jsonl episode lengths (preserving tasks)
      - meta/episodes_stats.jsonl per-episode stats for numeric features
        (image/video feature stats are kept as-is: they can't be recomputed
        without decoding video, and trimming idle head/tail frames — visually
        near-identical to the retained boundary frames — barely affects them)
      - meta/stats.json (legacy aggregate, if present): numeric features are
        recomputed exactly from the trimmed data; image features are
        re-aggregated from the per-episode stats

    Note on ``timestamp``: without --trim-videos it is deliberately left
    untouched — the videos are not re-cut, and LeRobot fetches frames by
    timestamp, so keeping the original timestamps is what keeps the retained
    frames aligned to the untrimmed mp4. With --trim-videos the mp4s are cut to
    the kept range (starting at t=0), so timestamps are rebased to 0 per
    episode before this repair runs, keeping both in sync.
    """
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}

    # info.json totals
    from .fix import _fix_metadata, FixResult
    _fix_metadata(root, FixResult(fixed=[], skipped=[], errors=[]), dry_run=False)

    _rebuild_episode_metadata_v2(root, info)


def _col_as_2d(table: pa.Table, key: str):
    """Return a numeric column as a float (n, d) array, or None if non-numeric."""
    try:
        vals = np.array(table.column(key).to_pylist(), dtype=np.float64)
    except (ValueError, TypeError):
        return None
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    if vals.ndim != 2:
        return None
    return vals


def _recompute_episode_stats(table: pa.Table, key_order, video_keys, old_stats: dict) -> dict:
    """Recompute per-episode stats for numeric features; keep image/video stats."""
    stats = {}
    for key in key_order:
        if key in video_keys:
            stats[key] = old_stats.get(key)  # can't recompute without video decode
            continue
        arr = _col_as_2d(table, key) if key in table.column_names else None
        if arr is None:
            stats[key] = old_stats.get(key)
            continue
        stats[key] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(arr.shape[0])],
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q10": np.percentile(arr, 10, axis=0).tolist(),
            "q50": np.percentile(arr, 50, axis=0).tolist(),
            "q90": np.percentile(arr, 90, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
        }
    return stats


def _rebuild_episode_metadata_v2(root: Path, info: dict):
    """Rebuild episodes.jsonl + episodes_stats.jsonl and reindex the global index."""
    meta = root / "meta"
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]

    # Preserve tasks from the existing episodes.jsonl.
    tasks_map = {}
    ep_jsonl = meta / "episodes.jsonl"
    if ep_jsonl.exists():
        for line in ep_jsonl.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                tasks_map[d["episode_index"]] = d.get("tasks")

    # Preserve image stats + the feature key order from the existing stats file.
    old_stats_map = {}
    key_order = None
    stats_jsonl = meta / "episodes_stats.jsonl"
    if stats_jsonl.exists():
        for line in stats_jsonl.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                old_stats_map[d["episode_index"]] = d["stats"]
                if key_order is None:
                    key_order = list(d["stats"].keys())

    data_files = sorted((root / "data").rglob("*.parquet"))
    running_index = 0
    new_eps, new_stats = [], []

    for pf in data_files:
        table = pq.read_table(pf)
        if "episode_index" not in table.column_names:
            continue
        n = table.num_rows
        ep_idx = table.column("episode_index")[0].as_py()

        # Contiguous global index and 0-based per-episode frame_index.
        if "index" in table.column_names:
            table = table.set_column(
                table.column_names.index("index"), "index",
                pa.array(list(range(running_index, running_index + n))),
            )
        if "frame_index" in table.column_names:
            table = table.set_column(
                table.column_names.index("frame_index"), "frame_index",
                pa.array(list(range(n))),
            )
        running_index += n
        pq.write_table(table, pf)

        new_eps.append({"episode_index": ep_idx, "tasks": tasks_map.get(ep_idx), "length": n})
        if key_order is not None:
            new_stats.append({
                "episode_index": ep_idx,
                "stats": _recompute_episode_stats(table, key_order, video_keys,
                                                  old_stats_map.get(ep_idx, {})),
            })

    if new_eps:
        ep_jsonl.write_text("\n".join(json.dumps(e) for e in new_eps) + "\n")
    if new_stats:
        stats_jsonl.write_text("\n".join(json.dumps(s) for s in new_stats) + "\n")
        _rebuild_stats_json(root, video_keys, data_files, new_stats)

    # Drop the stray file the old repair path used to write (wrong for v2.x).
    stray = meta / "episodes" / "00000.parquet"
    if stray.exists():
        stray.unlink()


def _rebuild_stats_json(root: Path, video_keys, data_files, new_stats):
    """Rebuild the legacy aggregate meta/stats.json after a trim.

    v2.1 readers use episodes_stats.jsonl and ignore this file, but keep it
    coherent anyway: numeric features are recomputed exactly (quantiles
    included) from the trimmed parquet; image/video features are aggregated
    from the per-episode stats (elementwise min/max, count-weighted mean,
    pooled std, count-weighted quantiles — per-episode image stats themselves
    are not recomputed).
    """
    stats_path = root / "meta" / "stats.json"
    if not stats_path.exists():
        return
    old = json.loads(stats_path.read_text())

    # Exact global stats for numeric (parquet-backed) features.
    numeric_arrays: dict[str, list[np.ndarray]] = {}
    for pf in data_files:
        if not pf.exists():  # fully-static episodes may have been deleted
            continue
        table = pq.read_table(pf)
        for key in old:
            if key in video_keys or key not in table.column_names:
                continue
            arr = _col_as_2d(table, key)
            if arr is not None:
                numeric_arrays.setdefault(key, []).append(arr)

    new = {}
    for key, entry in old.items():
        if key in numeric_arrays:
            arr = np.concatenate(numeric_arrays[key], axis=0)
            new[key] = {
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "count": [int(arr.shape[0])],
                "q01": np.percentile(arr, 1, axis=0).tolist(),
                "q10": np.percentile(arr, 10, axis=0).tolist(),
                "q50": np.percentile(arr, 50, axis=0).tolist(),
                "q90": np.percentile(arr, 90, axis=0).tolist(),
                "q99": np.percentile(arr, 99, axis=0).tolist(),
            }
        elif key in video_keys:
            agg = _aggregate_feature_stats([s["stats"].get(key) for s in new_stats])
            new[key] = agg if agg is not None else entry
        else:
            new[key] = entry  # unknown feature: keep the old entry

    stats_path.write_text(json.dumps(new, indent=4) + "\n")


def _aggregate_feature_stats(per_episode: list):
    """Aggregate per-episode stats dicts for one feature into dataset-level stats."""
    entries = [e for e in per_episode if e and e.get("count")]
    if not entries:
        return None
    counts = np.array([float(e["count"][0]) for e in entries])
    total = counts.sum()
    if total <= 0:
        return None
    mins = np.array([e["min"] for e in entries], dtype=np.float64)
    maxs = np.array([e["max"] for e in entries], dtype=np.float64)
    means = np.array([e["mean"] for e in entries], dtype=np.float64)
    stds = np.array([e["std"] for e in entries], dtype=np.float64)
    w = (counts / total).reshape((-1,) + (1,) * (means.ndim - 1))
    mean = (w * means).sum(axis=0)
    # Pooled variance: E[x^2] - E[x]^2 over all episodes.
    var = (w * (stds ** 2 + means ** 2)).sum(axis=0) - mean ** 2
    out = {
        "min": mins.min(axis=0).tolist(),
        "max": maxs.max(axis=0).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(var, 0.0)).tolist(),
        "count": [int(total)],
    }
    for q in ("q01", "q10", "q50", "q90", "q99"):
        if all(q in e for e in entries):
            qs = np.array([e[q] for e in entries], dtype=np.float64)
            out[q] = (w * qs).sum(axis=0).tolist()  # count-weighted approximation
    return out
