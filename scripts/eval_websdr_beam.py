#!/usr/bin/env python3
"""Benchmark the pure beam/timing decoder on labelled WebSDR recordings."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.audio_io import read_wav_mono
from morse_char_recognizer.beam_decode import BeamConfig, beam_decode_audio
from morse_char_recognizer.classes import CLASS_PRESETS, parse_class_spec
from morse_char_recognizer.live import (
    CarrierSegment,
    alignment_summary,
    carrier_segments_from_track,
    carrier_track_is_stable,
    compare_text,
    default_label_path,
    estimate_carrier_hz,
    estimate_carrier_track,
    read_label,
)
from morse_char_recognizer.segment import SegmentConfig


def main() -> None:
    args = _parse_args()
    wavs = _collect_wavs(args.inputs)
    if not wavs:
        raise SystemExit("no WAV files found")

    results = [_evaluate_file(path, args) for path in wavs]
    summary = _summarize(results)
    _print_summary(results, summary)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "metadata": _metadata(args),
                    "summary": summary,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {out}")

    if args.preview_dir:
        preview_dir = Path(args.preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            _write_preview(Path(result["wav"]), result, preview_dir, args)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="*", default=["livetests/websdr"])
    p.add_argument("--label", default=None, help="truth .txt for a single WAV")
    p.add_argument("--allowed-class-preset", choices=tuple(CLASS_PRESETS), default="ham-copy")
    p.add_argument("--allowed-classes", default=None)
    p.add_argument("--freq-min", type=float, default=400.0)
    p.add_argument("--freq-max", type=float, default=900.0)
    p.add_argument("--track-window-s", type=float, default=4.0)
    p.add_argument("--track-hop-s", type=float, default=2.0)
    p.add_argument("--track-jump-hz", type=float, default=60.0)
    p.add_argument("--stable-tolerance-hz", type=float, default=50.0)
    p.add_argument("--min-carrier-segment-s", type=float, default=6.0)
    p.add_argument("--no-track", action="store_true")
    p.add_argument("--bandpass-width-hz", type=float, default=70.0)
    p.add_argument(
        "--threshold-mode",
        choices=("fixed", "adaptive", "hysteresis"),
        default="hysteresis",
    )
    p.add_argument("--threshold", type=float, default=0.22)
    p.add_argument("--hysteresis-low", type=float, default=0.12)
    p.add_argument("--smoothing-ms", type=float, default=4.0)
    p.add_argument("--noise-floor-percentile", type=float, default=20.0)
    p.add_argument("--min-keydown-ms", type=float, default=10.0)
    p.add_argument("--merge-gap-ms", type=float, default=8.0)
    p.add_argument("--min-unit-ms", type=float, default=35.0)
    p.add_argument("--max-unit-ms", type=float, default=160.0)
    p.add_argument("--min-element-units", type=float, default=0.45)
    p.add_argument("--dot-dash-split", type=float, default=1.8)
    p.add_argument("--char-cost", type=float, default=0.0)
    p.add_argument("--boundary-threshold", type=float, default=2.0)
    p.add_argument("--w-internal", type=float, default=1.0)
    p.add_argument("--w-boundary", type=float, default=1.0)
    p.add_argument("--word-gap-units", type=float, default=5.0)
    p.add_argument("--alignment-sample-limit", type=int, default=80)
    p.add_argument("--json-out", default=None)
    p.add_argument("--preview-dir", default=None)
    return p.parse_args()


def _collect_wavs(inputs: list[str]) -> list[Path]:
    wavs: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            wavs.extend(sorted(path.glob("*.wav")))
        elif path.suffix.lower() == ".wav":
            wavs.append(path)
    return wavs


def _evaluate_file(wav_path: Path, args: argparse.Namespace) -> dict:
    sample_rate, audio = read_wav_mono(wav_path)
    label_path = Path(args.label) if args.label else default_label_path(wav_path)
    reference = read_label(label_path)
    track = (
        []
        if args.no_track
        else estimate_carrier_track(
            audio,
            sample_rate,
            args.freq_min,
            args.freq_max,
            window_s=args.track_window_s,
            hop_s=args.track_hop_s,
        )
    )
    stable = carrier_track_is_stable(track, tolerance_hz=args.stable_tolerance_hz)
    carrier_segments = _carrier_segments(audio, sample_rate, track, stable, args)
    decoded_parts = []
    segment_reports = []
    for segment in carrier_segments:
        decoded, elements, unit_ms = _decode_carrier_segment(audio, sample_rate, segment, args)
        if decoded:
            decoded_parts.append(decoded)
        segment_reports.append(
            {
                "start_s": segment.start_s,
                "end_s": segment.end_s,
                "center_hz": segment.center_hz,
                "track_frames": segment.frames,
                "decoded": decoded,
                "elements": elements,
                "unit_ms": unit_ms,
            }
        )
    decoded = " ".join(decoded_parts).strip()
    comparison = compare_text(reference, decoded)
    return {
        "wav": str(wav_path),
        "label": str(label_path) if label_path else None,
        "duration_s": audio.size / sample_rate,
        "sample_rate": sample_rate,
        "reference": reference,
        "decoded": decoded,
        "cer": comparison.cer,
        "accuracy": comparison.accuracy,
        "edit_distance": comparison.distance,
        "substitutions": comparison.substitutions,
        "insertions": comparison.insertions,
        "deletions": comparison.deletions,
        "reference_length": comparison.reference_length,
        "hypothesis_length": comparison.hypothesis_length,
        "elements": sum(segment["elements"] for segment in segment_reports),
        "unit_ms": _weighted_mean_unit(segment_reports),
        "carrier_stable": stable,
        "carrier_segments": segment_reports,
        "center_hz_track": [asdict(frame) for frame in track],
        "alignment": alignment_summary(
            reference,
            decoded,
            sample_limit=args.alignment_sample_limit,
        ),
    }


def _carrier_segments(
    audio,
    sample_rate: int,
    track,
    stable: bool,
    args: argparse.Namespace,
) -> list[CarrierSegment]:
    duration_s = audio.size / sample_rate
    if args.no_track or not track:
        center = estimate_carrier_hz(audio, sample_rate, args.freq_min, args.freq_max)
        return [CarrierSegment(0.0, duration_s, center, 0)]
    if stable:
        center = float(sorted(frame.center_hz for frame in track)[len(track) // 2])
        return [CarrierSegment(0.0, duration_s, center, len(track))]
    segments = carrier_segments_from_track(
        track,
        duration_s,
        jump_hz=args.track_jump_hz,
        min_segment_s=args.min_carrier_segment_s,
    )
    return segments or [
        CarrierSegment(
            0.0,
            duration_s,
            estimate_carrier_hz(audio, sample_rate, args.freq_min, args.freq_max),
            0,
        )
    ]


def _decode_carrier_segment(
    audio,
    sample_rate: int,
    carrier: CarrierSegment,
    args: argparse.Namespace,
) -> tuple[str, int, float | None]:
    start = max(0, int(round(carrier.start_s * sample_rate)))
    end = min(audio.size, int(round(carrier.end_s * sample_rate)))
    if end <= start:
        return "", 0, None
    chunk = audio[start:end]
    segment_config = SegmentConfig(
        method="hilbert",
        smoothing_ms=args.smoothing_ms,
        bandpass_center_hz=carrier.center_hz,
        bandpass_width_hz=args.bandpass_width_hz,
        noise_floor_percentile=args.noise_floor_percentile,
        threshold_mode=args.threshold_mode,
        threshold=args.threshold,
        hysteresis_low=args.hysteresis_low,
        min_keydown_ms=args.min_keydown_ms,
        merge_gap_ms=args.merge_gap_ms,
        min_unit_ms=args.min_unit_ms,
        max_unit_ms=args.max_unit_ms,
    )
    beam_config = BeamConfig(
        min_element_units=args.min_element_units,
        dot_dash_split=args.dot_dash_split,
        char_cost=args.char_cost,
        boundary_threshold=args.boundary_threshold,
        w_internal=args.w_internal,
        w_boundary=args.w_boundary,
        word_gap_units=args.word_gap_units,
        allowed_tokens=parse_class_spec(
            args.allowed_classes,
            preset=args.allowed_class_preset,
        ),
    )
    decoded, regions, unit_samples = beam_decode_audio(
        chunk,
        sample_rate,
        segment_config=segment_config,
        beam_config=beam_config,
    )
    unit_ms = 1000.0 * unit_samples / sample_rate if unit_samples else None
    return decoded, len(regions), unit_ms


def _weighted_mean_unit(segment_reports: list[dict]) -> float | None:
    total_elements = sum(segment["elements"] for segment in segment_reports)
    if total_elements <= 0:
        return None
    weighted = sum(
        segment["unit_ms"] * segment["elements"]
        for segment in segment_reports
        if segment["unit_ms"] is not None
    )
    return weighted / total_elements


def _summarize(results: list[dict]) -> dict:
    total_ref = sum(result["reference_length"] for result in results)
    total_distance = sum(result["edit_distance"] for result in results)
    total_subs = sum(result["substitutions"] for result in results)
    total_ins = sum(result["insertions"] for result in results)
    total_del = sum(result["deletions"] for result in results)
    return {
        "files": len(results),
        "cer": total_distance / total_ref if total_ref else None,
        "edit_distance": total_distance,
        "reference_length": total_ref,
        "substitutions": total_subs,
        "insertions": total_ins,
        "deletions": total_del,
    }


def _print_summary(results: list[dict], summary: dict) -> None:
    for result in results:
        centers = ", ".join(
            f"{segment['center_hz']:.1f}Hz"
            for segment in result["carrier_segments"]
        )
        unit = "n/a" if result["unit_ms"] is None else f"{result['unit_ms']:.1f}ms"
        print(
            f"{Path(result['wav']).name}: CER={result['cer']:.3f} "
            f"subs={result['substitutions']} ins={result['insertions']} "
            f"del={result['deletions']} elements={result['elements']} "
            f"unit={unit} centers=[{centers}]"
        )
        print(f"decoded: {result['decoded']}")
        print("top errors:", _format_top_errors(result["alignment"]))
    cer = "n/a" if summary["cer"] is None else f"{summary['cer']:.3f}"
    print(
        f"summary: files={summary['files']} CER={cer} "
        f"subs={summary['substitutions']} ins={summary['insertions']} "
        f"del={summary['deletions']} dist={summary['edit_distance']}/"
        f"{summary['reference_length']}"
    )


def _format_top_errors(alignment: dict) -> str:
    parts = []
    for key, label in (
        ("top_substitutions", "subs"),
        ("top_insertions", "ins"),
        ("top_deletions", "del"),
    ):
        counts = alignment[key][:3]
        if counts:
            formatted = ",".join(f"{item['item']}:{item['count']}" for item in counts)
            parts.append(f"{label}={formatted}")
    return " ".join(parts) if parts else "none"


def _metadata(args: argparse.Namespace) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "git_commit": _git_commit(),
        "args": vars(args),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_preview(
    wav_path: Path,
    result: dict,
    preview_dir: Path,
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt

    sample_rate, audio = read_wav_mono(wav_path)
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.specgram(audio, NFFT=512, Fs=sample_rate, noverlap=384, cmap="magma")
    for frame in result["center_hz_track"]:
        mid = (frame["start_s"] + frame["end_s"]) / 2.0
        axis.plot(mid, frame["center_hz"], marker=".", color="cyan", markersize=3)
    for segment in result["carrier_segments"]:
        axis.hlines(
            segment["center_hz"],
            segment["start_s"],
            segment["end_s"],
            colors="white",
            linewidth=1.5,
        )
    axis.set_ylim(args.freq_min, args.freq_max)
    axis.set_xlabel("time (s)")
    axis.set_ylabel("frequency (Hz)")
    axis.set_title(f"{wav_path.name} CER={result['cer']:.3f}")
    fig.tight_layout()
    out = preview_dir / f"{wav_path.stem}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
