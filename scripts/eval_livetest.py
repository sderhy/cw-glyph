"""Evaluate trained CNN decoding on labelled real WAV files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.audio_io import read_wav_mono
from morse_char_recognizer.classes import CLASS_PRESETS, parse_class_spec
from morse_char_recognizer.inference import (
    join_segment_predictions,
    load_checkpoint_model,
    predict_audio_segments,
    predict_envelopes,
    predict_envelopes_topk,
)
from morse_char_recognizer.live import (
    alignment_summary,
    compare_best_substring,
    compare_text,
    default_label_path,
    estimate_carrier_hz,
    read_label,
    slice_audio,
    window_starts,
)
from morse_char_recognizer.segment import (
    SegmentConfig,
    detect_active_regions,
    estimate_unit_samples,
    extract_segment_envelopes,
    segment_characters,
)


def main() -> None:
    args = _parse_args()
    model, classes, envelope, checkpoint_args = load_checkpoint_model(
        args.checkpoint,
        device=args.device,
    )
    aux = None
    if args.aux_checkpoint:
        aux_model, aux_classes, aux_envelope, _aux_args = load_checkpoint_model(
            args.aux_checkpoint,
            device=args.device,
        )
        if aux_classes != classes:
            raise SystemExit("aux checkpoint classes do not match primary checkpoint")
        aux = (aux_model, aux_envelope)
    if args.scale_mode != "checkpoint":
        envelope = replace(
            envelope,
            scale_mode=args.scale_mode,
            canonical_units=args.canonical_units,
        )
    allowed_classes = parse_class_spec(
        args.allowed_classes,
        preset=args.allowed_class_preset,
    )
    files = _collect_wavs(args.inputs)
    if not files:
        raise SystemExit("no WAV files found")

    all_results = []
    for wav_path in files:
        label_path = _label_for(wav_path, args.label)
        reference = read_label(label_path)
        sample_rate, audio = read_wav_mono(wav_path)
        starts = _starts(args, audio.size / sample_rate)
        for start in starts:
            result = _evaluate_window(
                args,
                wav_path,
                reference,
                sample_rate,
                audio,
                start,
                model,
                classes,
                envelope,
                allowed_classes,
                aux,
            )
            all_results.append(result)
            _print_result(result)

    summary = _summarize(all_results, alignment_sample_limit=args.alignment_sample_limit)
    cer_text = "n/a" if summary["cer"] is None else f"{summary['cer']:.3f}"
    accuracy_text = "n/a" if summary["accuracy"] is None else f"{summary['accuracy']:.3f}"
    print(
        "summary: "
        f"windows={summary['windows']} cer={cer_text} "
        f"accuracy={accuracy_text} "
        f"subs={summary['substitutions']} ins={summary['insertions']} "
        f"del={summary['deletions']} complete_files={summary['complete_files']}"
    )
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "metadata": _metadata(args, checkpoint_args),
                    "summary": summary,
                    "results": all_results,
                },
                indent=2,
            )
        )
        print(f"wrote {out}")
    if args.jsonl_out:
        out = Path(args.jsonl_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(item) for item in all_results) + "\n")
        print(f"wrote {out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--aux-checkpoint", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--window", type=float, default=None)
    parser.add_argument("--hop", type=float, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--threshold-mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--adaptive-window-ms", type=float, default=650.0)
    parser.add_argument("--adaptive-floor-percentile", type=float, default=20.0)
    parser.add_argument("--adaptive-peak-percentile", type=float, default=90.0)
    parser.add_argument("--adaptive-min-threshold", type=float, default=0.03)
    parser.add_argument("--min-keydown-ms", type=float, default=10.0)
    parser.add_argument("--merge-gap-ms", type=float, default=8.0)
    parser.add_argument("--char-gap-units", type=float, default=2.0)
    parser.add_argument("--pad-ms", type=float, default=20.0)
    parser.add_argument("--word-gap-ms", type=float, default=250.0)
    parser.add_argument("--word-gap-mode", choices=("fixed", "unit"), default="fixed")
    parser.add_argument("--word-gap-units", type=float, default=4.5)
    parser.add_argument(
        "--scale-mode",
        choices=("checkpoint", "stretch", "unit"),
        default="checkpoint",
    )
    parser.add_argument("--canonical-units", type=float, default=24.0)
    parser.add_argument("--center-hz", type=float, default=None)
    parser.add_argument("--auto-center", action="store_true")
    parser.add_argument("--freq-min", type=float, default=250.0)
    parser.add_argument("--freq-max", type=float, default=1200.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allowed-class-preset", choices=tuple(CLASS_PRESETS), default="real")
    parser.add_argument("--allowed-classes", default=None)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-segment-units", type=float, default=0.0)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--jsonl-out", default=None)
    parser.add_argument("--alignment-sample-limit", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--match-mode",
        choices=("full", "best-substring"),
        default="full",
    )
    return parser.parse_args()


def _collect_wavs(inputs: list[str]) -> list[Path]:
    wavs: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            wavs.extend(sorted(path.rglob("*.wav")))
        elif path.suffix.lower() == ".wav":
            wavs.append(path)
    return wavs


def _label_for(wav_path: Path, explicit_label: str | None) -> Path | None:
    if explicit_label is not None:
        return Path(explicit_label)
    return default_label_path(wav_path)


def _starts(args: argparse.Namespace, audio_duration_s: float) -> list[float]:
    if args.window is None:
        return [args.start]
    starts = window_starts(audio_duration_s, args.window, args.hop or args.window)
    starts = [start for start in starts if start >= args.start]
    if args.max_windows is not None:
        starts = starts[: args.max_windows]
    return starts


def _evaluate_window(
    args: argparse.Namespace,
    wav_path: Path,
    reference: str,
    sample_rate: int,
    full_audio,
    start: float,
    model,
    classes: tuple[str, ...],
    envelope,
    allowed_classes: tuple[str, ...],
    aux,
) -> dict:
    duration = args.window if args.window is not None else args.duration
    audio, offset_s = slice_audio(full_audio, sample_rate, start, duration)

    center_hz = args.center_hz
    if args.auto_center:
        center_hz = estimate_carrier_hz(audio, sample_rate, args.freq_min, args.freq_max)
    local_envelope = envelope
    if center_hz is not None:
        local_envelope = replace(envelope, bandpass_center_hz=center_hz)

    segment_config = SegmentConfig(
        threshold_mode=args.threshold_mode,
        threshold=args.threshold,
        adaptive_window_ms=args.adaptive_window_ms,
        adaptive_floor_percentile=args.adaptive_floor_percentile,
        adaptive_peak_percentile=args.adaptive_peak_percentile,
        adaptive_min_threshold=args.adaptive_min_threshold,
        min_keydown_ms=args.min_keydown_ms,
        merge_gap_ms=args.merge_gap_ms,
        char_gap_units=args.char_gap_units,
        pad_ms=args.pad_ms,
    )
    unit_samples = estimate_unit_samples(detect_active_regions(audio, sample_rate, segment_config))
    if aux is None:
        predictions = predict_audio_segments(
            model,
            audio,
            sample_rate,
            classes=classes,
            envelope=local_envelope,
            segment_config=segment_config,
            allowed_classes=allowed_classes,
            device=args.device,
        )
    else:
        aux_model, aux_envelope = aux
        if center_hz is not None:
            aux_envelope = replace(aux_envelope, bandpass_center_hz=center_hz)
        predictions = _predict_ensemble(
            model,
            local_envelope,
            aux_model,
            aux_envelope,
            audio,
            sample_rate,
            classes,
            segment_config,
            allowed_classes,
            args.device,
        )
    raw_prediction_count = len(predictions)
    if args.min_score > 0:
        predictions = [
            (char, score, segment)
            for char, score, segment in predictions
            if score >= args.min_score
        ]
    if args.min_segment_units > 0 and unit_samples > 0:
        predictions = [
            (char, score, segment)
            for char, score, segment in predictions
            if segment.duration >= args.min_segment_units * unit_samples
        ]
    decoded = join_segment_predictions(
        predictions,
        sample_rate=sample_rate,
        word_gap_ms=args.word_gap_ms,
        unit_samples=unit_samples if args.word_gap_mode == "unit" else None,
        word_gap_units=args.word_gap_units,
    )
    diagnostic = (
        compare_best_substring(reference, decoded)
        if args.match_mode == "best-substring"
        else None
    )
    scores = [score for _char, score, _segment in predictions]
    topk_by_segment = _topk_predictions(
        model,
        audio,
        sample_rate,
        predictions,
        classes,
        local_envelope,
        allowed_classes,
        unit_samples,
        args,
    )
    detailed_predictions = [
        {
            "char": char,
            "score": score,
            "start_s": offset_s + segment.start / sample_rate,
            "end_s": offset_s + segment.end / sample_rate,
            "top_k": [
                {"char": label, "score": top_score}
                for label, top_score in topk_by_segment[index]
            ],
        }
        for index, (char, score, segment) in enumerate(predictions)
    ]
    return {
        "wav": str(wav_path),
        "start_s": offset_s,
        "duration_s": audio.size / sample_rate,
        "audio_total_duration_s": full_audio.size / sample_rate,
        "center_hz": center_hz,
        "unit_ms": unit_samples / sample_rate * 1000.0 if unit_samples else None,
        "decoded": decoded,
        "reference": reference,
        "matched_reference": None if diagnostic is None else diagnostic.reference,
        "edit_distance": None if diagnostic is None else diagnostic.distance,
        "substitutions": None if diagnostic is None else diagnostic.substitutions,
        "insertions": None if diagnostic is None else diagnostic.insertions,
        "deletions": None if diagnostic is None else diagnostic.deletions,
        "reference_length": None if diagnostic is None else diagnostic.reference_length,
        "hypothesis_length": len(decoded.replace(" ", "")),
        "cer": None if diagnostic is None else diagnostic.cer,
        "accuracy": None if diagnostic is None else diagnostic.accuracy,
        "segments": len(predictions),
        "raw_segments": raw_prediction_count,
        "rejected_segments": raw_prediction_count - len(predictions),
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "predictions": detailed_predictions,
    }


def _topk_predictions(
    model,
    audio,
    sample_rate: int,
    predictions,
    classes: tuple[str, ...],
    envelope,
    allowed_classes: tuple[str, ...],
    unit_samples: int,
    args: argparse.Namespace,
) -> list[list[tuple[str, float]]]:
    if args.top_k <= 1 or not predictions:
        return [[] for _prediction in predictions]
    segments = [segment for _char, _score, segment in predictions]
    envs = extract_segment_envelopes(
        audio,
        sample_rate,
        segments,
        envelope,
        unit_samples=unit_samples if envelope.scale_mode == "unit" else None,
    )
    return predict_envelopes_topk(
        model,
        envs,
        classes=classes,
        allowed_classes=allowed_classes,
        k=args.top_k,
        device=args.device,
    )


def _print_result(result: dict) -> None:
    metric = (
        f"diag_acc={result['accuracy']:.3f} edit={result['edit_distance']}"
        if result["accuracy"] is not None
        else "diag_acc=n/a edit=n/a"
    )
    print(
        f"{Path(result['wav']).name} start={result['start_s']:.1f}s "
        f"{metric} segments={result['segments']} decoded={result['decoded'][:80]}"
    )


def _predict_ensemble(
    model,
    envelope,
    aux_model,
    aux_envelope,
    audio,
    sample_rate: int,
    classes: tuple[str, ...],
    segment_config: SegmentConfig,
    allowed_classes: tuple[str, ...],
    device: str,
):
    segments = segment_characters(audio, sample_rate, segment_config)
    primary_envs = extract_segment_envelopes(audio, sample_rate, segments, envelope)
    aux_envs = extract_segment_envelopes(audio, sample_rate, segments, aux_envelope)
    p_labels, p_scores = predict_envelopes(
        model,
        primary_envs,
        classes=classes,
        allowed_classes=allowed_classes,
        device=device,
    )
    a_labels, a_scores = predict_envelopes(
        aux_model,
        aux_envs,
        classes=classes,
        allowed_classes=allowed_classes,
        device=device,
    )
    predictions = []
    for segment, p_label, p_score, a_label, a_score in zip(
        segments,
        p_labels,
        p_scores,
        a_labels,
        a_scores,
    ):
        if float(a_score) > float(p_score):
            predictions.append((classes[int(a_label)], float(a_score), segment))
        else:
            predictions.append((classes[int(p_label)], float(p_score), segment))
    return predictions


def _summarize(results: list[dict], *, alignment_sample_limit: int = 80) -> dict:
    if not results:
        return {
            "windows": 0,
            "files": [],
            "cer": None,
            "accuracy": None,
            "complete_files": 0,
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
        }

    by_file: dict[str, list[dict]] = {}
    for item in results:
        by_file.setdefault(item["wav"], []).append(item)

    file_reports = []
    total_ref = 0
    total_dist = 0
    total_subs = 0
    total_ins = 0
    total_dels = 0
    for wav, items in by_file.items():
        ordered = sorted(items, key=lambda item: item["start_s"])
        reference = ordered[0]["reference"]
        decoded = " ".join(item["decoded"] for item in ordered)
        coverage_start = min(item["start_s"] for item in ordered)
        coverage_end = max(item["start_s"] + item["duration_s"] for item in ordered)
        audio_duration = max(item["audio_total_duration_s"] for item in ordered)
        coverage_complete = coverage_start <= 0.1 and coverage_end >= audio_duration - 0.1
        if not coverage_complete:
            file_reports.append(
                {
                    "wav": wav,
                    "cer": None,
                    "accuracy": None,
                    "edit_distance": None,
                    "substitutions": None,
                    "insertions": None,
                    "deletions": None,
                    "reference_length": len(reference.replace(" ", "")),
                    "hypothesis_length": len(decoded.replace(" ", "")),
                    "decoded": decoded,
                    "coverage_complete": False,
                    "coverage_start_s": coverage_start,
                    "coverage_end_s": coverage_end,
                    "audio_duration_s": audio_duration,
                }
            )
            continue

        comparison = compare_text(reference, decoded)
        alignment = alignment_summary(
            reference,
            decoded,
            sample_limit=alignment_sample_limit,
        )
        file_reports.append(
            {
                "wav": wav,
                "cer": comparison.cer,
                "accuracy": comparison.accuracy,
                "edit_distance": comparison.distance,
                "substitutions": comparison.substitutions,
                "insertions": comparison.insertions,
                "deletions": comparison.deletions,
                "reference_length": comparison.reference_length,
                "hypothesis_length": comparison.hypothesis_length,
                "decoded": decoded,
                "coverage_complete": True,
                "coverage_start_s": coverage_start,
                "coverage_end_s": coverage_end,
                "audio_duration_s": audio_duration,
                "alignment": alignment,
            }
        )
        total_ref += comparison.reference_length
        total_dist += comparison.distance
        total_subs += comparison.substitutions
        total_ins += comparison.insertions
        total_dels += comparison.deletions

    cer = None if total_ref == 0 else total_dist / total_ref
    return {
        "windows": len(results),
        "files": file_reports,
        "cer": cer,
        "accuracy": None if cer is None else max(0.0, 1.0 - cer),
        "complete_files": sum(1 for item in file_reports if item["coverage_complete"]),
        "total_reference_length": total_ref,
        "total_edit_distance": total_dist,
        "substitutions": total_subs,
        "insertions": total_ins,
        "deletions": total_dels,
    }


def _metadata(args: argparse.Namespace, checkpoint_args) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": sys.argv,
        "args": vars(args),
        "checkpoint_args": checkpoint_args,
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    main()
