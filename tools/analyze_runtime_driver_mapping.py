import argparse
import csv
import re
from collections import Counter
from pathlib import Path


DEFAULT_LOG = Path("build/bin/test_vectoradd_callback_trace.log")
DEFAULT_DETAIL = Path("build/bin/runtime_driver_mapping.csv")
DEFAULT_TEXT_SUMMARY = Path("build/bin/runtime_driver_mapping_summary.txt")

LOG_PATTERN = re.compile(
    r"^\[CUPTI\]\s+"
    r"domain=(?P<domain>\S+)\s+"
    r"callbacksite=(?P<site>\S+)\s+"
    r"function=(?P<function>.*?)\s+"
    r"start_time=(?P<start>\d+)\s+"
    r"end_time=(?P<end>\d+)\s+"
    r"correlation_id=(?P<correlation>\d+)\s*$"
)


def parse_log(path):
    events = []
    skipped = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            match = LOG_PATTERN.match(line.strip())
            if not match:
                skipped += 1
                continue
            event = match.groupdict()
            event["line_no"] = line_no
            event["start"] = int(event["start"])
            event["end"] = int(event["end"])
            event["correlation"] = int(event["correlation"])
            events.append(event)
    return events, skipped


def build_spans(events):
    spans = []
    for index, event in enumerate(events, 1):
        if event["site"] != "EXIT":
            continue
        if event["end"] <= 0:
            continue
        span = {
            "seq": len(spans) + 1,
            "domain": event["domain"],
            "function": event["function"],
            "correlation": event["correlation"],
            "start": event["start"],
            "end": event["end"],
            "duration_ns": event["end"] - event["start"],
            "line_no": event["line_no"],
            "event_index": index,
        }
        spans.append(span)
    spans.sort(key=lambda item: (item["start"], item["end"], item["seq"]))
    return spans


def overlap_ns(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def classify_driver_for_runtime(runtime, driver):
    if runtime["start"] <= driver["start"] and driver["end"] <= runtime["end"]:
        if runtime["correlation"] == driver["correlation"]:
            return "nested_same_correlation", "high"
        return "nested_time_window", "medium"

    overlap = overlap_ns(runtime["start"], runtime["end"], driver["start"], driver["end"])
    if overlap <= 0:
        return None, None

    driver_ratio = overlap / max(1, driver["duration_ns"])
    runtime_ratio = overlap / max(1, runtime["duration_ns"])
    if runtime["correlation"] == driver["correlation"] and driver_ratio >= 0.5:
        return "overlap_same_correlation", "medium"
    if driver_ratio >= 0.8 or runtime_ratio >= 0.8:
        return "overlap_time_window", "low"
    return None, None


def find_runtime_mappings(runtime_spans, driver_spans):
    detail_rows = []
    runtime_rows = []
    mapped_driver_seq = set()

    for runtime in runtime_spans:
        candidates = []
        for driver in driver_spans:
            mapping_type, confidence = classify_driver_for_runtime(runtime, driver)
            if mapping_type is None:
                continue
            candidates.append((driver, mapping_type, confidence))

        candidates.sort(key=lambda item: (item[0]["start"], item[0]["end"], item[0]["seq"]))

        if not candidates:
            detail_rows.append(make_detail_row(runtime, None, "", "no_driver_api", "none", 0))
        else:
            for order, (driver, mapping_type, confidence) in enumerate(candidates, 1):
                mapped_driver_seq.add(driver["seq"])
                detail_rows.append(
                    make_detail_row(runtime, driver, order, mapping_type, confidence, len(candidates))
                )

        runtime_rows.append(make_runtime_row(runtime, candidates))

    unmapped_driver_rows = []
    for driver in driver_spans:
        if driver["seq"] not in mapped_driver_seq:
            unmapped_driver_rows.append(
                {
                    "driver_seq": driver["seq"],
                    "driver_function": driver["function"],
                    "driver_correlation_id": driver["correlation"],
                    "driver_start_ns": driver["start"],
                    "driver_end_ns": driver["end"],
                    "driver_duration_us": ns_to_us(driver["duration_ns"]),
                    "line_no": driver["line_no"],
                }
            )

    return detail_rows, runtime_rows, unmapped_driver_rows


def ns_to_us(value):
    return round(value / 1000.0, 3)


def make_detail_row(runtime, driver, order, mapping_type, confidence, driver_count):
    row = {
        "runtime_seq": runtime["seq"],
        "runtime_function": runtime["function"],
        "runtime_correlation_id": runtime["correlation"],
        "runtime_start_ns": runtime["start"],
        "runtime_end_ns": runtime["end"],
        "runtime_duration_us": ns_to_us(runtime["duration_ns"]),
        "driver_count_for_runtime": driver_count,
        "driver_order": order,
        "driver_seq": "",
        "driver_function": "",
        "driver_correlation_id": "",
        "driver_start_ns": "",
        "driver_end_ns": "",
        "driver_duration_us": "",
        "mapping_type": mapping_type,
        "mapping_confidence": confidence,
        "runtime_line_no": runtime["line_no"],
        "driver_line_no": "",
    }

    if driver is not None:
        row.update(
            {
                "driver_seq": driver["seq"],
                "driver_function": driver["function"],
                "driver_correlation_id": driver["correlation"],
                "driver_start_ns": driver["start"],
                "driver_end_ns": driver["end"],
                "driver_duration_us": ns_to_us(driver["duration_ns"]),
                "driver_line_no": driver["line_no"],
            }
        )
    return row


def make_runtime_row(runtime, candidates):
    driver_functions = [driver["function"] for driver, _, _ in candidates]
    confidence_counts = Counter(confidence for _, _, confidence in candidates)
    mapping_counts = Counter(mapping_type for _, mapping_type, _ in candidates)
    return {
        "runtime_seq": runtime["seq"],
        "runtime_function": runtime["function"],
        "runtime_correlation_id": runtime["correlation"],
        "runtime_start_ns": runtime["start"],
        "runtime_end_ns": runtime["end"],
        "runtime_duration_us": ns_to_us(runtime["duration_ns"]),
        "driver_count": len(candidates),
        "driver_functions": ";".join(driver_functions),
        "mapping_types": ";".join(f"{key}:{value}" for key, value in mapping_counts.items()),
        "confidence_counts": ";".join(f"{key}:{value}" for key, value in confidence_counts.items()),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(detail_path, unmapped_path, detail_rows, unmapped_driver_rows):
    detail_fields = [
        "runtime_seq",
        "runtime_function",
        "runtime_correlation_id",
        "runtime_start_ns",
        "runtime_end_ns",
        "runtime_duration_us",
        "driver_count_for_runtime",
        "driver_order",
        "driver_seq",
        "driver_function",
        "driver_correlation_id",
        "driver_start_ns",
        "driver_end_ns",
        "driver_duration_us",
        "mapping_type",
        "mapping_confidence",
        "runtime_line_no",
        "driver_line_no",
    ]
    unmapped_fields = [
        "driver_seq",
        "driver_function",
        "driver_correlation_id",
        "driver_start_ns",
        "driver_end_ns",
        "driver_duration_us",
        "line_no",
    ]
    write_csv(detail_path, detail_rows, detail_fields)
    write_csv(unmapped_path, unmapped_driver_rows, unmapped_fields)


def write_text_summary(
    path,
    log_path,
    events,
    skipped,
    runtime_spans,
    driver_spans,
    detail_rows,
    runtime_rows,
    unmapped_driver_rows,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    detail_mapped = [row for row in detail_rows if row["driver_function"]]
    runtime_func_counts = Counter(row["runtime_function"] for row in runtime_rows)
    mapping_counts = Counter(row["mapping_type"] for row in detail_rows)
    pair_counts = Counter(
        (row["runtime_function"], row["driver_function"])
        for row in detail_mapped
    )
    runtime_to_driver_count = Counter(
        (row["runtime_function"], row["driver_count"])
        for row in runtime_rows
    )
    unmapped_driver_counts = Counter(row["driver_function"] for row in unmapped_driver_rows)

    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("Runtime API to Driver API mapping summary\n")
        f.write("========================================\n\n")
        f.write(f"Log file: {log_path}\n")
        f.write(f"Parsed CUPTI callback rows: {len(events)}\n")
        f.write(f"Non-CUPTI / skipped lines: {skipped}\n")
        f.write(f"Runtime API spans: {len(runtime_spans)}\n")
        f.write(f"Driver API spans: {len(driver_spans)}\n")
        f.write(f"Mapped runtime-driver rows: {len(detail_mapped)}\n")
        f.write(f"Unmapped driver spans: {len(unmapped_driver_rows)}\n\n")

        f.write("Mapping type counts:\n")
        for key, value in mapping_counts.most_common():
            f.write(f"  {key}: {value}\n")

        f.write("\nRuntime API counts:\n")
        for key, value in runtime_func_counts.most_common():
            f.write(f"  {key}: {value}\n")

        f.write("\nRuntime function -> driver-count distribution:\n")
        for (runtime_func, driver_count), value in runtime_to_driver_count.most_common():
            f.write(f"  {runtime_func} -> {driver_count} driver call(s): {value} runtime call(s)\n")

        f.write("\nTop runtime -> driver pairs:\n")
        for (runtime_func, driver_func), value in pair_counts.most_common(30):
            f.write(f"  {value}x  {runtime_func} -> {driver_func}\n")

        f.write("\nTop unmapped driver APIs:\n")
        for key, value in unmapped_driver_counts.most_common(30):
            f.write(f"  {value}x  {key}\n")

        f.write("\nInterpretation notes:\n")
        f.write("  nested_same_correlation is the strongest result: the driver API span is inside the runtime API span and uses the same CUPTI correlation ID.\n")
        f.write("  nested_time_window means the driver API is nested inside a runtime API span, but the correlation ID differs.\n")
        f.write("  overlap_* results are fallback time-window matches and should be treated as weaker evidence.\n")
        f.write("  unmapped driver spans are usually CUDA driver initialization or direct Driver API work outside any Runtime API span.\n")


def main():
    parser = argparse.ArgumentParser(description="Parse CUPTI callback logs into Runtime API -> Driver API mappings.")
    parser.add_argument("--input", default=str(DEFAULT_LOG), help="input callback log")
    parser.add_argument("--detail-csv", default=str(DEFAULT_DETAIL), help="runtime-driver detail CSV")
    parser.add_argument("--unmapped-driver-csv", default="build/bin/unmapped_driver_api.csv", help="unmapped driver API CSV")
    parser.add_argument("--summary", default=str(DEFAULT_TEXT_SUMMARY), help="text summary")
    args = parser.parse_args()

    log_path = Path(args.input)
    if not log_path.exists():
        raise SystemExit(f"log file not found: {log_path}")

    events, skipped = parse_log(log_path)
    spans = build_spans(events)
    runtime_spans = [span for span in spans if span["domain"] == "RUNTIME_API"]
    driver_spans = [span for span in spans if span["domain"] == "DRIVER_API"]

    detail_rows, runtime_rows, unmapped_driver_rows = find_runtime_mappings(runtime_spans, driver_spans)

    detail_path = Path(args.detail_csv)
    unmapped_path = Path(args.unmapped_driver_csv)
    text_summary_path = Path(args.summary)

    write_outputs(detail_path, unmapped_path, detail_rows, unmapped_driver_rows)
    write_text_summary(
        text_summary_path,
        log_path,
        events,
        skipped,
        runtime_spans,
        driver_spans,
        detail_rows,
        runtime_rows,
        unmapped_driver_rows,
    )

    print(f"Wrote {detail_path}")
    print(f"Wrote {unmapped_path}")
    print(f"Wrote {text_summary_path}")


if __name__ == "__main__":
    main()
