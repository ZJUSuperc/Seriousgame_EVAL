from __future__ import annotations

import argparse
import gc
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.config import load_config
from app.services.pipeline_service import PipelineService
from app.services.task_store import TaskStore
from app.utils.json_io import read_json, write_json


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".webm",
    ".m4v",
    ".flv",
    ".mpeg",
    ".mpg",
}

NAME_PATTERN = re.compile(r"^(?P<person>\d+)_(?P<date>\d{8})(?P<tail>.*)$")
AU_PATTERN = re.compile(r"^AU\d{2}_[rc]$", re.IGNORECASE)


@dataclass(frozen=True)
class VideoItem:
    path: Path
    person_id: str
    date_code: str
    suffix: str
    key: str
    is_screen: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行视频分析并产出综合报表（串行执行）")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(r"C:\Users\Administrator\Desktop\video_analyse\new"),
        help="视频样例根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="批处理输出目录（默认 data/batch_reports/<时间戳>）",
    )
    parser.add_argument(
        "--macro-stride",
        type=int,
        default=5,
        help="宏表情帧采样步长（默认 5）",
    )
    parser.add_argument(
        "--macro-interval-sec",
        type=float,
        default=None,
        help="宏表情按秒采样，设置后优先生效",
    )
    parser.add_argument(
        "--game-ocr-interval-sec",
        type=float,
        default=None,
        help="游戏 OCR 采样间隔（秒），默认读取配置",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto_main",
        choices=["auto_main", "multi", "single"],
        help="OpenFace 模式",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅处理前 N 个主视频（0 表示全部）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅输出匹配清单，不执行分析",
    )
    parser.add_argument(
        "--include-fix",
        action="store_true",
        help="包含文件名含 fix 的视频（默认跳过）",
    )
    parser.add_argument(
        "--skip-processed",
        dest="skip_processed",
        action="store_true",
        default=True,
        help="根据 data/reports 已有报告自动跳过已处理视频（默认开启）",
    )
    parser.add_argument(
        "--no-skip-processed",
        dest="skip_processed",
        action="store_false",
        help="关闭自动跳过已处理视频",
    )
    parser.add_argument(
        "--no-ocr",
        "--video-only",
        dest="no_ocr",
        action="store_true",
        help="只做视频分析，跳过游戏画面 OCR 和后续游戏流程分析",
    )
    return parser.parse_args()


def normalize_tail(tail: str) -> str:
    return re.sub(r"\s+", " ", str(tail or "")).strip()


def make_key(person_id: str, date_code: str, suffix: str) -> str:
    return f"{person_id}_{date_code}" if not suffix else f"{person_id}_{date_code} {suffix}"


def parse_video_item(path: Path, include_fix: bool = False) -> tuple[VideoItem | None, str | None]:
    stem = path.stem
    lower_stem = stem.lower()

    if not include_fix and "fix" in lower_stem:
        return None, "skip_fix"

    match = NAME_PATTERN.match(stem)
    if not match:
        return None, "skip_name_unmatched"

    person_id = str(match.group("person") or "").strip()
    date_code = str(match.group("date") or "").strip()
    tail = str(match.group("tail") or "")

    is_screen = False
    if tail.lower().startswith("_screen"):
        is_screen = True
        tail = tail[len("_screen") :]

    suffix = normalize_tail(tail)
    key = make_key(person_id, date_code, suffix)

    return (
        VideoItem(
            path=path,
            person_id=person_id,
            date_code=date_code,
            suffix=suffix,
            key=key,
            is_screen=is_screen,
        ),
        None,
    )


def choose_screen(main_item: VideoItem, candidates: list[VideoItem]) -> VideoItem | None:
    if not candidates:
        return None

    main_suffix = main_item.path.suffix.lower()
    same_ext = [item for item in candidates if item.path.suffix.lower() == main_suffix]
    pool = same_ext if same_ext else candidates
    pool = sorted(pool, key=lambda item: str(item.path).lower())
    return pool[0] if pool else None


def scan_and_pair_videos(source_dir: Path, include_fix: bool = False) -> dict:
    all_video_files = []
    for file_path in sorted(source_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        all_video_files.append(file_path)

    parsed_items: list[VideoItem] = []
    skipped_fix: list[str] = []
    skipped_name_unmatched: list[str] = []

    for video_path in all_video_files:
        item, reason = parse_video_item(video_path, include_fix=include_fix)
        if item is not None:
            parsed_items.append(item)
            continue

        if reason == "skip_fix":
            skipped_fix.append(str(video_path))
        elif reason == "skip_name_unmatched":
            skipped_name_unmatched.append(str(video_path))

    main_items = [item for item in parsed_items if not item.is_screen]
    screen_items = [item for item in parsed_items if item.is_screen]

    screen_by_key: dict[str, list[VideoItem]] = {}
    for item in screen_items:
        screen_by_key.setdefault(item.key, []).append(item)

    pair_list = []
    used_screen_paths: set[str] = set()

    for main_item in sorted(main_items, key=lambda item: str(item.path).lower()):
        candidates = screen_by_key.get(main_item.key, [])
        screen_item = choose_screen(main_item, candidates)
        screen_path = str(screen_item.path) if screen_item is not None else None
        if screen_item is not None:
            used_screen_paths.add(str(screen_item.path))

        pair_list.append(
            {
                "person_id": main_item.person_id,
                "date_code": main_item.date_code,
                "suffix": main_item.suffix,
                "key": main_item.key,
                "video_path": str(main_item.path),
                "video_name": main_item.path.name,
                "game_video_path": screen_path,
                "game_video_name": Path(screen_path).name if screen_path else None,
                "has_screen": bool(screen_path),
            }
        )

    orphan_screens = []
    for screen_item in sorted(screen_items, key=lambda item: str(item.path).lower()):
        if str(screen_item.path) not in used_screen_paths:
            orphan_screens.append(str(screen_item.path))

    return {
        "pairs": pair_list,
        "total_video_files": len(all_video_files),
        "total_parsed_files": len(parsed_items),
        "main_video_count": len(main_items),
        "screen_video_count": len(screen_items),
        "skipped_fix": skipped_fix,
        "skipped_name_unmatched": skipped_name_unmatched,
        "orphan_screens": orphan_screens,
    }


def ensure_config_dirs(config: dict) -> None:
    for key in ("DATA_DIR", "UPLOAD_DIR", "RESULT_DIR", "TASK_DIR", "REPORT_DIR"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)


def normalize_path_for_compare(path_text: str | None) -> str | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    try:
        return str(Path(text).resolve()).lower()
    except Exception:
        return str(Path(text)).replace("/", "\\").lower()


def collect_processed_from_reports(report_root: Path, include_fix: bool = False) -> dict:
    processed_video_paths: set[str] = set()
    processed_keys: set[str] = set()
    parsed_reports = 0
    skipped_reports = 0

    if not report_root.exists() or not report_root.is_dir():
        return {
            "processed_video_paths": processed_video_paths,
            "processed_keys": processed_keys,
            "parsed_reports": 0,
            "skipped_reports": 0,
        }

    for report_dir in sorted(report_root.iterdir(), key=lambda p: str(p).lower()):
        if not report_dir.is_dir():
            continue

        json_candidates = sorted(report_dir.glob("*.json"), key=lambda p: str(p).lower())
        if not json_candidates:
            skipped_reports += 1
            continue

        report_json_path = json_candidates[0]
        try:
            report_data = read_json(report_json_path)
        except Exception:
            skipped_reports += 1
            continue

        if not isinstance(report_data, dict):
            skipped_reports += 1
            continue

        parsed_reports += 1
        report_input = report_data.get("input") if isinstance(report_data.get("input"), dict) else {}

        video_path = str(report_input.get("video_path") or "").strip()
        video_name = str(report_input.get("video_name") or "").strip()

        video_path_key = normalize_path_for_compare(video_path)
        if video_path_key:
            processed_video_paths.add(video_path_key)

        # 兼容历史报告路径可能缺失时，退化到文件名解析 key。
        name_for_key = video_name or (Path(video_path).name if video_path else "")
        if name_for_key:
            item, _ = parse_video_item(Path(name_for_key), include_fix=include_fix)
            if item is not None and item.key:
                processed_keys.add(item.key)

    return {
        "processed_video_paths": processed_video_paths,
        "processed_keys": processed_keys,
        "parsed_reports": parsed_reports,
        "skipped_reports": skipped_reports,
    }


def safe_json_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        number = float(value)
        if math.isfinite(number):
            return number
        return None
    except Exception:
        return str(value)


def append_openface_au_gaze_jsonl(
    task_id: str,
    pair: dict,
    artifacts: dict,
    output_jsonl_path: Path,
    au_columns_union: set[str],
    gaze_columns_union: set[str],
) -> tuple[int, str | None]:
    openface_csv_path = Path(str(artifacts.get("openface_csv") or "")).resolve()
    frame_analysis_path = Path(str(artifacts.get("frame_analysis_json") or "")).resolve()

    if not openface_csv_path.exists() or not openface_csv_path.is_file():
        return 0, "缺少 openface_csv"
    if not frame_analysis_path.exists() or not frame_analysis_path.is_file():
        return 0, "缺少 frame_analysis_json"

    try:
        frame_analysis = read_json(frame_analysis_path)
    except Exception as exc:
        return 0, f"读取 frame_analysis_json 失败: {exc}"

    if not isinstance(frame_analysis, list):
        return 0, "frame_analysis_json 格式异常"

    meta_by_openface_row: dict[int, dict] = {}
    for row in frame_analysis:
        if not isinstance(row, dict):
            continue
        raw_idx = row.get("openface_row")
        try:
            openface_row = int(raw_idx)
        except Exception:
            continue
        if openface_row in meta_by_openface_row:
            continue
        meta_by_openface_row[openface_row] = {
            "analysis_index": safe_json_value(row.get("index")),
            "frame": safe_json_value(row.get("frame")),
            "face_id": safe_json_value(row.get("face_id")),
            "timestamp": safe_json_value(row.get("timestamp")),
            "timestamp_source": str(row.get("timestamp_source") or "").strip() or None,
            "frame_index": safe_json_value(row.get("frame_index")),
            "macro_inference": bool(row.get("macro_inference")),
        }

    if not meta_by_openface_row:
        return 0, "未找到可映射的 openface_row"

    try:
        df = pd.read_csv(openface_csv_path)
    except Exception as exc:
        return 0, f"读取 OpenFace CSV 失败: {exc}"

    df.columns = [str(col).strip() for col in df.columns]
    df["openface_row"] = list(range(len(df)))
    df = df[df["openface_row"].isin(meta_by_openface_row.keys())].copy()
    if df.empty:
        return 0, "OpenFace CSV 与 frame_analysis 未匹配到行"

    df.sort_values(by=["openface_row"], inplace=True)

    au_cols = sorted([col for col in df.columns if AU_PATTERN.match(str(col))])
    gaze_cols = sorted([col for col in df.columns if str(col).startswith("gaze_")])
    au_columns_union.update(au_cols)
    gaze_columns_union.update(gaze_cols)

    rows_written = 0
    with output_jsonl_path.open("a", encoding="utf-8") as f:
        for _, row in df.iterrows():
            openface_row = int(row.get("openface_row"))
            meta = meta_by_openface_row.get(openface_row, {})

            record = {
                "task_id": task_id,
                "person_id": pair.get("person_id"),
                "key": pair.get("key"),
                "video_name": pair.get("video_name"),
                "video_path": pair.get("video_path"),
                "game_video_name": pair.get("game_video_name"),
                "game_video_path": pair.get("game_video_path"),
                "openface_csv": str(openface_csv_path),
                "openface_row": openface_row,
                "frame": safe_json_value(row.get("frame")),
                "timestamp": safe_json_value(row.get("timestamp")),
                "face_id": safe_json_value(row.get("face_id")),
                "success": safe_json_value(row.get("success")),
                "confidence": safe_json_value(row.get("confidence")),
                "analysis_meta": meta,
                "au": {col: safe_json_value(row.get(col)) for col in au_cols},
                "gaze": {col: safe_json_value(row.get(col)) for col in gaze_cols},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows_written += 1

    return rows_written, None


def release_gpu_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def sanitize_file_stem(raw_text: str, fallback: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        text = str(fallback)
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("._ ")
    return text or str(fallback)


def copy_report_files(
    pair: dict,
    task_id: str,
    artifacts: dict,
    report_collect_dir: Path,
) -> dict:
    report_collect_dir.mkdir(parents=True, exist_ok=True)

    report_json_src = Path(str(artifacts.get("analysis_report_json") or "")).resolve()
    report_md_src = Path(str(artifacts.get("analysis_report_md") or "")).resolve()

    key_safe = sanitize_file_stem(str(pair.get("key") or task_id), fallback=task_id[:8])
    person_safe = sanitize_file_stem(str(pair.get("person_id") or "unknown"), fallback="unknown")
    prefix = f"{person_safe}__{key_safe}__{task_id[:8]}"

    copied_json = None
    copied_md = None

    if report_json_src.exists() and report_json_src.is_file():
        copied_json_path = report_collect_dir / f"{prefix}.json"
        shutil.copy2(report_json_src, copied_json_path)
        copied_json = str(copied_json_path)

    if report_md_src.exists() and report_md_src.is_file():
        copied_md_path = report_collect_dir / f"{prefix}.md"
        shutil.copy2(report_md_src, copied_md_path)
        copied_md = str(copied_md_path)

    return {
        "analysis_report_json": str(report_json_src) if report_json_src.exists() else None,
        "analysis_report_md": str(report_md_src) if report_md_src.exists() else None,
        "copied_report_json": copied_json,
        "copied_report_md": copied_md,
    }


def cleanup_task_result_root(task_result: dict, config: dict) -> tuple[bool, str | None]:
    artifacts = task_result.get("artifacts") if isinstance(task_result.get("artifacts"), dict) else {}
    result_root_raw = str(artifacts.get("result_root") or "").strip()
    if not result_root_raw:
        return False, "缺少 result_root"

    try:
        result_root = Path(result_root_raw).resolve()
    except Exception as exc:
        return False, f"result_root 非法: {exc}"

    result_dir_raw = str(config.get("RESULT_DIR") or "").strip()
    if not result_dir_raw:
        return False, "配置缺少 RESULT_DIR"

    try:
        result_dir = Path(result_dir_raw).resolve()
    except Exception as exc:
        return False, f"RESULT_DIR 非法: {exc}"

    # 安全防护：仅允许删除 RESULT_DIR 下的任务目录，避免误删。
    if result_root == result_dir or result_dir not in result_root.parents:
        return False, f"跳过删除（不在 RESULT_DIR 下）: {result_root}"

    if not result_root.exists():
        return False, "result_root 不存在"
    if not result_root.is_dir():
        return False, "result_root 不是目录"

    try:
        shutil.rmtree(result_root)
        return True, None
    except Exception as exc:
        return False, f"删除失败: {exc}"


def build_summary_markdown(summary: dict) -> str:
    counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
    success_items = summary.get("success_items") if isinstance(summary.get("success_items"), list) else []
    failed_items = summary.get("failed_items") if isinstance(summary.get("failed_items"), list) else []
    skipped = summary.get("scan") if isinstance(summary.get("scan"), dict) else {}

    lines = [
        "# 批量分析汇总",
        "",
        f"- 开始时间: {summary.get('started_at') or '-'}",
        f"- 结束时间: {summary.get('finished_at') or '-'}",
        f"- 源目录: {summary.get('source_dir') or '-'}",
        f"- 批处理目录: {summary.get('batch_dir') or '-'}",
        f"- 综合报表复制目录: {summary.get('collected_report_dir') or '-'}",
        f"- OpenFace AU+视线汇总: {summary.get('openface_au_gaze_jsonl') or '-'}",
        "",
        "## 数量统计",
        f"- 扫描到视频文件: {counts.get('total_video_files', 0)}",
        f"- 主视频数量: {counts.get('main_video_count', 0)}",
        f"- 屏幕视频数量: {counts.get('screen_video_count', 0)}",
        f"- 实际处理数量: {counts.get('processed_total', 0)}",
        f"- 成功: {counts.get('success', 0)}",
        f"- 失败: {counts.get('failed', 0)}",
        f"- 已处理自动跳过: {counts.get('skipped_processed', 0)}",
        f"- 跳过 fix 文件: {len(skipped.get('skipped_fix', []))}",
        f"- 命名不匹配文件: {len(skipped.get('skipped_name_unmatched', []))}",
        f"- 未匹配到主视频的 screen 文件: {len(skipped.get('orphan_screens', []))}",
        "",
        "## 成功条目",
    ]

    if success_items:
        for item in success_items:
            lines.append(
                (
                    f"- [{item.get('person_id')}] {item.get('video_name')}"
                    f" | screen={item.get('has_screen')}"
                    f" | task={item.get('task_id')}"
                    f" | report={item.get('copied_report_json') or item.get('analysis_report_json') or '-'}"
                    f" | openface_rows={item.get('openface_rows_written', 0)}"
                )
            )
    else:
        lines.append("- 无")

    lines.extend(["", "## 失败条目"])
    if failed_items:
        for item in failed_items:
            lines.append(
                (
                    f"- [{item.get('person_id')}] {item.get('video_name')}"
                    f" | screen={item.get('has_screen')}"
                    f" | task={item.get('task_id')}"
                    f" | status={item.get('status') or '-'}"
                    f" | stage={item.get('stage') or '-'}"
                    f" | error={item.get('error') or item.get('last_log') or '-'}"
                )
            )
    else:
        lines.append("- 无")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()

    source_dir = args.source_dir.resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"[ERROR] 源目录不存在: {source_dir}")
        return 1

    config = load_config()
    ensure_config_dirs(config)

    started_at = datetime.now()
    batch_ts = started_at.strftime("%Y%m%d_%H%M%S")
    default_output_dir = Path(config["DATA_DIR"]) / "batch_reports" / batch_ts
    batch_dir = (args.output_dir.resolve() if args.output_dir else default_output_dir.resolve())
    batch_dir.mkdir(parents=True, exist_ok=True)

    report_collect_dir = batch_dir / "reports"
    report_collect_dir.mkdir(parents=True, exist_ok=True)

    openface_jsonl_path = batch_dir / "openface_au_gaze_all.jsonl"
    summary_json_path = batch_dir / "batch_summary.json"
    summary_md_path = batch_dir / "batch_summary.md"
    manifest_json_path = batch_dir / "pair_manifest.json"
    report_index_csv_path = batch_dir / "report_index.csv"

    scan_result = scan_and_pair_videos(source_dir, include_fix=bool(args.include_fix))
    all_pairs = list(scan_result.get("pairs") or [])

    processed_info = {
        "processed_video_paths": set(),
        "processed_keys": set(),
        "parsed_reports": 0,
        "skipped_reports": 0,
    }
    skipped_processed_items = []
    if args.skip_processed:
        processed_info = collect_processed_from_reports(
            report_root=Path(config["REPORT_DIR"]).resolve(),
            include_fix=bool(args.include_fix),
        )

    processed_video_paths = processed_info.get("processed_video_paths") or set()
    processed_keys = processed_info.get("processed_keys") or set()

    pairs = []
    for pair in all_pairs:
        video_path_key = normalize_path_for_compare(str(pair.get("video_path") or ""))
        pair_key = str(pair.get("key") or "").strip()
        already_processed = bool(
            (video_path_key and video_path_key in processed_video_paths)
            or (pair_key and pair_key in processed_keys)
        )
        if already_processed:
            skipped_processed_items.append(
                {
                    "person_id": pair.get("person_id"),
                    "key": pair.get("key"),
                    "video_name": pair.get("video_name"),
                    "video_path": pair.get("video_path"),
                }
            )
            continue
        pairs.append(pair)

    if args.limit and args.limit > 0:
        pairs = pairs[: int(args.limit)]

    write_json(manifest_json_path, pairs)

    print(f"[INFO] 扫描目录: {source_dir}")
    print(f"[INFO] 主视频数量: {scan_result.get('main_video_count', 0)}")
    print(f"[INFO] 屏幕视频数量: {scan_result.get('screen_video_count', 0)}")
    print(
        "[INFO] 历史报告解析: "
        f"有效={processed_info.get('parsed_reports', 0)}"
        f" | 跳过异常报告={processed_info.get('skipped_reports', 0)}"
    )
    print(f"[INFO] 已处理自动跳过: {len(skipped_processed_items)}")
    print(f"[INFO] 待处理数量: {len(pairs)}")
    print(f"[INFO] 匹配清单: {manifest_json_path}")

    if args.dry_run:
        print("[INFO] dry-run 模式，未执行分析")
        return 0

    task_store = TaskStore(Path(config["TASK_DIR"]))
    pipeline_service = PipelineService(config=config, task_store=task_store)

    game_ocr_interval_default = args.game_ocr_interval_sec
    if args.no_ocr:
        game_ocr_interval_default = None
    if game_ocr_interval_default is None:
        try:
            game_ocr_interval_default = float(config.get("GAME_OCR_INTERVAL_SEC_DEFAULT", 2.0))
        except Exception:
            game_ocr_interval_default = 2.0

    success_items = []
    failed_items = []
    au_columns_union: set[str] = set()
    gaze_columns_union: set[str] = set()

    if openface_jsonl_path.exists():
        openface_jsonl_path.unlink()

    total_to_process = len(pairs)
    for index, pair in enumerate(pairs, start=1):
        video_name = str(pair.get("video_name") or "")
        print(f"\n[INFO] ({index}/{total_to_process}) 开始: {video_name}")

        payload = {
            "video_path": str(pair.get("video_path")),
            "video_name": str(pair.get("video_name")),
            "mode": str(args.mode),
            "frame_stride": 1,
            "max_frames": 0,
            "macro_stride": max(1, int(args.macro_stride)),
            "macro_interval_sec": float(args.macro_interval_sec) if args.macro_interval_sec else None,
            "game_video_path": None if args.no_ocr else (str(pair.get("game_video_path")) if pair.get("game_video_path") else None),
            "game_video_name": None if args.no_ocr else (str(pair.get("game_video_name")) if pair.get("game_video_name") else None),
            "game_ocr_interval_sec": None if args.no_ocr else float(game_ocr_interval_default),
        }

        task = task_store.create_task(payload)
        task_id = str(task.get("task_id"))
        run_started_at = datetime.now()

        try:
            pipeline_service._run_task(task_id)
        except Exception as exc:
            failed_items.append(
                {
                    "person_id": pair.get("person_id"),
                    "key": pair.get("key"),
                    "video_name": pair.get("video_name"),
                    "video_path": pair.get("video_path"),
                    "has_screen": bool(pair.get("game_video_path")),
                    "task_id": task_id,
                    "status": "failed",
                    "stage": "exception",
                    "error": f"脚本层异常: {exc}",
                    "last_log": None,
                    "runtime_sec": (datetime.now() - run_started_at).total_seconds(),
                }
            )
            release_gpu_cache()
            print(f"[WARN] 分析异常，已跳过: {exc}")
            continue

        task_result = task_store.get_task(task_id) or {}
        artifacts = task_result.get("artifacts") if isinstance(task_result.get("artifacts"), dict) else {}
        status = str(task_result.get("status") or "unknown")
        stage = str(task_result.get("stage") or "unknown")
        error = task_result.get("error")
        logs = task_result.get("logs") if isinstance(task_result.get("logs"), list) else []
        last_log = None
        if logs:
            last_item = logs[-1]
            if isinstance(last_item, dict):
                last_log = str(last_item.get("message") or "").strip() or None

        if status == "done":
            report_paths = copy_report_files(pair=pair, task_id=task_id, artifacts=artifacts, report_collect_dir=report_collect_dir)
            openface_rows, openface_error = append_openface_au_gaze_jsonl(
                task_id=task_id,
                pair=pair,
                artifacts=artifacts,
                output_jsonl_path=openface_jsonl_path,
                au_columns_union=au_columns_union,
                gaze_columns_union=gaze_columns_union,
            )

            item = {
                "person_id": pair.get("person_id"),
                "key": pair.get("key"),
                "video_name": pair.get("video_name"),
                "video_path": pair.get("video_path"),
                "game_video_name": pair.get("game_video_name"),
                "game_video_path": pair.get("game_video_path"),
                "has_screen": bool(pair.get("game_video_path")),
                "task_id": task_id,
                "status": status,
                "stage": stage,
                "error": None,
                "last_log": last_log,
                "runtime_sec": (datetime.now() - run_started_at).total_seconds(),
                "result_root": artifacts.get("result_root"),
                "openface_csv": artifacts.get("openface_csv"),
                "frame_analysis_json": artifacts.get("frame_analysis_json"),
                "openface_rows_written": int(openface_rows),
                "openface_extract_error": openface_error,
            }

            cleaned, clean_note = cleanup_task_result_root(task_result=task_result, config=config)
            item["cleaned_result_root"] = bool(cleaned)
            item["clean_result_note"] = clean_note

            item.update(report_paths)
            success_items.append(item)
            print(
                (
                    f"[OK] 完成: task={task_id}"
                    f" | report={item.get('copied_report_json') or item.get('analysis_report_json') or '-'}"
                    f" | openface_rows={openface_rows}"
                    f" | cleaned={cleaned}"
                )
            )
        else:
            cleaned, clean_note = cleanup_task_result_root(task_result=task_result, config=config)
            failed_items.append(
                {
                    "person_id": pair.get("person_id"),
                    "key": pair.get("key"),
                    "video_name": pair.get("video_name"),
                    "video_path": pair.get("video_path"),
                    "game_video_name": pair.get("game_video_name"),
                    "game_video_path": pair.get("game_video_path"),
                    "has_screen": bool(pair.get("game_video_path")),
                    "task_id": task_id,
                    "status": status,
                    "stage": stage,
                    "error": str(error or "").strip() or None,
                    "last_log": last_log,
                    "runtime_sec": (datetime.now() - run_started_at).total_seconds(),
                    "cleaned_result_root": bool(cleaned),
                    "clean_result_note": clean_note,
                }
            )
            print(
                f"[WARN] 失败: task={task_id} | stage={stage} | error={error or last_log or '-'}"
                f" | cleaned={cleaned}"
            )

        release_gpu_cache()

    finished_at = datetime.now()

    report_index_rows = []
    for row in success_items:
        report_index_rows.append(
            {
                "status": "done",
                "person_id": row.get("person_id"),
                "key": row.get("key"),
                "video_name": row.get("video_name"),
                "video_path": row.get("video_path"),
                "game_video_name": row.get("game_video_name"),
                "game_video_path": row.get("game_video_path"),
                "task_id": row.get("task_id"),
                "analysis_report_json": row.get("analysis_report_json"),
                "analysis_report_md": row.get("analysis_report_md"),
                "copied_report_json": row.get("copied_report_json"),
                "copied_report_md": row.get("copied_report_md"),
                "openface_rows_written": row.get("openface_rows_written"),
                "error": row.get("openface_extract_error"),
            }
        )
    for row in failed_items:
        report_index_rows.append(
            {
                "status": "failed",
                "person_id": row.get("person_id"),
                "key": row.get("key"),
                "video_name": row.get("video_name"),
                "video_path": row.get("video_path"),
                "game_video_name": row.get("game_video_name"),
                "game_video_path": row.get("game_video_path"),
                "task_id": row.get("task_id"),
                "analysis_report_json": None,
                "analysis_report_md": None,
                "copied_report_json": None,
                "copied_report_md": None,
                "openface_rows_written": 0,
                "error": row.get("error") or row.get("last_log"),
            }
        )

    pd.DataFrame(report_index_rows).to_csv(report_index_csv_path, index=False, encoding="utf-8-sig")

    summary_payload = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "batch_dir": str(batch_dir),
        "collected_report_dir": str(report_collect_dir),
        "openface_au_gaze_jsonl": str(openface_jsonl_path),
        "pair_manifest_json": str(manifest_json_path),
        "report_index_csv": str(report_index_csv_path),
        "settings": {
            "mode": str(args.mode),
            "macro_stride": int(max(1, int(args.macro_stride))),
            "macro_interval_sec": float(args.macro_interval_sec) if args.macro_interval_sec else None,
            "game_ocr_interval_sec": None if args.no_ocr else float(game_ocr_interval_default),
            "dry_run": bool(args.dry_run),
            "limit": int(args.limit),
            "include_fix": bool(args.include_fix),
            "no_ocr": bool(args.no_ocr),
        },
        "scan": scan_result,
        "counts": {
            "total_video_files": int(scan_result.get("total_video_files", 0)),
            "main_video_count": int(scan_result.get("main_video_count", 0)),
            "screen_video_count": int(scan_result.get("screen_video_count", 0)),
            "processed_total": len(success_items) + len(failed_items),
            "success": len(success_items),
            "failed": len(failed_items),
            "skipped_processed": len(skipped_processed_items),
            "openface_au_columns": len(au_columns_union),
            "openface_gaze_columns": len(gaze_columns_union),
        },
        "resume": {
            "skip_processed_enabled": bool(args.skip_processed),
            "parsed_reports": int(processed_info.get("parsed_reports", 0)),
            "skipped_reports": int(processed_info.get("skipped_reports", 0)),
            "skipped_processed_items": skipped_processed_items,
        },
        "openface_columns": {
            "au_columns": sorted(au_columns_union),
            "gaze_columns": sorted(gaze_columns_union),
        },
        "success_items": success_items,
        "failed_items": failed_items,
    }

    write_json(summary_json_path, summary_payload)
    summary_md_path.write_text(build_summary_markdown(summary_payload), encoding="utf-8")

    print("\n[INFO] 批量分析完成")
    print(f"[INFO] 成功: {len(success_items)} | 失败: {len(failed_items)}")
    print(f"[INFO] 汇总 JSON: {summary_json_path}")
    print(f"[INFO] 汇总 Markdown: {summary_md_path}")
    print(f"[INFO] 报表索引 CSV: {report_index_csv_path}")
    print(f"[INFO] OpenFace AU+视线 JSONL: {openface_jsonl_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
