from __future__ import annotations

import argparse
import csv
import gc
import shutil
from datetime import datetime
from pathlib import Path

from app.config import load_config
from app.services.pipeline_service import PipelineService
from app.services.task_store import TaskStore
from app.utils.json_io import write_json
from batch_process_videos import cleanup_task_result_root, scan_and_pair_videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="后台批量分析普通视频，按人员分类保存 OpenFace CSV 与评估报表")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(r"C:\Users\Administrator\Desktop\video_analyse\new"),
        help="待分析视频根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\Administrator\Desktop\seriousgame_eval\data\reports"),
        help="最终输出目录，按人员编号分类保存",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path(r"C:\Users\Administrator\Desktop\seriousgame_eval\data\_runtime_backend"),
        help="运行期临时目录，任务结束后会尽量清理",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto_main",
        choices=["auto_main", "multi", "single"],
        help="OpenFace 模式",
    )
    parser.add_argument(
        "--macro-stride",
        type=int,
        default=5,
        help="宏表情帧采样步长",
    )
    parser.add_argument(
        "--macro-interval-sec",
        type=float,
        default=None,
        help="宏表情按秒采样，设置后优先生效",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅处理前 N 个普通视频，0 表示全部",
    )
    parser.add_argument(
        "--person-id",
        type=str,
        default="",
        help="仅处理指定人员编号，例如 1907",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标目录已有结果则覆盖，默认跳过",
    )
    return parser.parse_args()


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def release_gpu_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def build_runtime_config(runtime_dir: Path) -> dict:
    config = dict(load_config())
    config["DATA_DIR"] = str(runtime_dir)
    config["UPLOAD_DIR"] = str(runtime_dir / "uploads")
    config["RESULT_DIR"] = str(runtime_dir / "results")
    config["TASK_DIR"] = str(runtime_dir / "tasks")
    config["REPORT_DIR"] = str(runtime_dir / "reports")
    ensure_dirs(
        Path(config["DATA_DIR"]),
        Path(config["UPLOAD_DIR"]),
        Path(config["RESULT_DIR"]),
        Path(config["TASK_DIR"]),
        Path(config["REPORT_DIR"]),
    )
    return config


def cleanup_runtime_artifacts(task_store: TaskStore, task_id: str, task_result: dict, config: dict) -> None:
    cleanup_task_result_root(task_result=task_result, config=config)

    artifacts = task_result.get("artifacts") if isinstance(task_result.get("artifacts"), dict) else {}
    report_dir_value = str(artifacts.get("analysis_report_dir") or "").strip()
    if report_dir_value:
        report_dir = Path(report_dir_value)
        if report_dir.exists() and report_dir.is_dir():
            shutil.rmtree(report_dir, ignore_errors=True)

    task_path = Path(config["TASK_DIR"]) / f"{task_id}.json"
    if task_path.exists():
        task_path.unlink(missing_ok=True)


def copy_outputs(artifacts: dict, destination_dir: Path, stem: str) -> dict:
    ensure_dirs(destination_dir)

    openface_csv_src = Path(str(artifacts.get("openface_csv") or "")).resolve()
    report_json_src = Path(str(artifacts.get("analysis_report_json") or "")).resolve()
    report_md_src = Path(str(artifacts.get("analysis_report_md") or "")).resolve()

    copied = {
        "openface_csv": None,
        "report_json": None,
        "report_md": None,
    }

    if openface_csv_src.exists() and openface_csv_src.is_file():
        dst = destination_dir / f"{stem}.csv"
        shutil.copy2(openface_csv_src, dst)
        copied["openface_csv"] = str(dst)

    if report_json_src.exists() and report_json_src.is_file():
        dst = destination_dir / f"{stem}.json"
        shutil.copy2(report_json_src, dst)
        copied["report_json"] = str(dst)

    if report_md_src.exists() and report_md_src.is_file():
        dst = destination_dir / f"{stem}.md"
        shutil.copy2(report_md_src, dst)
        copied["report_md"] = str(dst)

    return copied


def should_skip_output(destination_dir: Path, stem: str) -> bool:
    return (
        (destination_dir / f"{stem}.csv").exists()
        and (destination_dir / f"{stem}.json").exists()
        and (destination_dir / f"{stem}.md").exists()
    )


def collect_pairs(source_dir: Path, person_id: str) -> list[dict]:
    scan_result = scan_and_pair_videos(source_dir=source_dir, include_fix=False)
    pairs = list(scan_result.get("pairs") or [])
    if person_id:
        pairs = [pair for pair in pairs if str(pair.get("person_id") or "") == person_id]
    return pairs


def write_summary(output_dir: Path, rows: list[dict]) -> None:
    summary_json_path = output_dir / "batch_backend_summary.json"
    summary_csv_path = output_dir / "batch_backend_summary.csv"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "items": rows,
    }
    write_json(summary_json_path, payload)

    fieldnames = [
        "status",
        "person_id",
        "video_name",
        "video_path",
        "output_dir",
        "openface_csv",
        "report_json",
        "report_md",
        "task_id",
        "error",
    ]
    with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> int:
    args = parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    runtime_dir = args.runtime_dir.resolve()

    if not source_dir.exists() or not source_dir.is_dir():
        print(f"[ERROR] 源目录不存在: {source_dir}")
        return 1

    ensure_dirs(output_dir, runtime_dir)

    config = build_runtime_config(runtime_dir)
    task_store = TaskStore(Path(config["TASK_DIR"]))
    pipeline_service = PipelineService(config=config, task_store=task_store)

    pairs = collect_pairs(source_dir=source_dir, person_id=str(args.person_id or "").strip())
    if args.limit and args.limit > 0:
        pairs = pairs[: int(args.limit)]

    print(f"[INFO] 源目录: {source_dir}")
    print(f"[INFO] 输出目录: {output_dir}")
    print(f"[INFO] 临时目录: {runtime_dir}")
    print(f"[INFO] 待处理普通视频数量: {len(pairs)}")

    rows: list[dict] = []
    total = len(pairs)
    for index, pair in enumerate(pairs, start=1):
        video_path = Path(str(pair.get("video_path") or "")).resolve()
        video_name = str(pair.get("video_name") or video_path.name)
        stem = video_path.stem.replace(" ", "_")
        person = str(pair.get("person_id") or "unknown")
        destination_dir = output_dir / person / stem

        if not args.overwrite and should_skip_output(destination_dir, stem):
            print(f"[SKIP] ({index}/{total}) 已存在: {video_name}")
            rows.append(
                {
                    "status": "skipped_existing",
                    "person_id": person,
                    "video_name": video_name,
                    "video_path": str(video_path),
                    "output_dir": str(destination_dir),
                    "openface_csv": str(destination_dir / f"{stem}.csv"),
                    "report_json": str(destination_dir / f"{stem}.json"),
                    "report_md": str(destination_dir / f"{stem}.md"),
                    "task_id": None,
                    "error": None,
                }
            )
            continue

        if args.overwrite and destination_dir.exists():
            shutil.rmtree(destination_dir, ignore_errors=True)

        payload = {
            "video_path": str(video_path),
            "video_name": video_name,
            "mode": str(args.mode),
            "frame_stride": 1,
            "max_frames": 0,
            "macro_stride": max(1, int(args.macro_stride)),
            "macro_interval_sec": float(args.macro_interval_sec) if args.macro_interval_sec else None,
            "game_video_path": None,
            "game_video_name": None,
            "game_ocr_interval_sec": None,
        }

        task = task_store.create_task(payload)
        task_id = str(task.get("task_id"))
        print(f"[INFO] ({index}/{total}) 开始: {video_name} | task={task_id}")

        try:
            pipeline_service._run_task(task_id)
            task_result = task_store.get_task(task_id) or {}
            status = str(task_result.get("status") or "unknown")
            artifacts = task_result.get("artifacts") if isinstance(task_result.get("artifacts"), dict) else {}

            if status != "done":
                error = str(task_result.get("error") or "分析失败").strip() or "分析失败"
                rows.append(
                    {
                        "status": status,
                        "person_id": person,
                        "video_name": video_name,
                        "video_path": str(video_path),
                        "output_dir": str(destination_dir),
                        "openface_csv": None,
                        "report_json": None,
                        "report_md": None,
                        "task_id": task_id,
                        "error": error,
                    }
                )
                print(f"[WARN] 失败: {video_name} | {error}")
                cleanup_runtime_artifacts(task_store=task_store, task_id=task_id, task_result=task_result, config=config)
                release_gpu_cache()
                continue

            copied = copy_outputs(artifacts=artifacts, destination_dir=destination_dir, stem=stem)
            rows.append(
                {
                    "status": "done",
                    "person_id": person,
                    "video_name": video_name,
                    "video_path": str(video_path),
                    "output_dir": str(destination_dir),
                    "openface_csv": copied.get("openface_csv"),
                    "report_json": copied.get("report_json"),
                    "report_md": copied.get("report_md"),
                    "task_id": task_id,
                    "error": None,
                }
            )
            print(f"[OK] 完成: {video_name} -> {destination_dir}")
            cleanup_runtime_artifacts(task_store=task_store, task_id=task_id, task_result=task_result, config=config)
            release_gpu_cache()
        except Exception as exc:
            rows.append(
                {
                    "status": "failed_exception",
                    "person_id": person,
                    "video_name": video_name,
                    "video_path": str(video_path),
                    "output_dir": str(destination_dir),
                    "openface_csv": None,
                    "report_json": None,
                    "report_md": None,
                    "task_id": task_id,
                    "error": str(exc),
                }
            )
            print(f"[WARN] 异常: {video_name} | {exc}")
            task_result = task_store.get_task(task_id) or {}
            cleanup_runtime_artifacts(task_store=task_store, task_id=task_id, task_result=task_result, config=config)
            release_gpu_cache()

    write_summary(output_dir=output_dir, rows=rows)

    runtime_dirs = [runtime_dir / name for name in ("results", "reports", "tasks", "uploads")]
    for path in runtime_dirs:
        if path.exists() and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)

    done_count = sum(1 for row in rows if row.get("status") == "done")
    print(f"[INFO] 完成，成功 {done_count} / {len(rows)}")
    print(f"[INFO] 汇总文件: {output_dir / 'batch_backend_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
