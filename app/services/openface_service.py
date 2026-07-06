import subprocess
from pathlib import Path


class OpenFaceService:
    def __init__(self, config: dict):
        self.config = config

    def run(self, video_path: Path, mode: str, output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)

        attempts = self._resolve_attempts(mode)
        failures = []

        for idx, attempt in enumerate(attempts, start=1):
            mode_name = attempt["mode"]
            exe_key = attempt["exe_key"]
            exe_path = Path(self.config.get(exe_key, ""))

            if not exe_path.exists():
                failures.append(f"{mode_name}: 可执行文件不存在 {exe_path}")
                continue

            attempt_dir = output_dir / f"attempt_{idx}_{mode_name}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                str(exe_path),
                "-f",
                str(video_path),
                "-out_dir",
                str(attempt_dir),
                "-2Dfp",
                "-3Dfp",
                "-pose",
                "-aus",
                "-gaze",
            ]

            result = subprocess.run(
                cmd,
                cwd=str(Path(self.config["OPENFACE_DIR"])),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                stdout = (result.stdout or "").strip()
                failures.append(f"{mode_name}: {stderr or stdout}")
                continue

            csv_files = sorted(attempt_dir.glob("*.csv"))
            if not csv_files:
                failures.append(f"{mode_name}: 未生成 CSV")
                continue

            return {
                "csv_path": str(csv_files[0]),
                "output_dir": str(attempt_dir),
                "cmd": cmd,
                "stdout_tail": (result.stdout or "")[-4000:],
                "mode_used": mode_name,
            }

        detail = " | ".join(failures) if failures else "未知错误"
        raise RuntimeError(f"OpenFace 执行失败: {detail}")

    @staticmethod
    def _resolve_attempts(mode: str) -> list[dict]:
        if mode == "single":
            return [{"mode": "single", "exe_key": "OPENFACE_SINGLE_EXE"}]
        if mode == "multi":
            return [{"mode": "multi", "exe_key": "OPENFACE_MULTI_EXE"}]
        return [
            {"mode": "multi", "exe_key": "OPENFACE_MULTI_EXE"},
            {"mode": "single", "exe_key": "OPENFACE_SINGLE_EXE"},
        ]
