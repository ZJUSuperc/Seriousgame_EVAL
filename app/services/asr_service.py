import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


class AsrService:
    def __init__(self, config: dict):
        self.config = config
        self._root = Path(config.get("SENSEVOICE_ROOT", ""))
        self._ffmpeg_bin = str(config.get("FFMPEG_BIN", "ffmpeg"))
        self._model = None
        self._model_kwargs = None
        self._model_device = None
        self._lock = threading.Lock()

        self._tag_pattern = re.compile(r"<\|[^>]+\|>")
        self._punctuation = set(["。", "！", "？", "；", "，", ".", "!", "?", ";", ","])

    def transcribe(
        self,
        video_path: Path,
        segment_length_sec: int = 60,
        language: str = "zh",
        use_itn: bool = True,
        device: str = "cpu",
    ) -> dict:
        if not video_path.exists():
            raise FileNotFoundError(f"视频不存在: {video_path}")

        segment_length_sec = int(segment_length_sec) if segment_length_sec else 60
        segment_length_sec = max(10, segment_length_sec)

        temp_dir = Path(tempfile.mkdtemp(prefix="sg_asr_"))
        audio_path = temp_dir / f"{video_path.stem}_audio.wav"

        try:
            self._extract_audio_from_video(video_path, audio_path)
            duration = self._get_audio_duration(audio_path)

            segments = self._split_segments(duration, segment_length_sec)

            model, kwargs = self._get_model(device)
            model.eval()

            all_segments = []
            all_sentences = []

            for idx, (start_sec, end_sec) in enumerate(segments, 1):
                segment_audio = temp_dir / f"segment_{idx}.wav"
                ok = self._extract_audio_segment(audio_path, start_sec, end_sec, segment_audio)
                if not ok:
                    continue

                try:
                    result = self._transcribe_segment(
                        model,
                        kwargs,
                        segment_audio,
                        language=language,
                        use_itn=use_itn,
                    )
                finally:
                    if segment_audio.exists():
                        segment_audio.unlink()

                if result is None:
                    continue

                text, timestamps = result
                clean_text = self._remove_tags(text)
                if not clean_text.strip():
                    continue

                adjusted = self._adjust_timestamps(timestamps, start_sec)
                sentences = self._build_sentences(adjusted, fallback_text=clean_text, start_sec=start_sec, end_sec=end_sec)

                all_segments.append(
                    {
                        "segment_index": idx,
                        "start_sec": float(start_sec),
                        "end_sec": float(end_sec) if end_sec is not None else None,
                        "text": clean_text,
                        "sentences": sentences,
                    }
                )
                all_sentences.extend(sentences)

            full_text = " ".join([item["text"] for item in all_segments]).strip()

            return {
                "summary": {
                    "language": language,
                    "segment_length_sec": segment_length_sec,
                    "duration_sec": float(duration) if duration is not None else None,
                    "segments_total": len(all_segments),
                    "sentences_total": len(all_sentences),
                    "device": device,
                },
                "full_text": full_text,
                "segments": all_segments,
                "sentences": all_sentences,
            }
        finally:
            if audio_path.exists():
                audio_path.unlink()
            if temp_dir.exists():
                for item in temp_dir.iterdir():
                    try:
                        item.unlink()
                    except OSError:
                        pass
                try:
                    temp_dir.rmdir()
                except OSError:
                    pass

    def _get_model(self, device: str):
        device = str(device or "cpu")
        with self._lock:
            if self._model is not None and self._model_device == device:
                return self._model, self._model_kwargs

            root = self._root
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))

            from model import SenseVoiceSmall

            model, kwargs = SenseVoiceSmall.from_pretrained(model="iic/SenseVoiceSmall", device=device)
            self._model = model
            self._model_kwargs = kwargs
            self._model_device = device
            return model, kwargs

    def _extract_audio_from_video(self, video_path: Path, output_audio_path: Path) -> None:
        cmd = [
            self._ffmpeg_bin,
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            str(output_audio_path),
        ]
        self._run_ffmpeg(cmd)

    def _extract_audio_segment(
        self,
        audio_path: Path,
        start_sec: float,
        end_sec: float | None,
        output_path: Path,
    ) -> bool:
        cmd = [
            self._ffmpeg_bin,
            "-i",
            str(audio_path),
            "-ss",
            str(start_sec),
        ]
        if end_sec is not None:
            cmd.extend(["-to", str(end_sec)])
        cmd.extend([
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            str(output_path),
        ])
        try:
            self._run_ffmpeg(cmd)
        except RuntimeError:
            return False
        return True

    def _run_ffmpeg(self, cmd: list[str]) -> None:
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                encoding="utf-8",
                errors="ignore",
            )
        except FileNotFoundError as exc:
            raise RuntimeError("未找到 ffmpeg，请检查安装与 PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stderr or "ffmpeg 执行失败") from exc

    @staticmethod
    def _get_audio_duration(audio_path: Path) -> float | None:
        try:
            import wave

            with wave.open(str(audio_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate <= 0:
                    return None
                return frames / float(rate)
        except Exception:
            return None

    @staticmethod
    def _split_segments(duration: float | None, segment_length: int) -> list[tuple[float, float | None]]:
        if duration is None:
            return [(0.0, None)]
        segments = []
        start = 0.0
        while start < duration:
            end = min(start + segment_length, duration)
            segments.append((start, end))
            start = end
        return segments

    def _transcribe_segment(self, model, kwargs, audio_path: Path, language: str, use_itn: bool):
        try:
            res = model.inference(
                data_in=str(audio_path),
                language=language,
                use_itn=use_itn,
                ban_emo_unk=True,
                output_timestamp=True,
                **(kwargs or {}),
            )
        except Exception:
            return None

        results = res[0] if isinstance(res, tuple) else res
        if not results:
            return None
        result = results[0]
        text = result.get("text", "")
        timestamps = result.get("timestamp", [])
        return text, timestamps

    def _remove_tags(self, text: str) -> str:
        cleaned = self._tag_pattern.sub("", text or "")
        return cleaned.strip()

    @staticmethod
    def _adjust_timestamps(timestamps, offset_sec: float) -> list[dict]:
        adjusted = []
        if not timestamps:
            return adjusted
        for item in timestamps:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            char = item[0]
            try:
                start = float(item[1]) + offset_sec
                end = float(item[2]) + offset_sec
            except (TypeError, ValueError):
                continue
            adjusted.append({"char": char, "start_sec": start, "end_sec": end})
        return adjusted

    def _build_sentences(
        self,
        timestamps: list[dict],
        fallback_text: str,
        start_sec: float,
        end_sec: float | None,
    ) -> list[dict]:
        if not timestamps:
            return [
                {
                    "text": fallback_text,
                    "start_sec": float(start_sec),
                    "end_sec": float(end_sec) if end_sec is not None else float(start_sec),
                }
            ]

        sentences = []
        current_text = ""
        sentence_start = None
        sentence_end = None

        for ts in timestamps:
            char = ts.get("char", "")
            if sentence_start is None:
                sentence_start = ts.get("start_sec")
            current_text += char
            sentence_end = ts.get("end_sec")

            if char in self._punctuation:
                if current_text.strip():
                    sentences.append(
                        {
                            "text": current_text.strip(),
                            "start_sec": float(sentence_start) if sentence_start is not None else float(start_sec),
                            "end_sec": float(sentence_end) if sentence_end is not None else float(start_sec),
                        }
                    )
                current_text = ""
                sentence_start = None
                sentence_end = None

        if current_text.strip():
            sentences.append(
                {
                    "text": current_text.strip(),
                    "start_sec": float(sentence_start) if sentence_start is not None else float(start_sec),
                    "end_sec": float(sentence_end) if sentence_end is not None else float(start_sec),
                }
            )

        return sentences
