from __future__ import annotations

import math
import re
from collections import Counter, defaultdict


class GameplayTimelineService:
    MODULE_NAMES = {
        "thinking": "思维苗圃",
        "lake": "静心湖畔",
        "attic": "回忆阁楼",
        "studio": "共创画室",
    }

    MODULE_ORDER = ("thinking", "lake", "attic", "studio")
    HOME_SCREEN_MARKERS = (
        "思维苗圃",
        "静心湖畔",
        "回忆阁楼",
        "回忆楼阁",
        "共创画室",
    )
    CONTENT_FORCED_ENTRY_WEIGHTS = {
        ("thinking", "tool"): 4,
        ("thinking", "reframe"): 4,
        ("thinking", "thought"): 4,
        ("attic", "photo"): 4,
        ("attic", "question"): 4,
    }
    MIN_SEGMENT_DURATION_SEC = 0.4
    MAX_TEXT_LINES_PER_SAMPLE = 24
    OCR_NOISE_URL_REGEX = re.compile(
        r"(https?://|www\.|localhost|127(?:\.\d{1,3}){3}|about:blank|edge://|chrome://|file://|[a-z0-9\-]+\.(?:com|cn|net|org|io|gov|edu)(?:[/:]|$))",
        re.IGNORECASE,
    )
    OCR_NOISE_RAW_MARKERS = (
        "search or type url",
        "搜索或输入网址",
        "新标签页",
        "地址栏",
        "about:blank",
        "microsoft edge",
        "google chrome",
        "mozilla firefox",
        "edge://",
        "chrome://",
        "localhost",
        "127.0.0.1",
    )
    OCR_NOISE_NORM_MARKERS = (
        "搜索或输入网址",
        "新标签页",
        "地址栏",
        "aboutblank",
        "microsoftedge",
        "googlechrome",
        "mozillafirefox",
        "localhost",
        "127001",
    )
    OUTPUT_KEYWORD_SOURCES = {
        "module_name",
        "strong",
        "thought",
        "tool",
        "period",
        "photo",
        "question",
        "studio_action",
        "breath_phase",
    }

    THINKING_DAILY_THOUGHTS = {
        1: ["我老了没用了", "孩子不来看我", "我什么都做不好"],
        2: ["我记忆力变差了", "我害怕孤独", "我是家人的负担"],
        3: ["我失去生活意义", "我担心身体", "我被社会遗忘"],
        4: ["我羡慕年轻人", "我害怕死亡", "我觉得自己没用"],
        5: ["我怀念过去", "我担心拖累子女", "我觉得没希望"],
        6: ["我害怕生病", "我跟不上时代", "我担心没人关心"],
        7: ["我害怕失去独立", "我拖累家人", "我担心未来"],
    }

    THINKING_THOUGHT_ALIASES = {
        "孩子不来看我": ["孩子们都不来看我", "孩子们不来看我", "孩子都不来看我", "孩子不来看望我"],
        "我是家人的负担": ["我是家里的负担", "我成了家人的负担", "我给家人添负担"],
        "我觉得自己没用": ["我觉得自己没有用", "我觉得我没用"],
        "我担心拖累子女": ["我担心拖累孩子", "我担心拖累儿女"],
        "我跟不上时代": ["我跟不上这个时代", "我跟不上社会"],
        "我担心没人关心": ["我担心没有人关心", "我担心没人会关心我"],
        "我害怕失去独立": ["我害怕失去独立能力", "我怕失去独立"],
        "我拖累家人": ["我拖累了家人", "我会拖累家人"],
    }

    THINKING_POSITIVE_THOUGHTS = {
        1: {
            "我老了没用了": {
                "浇水": "我虽然年纪大了，但经验丰富",
                "施肥": "年龄带来智慧，我可以分享经验",
                "除虫": "年龄不是限制，我可以继续成长",
            },
            "孩子不来看我": {
                "浇水": "孩子工作忙，我可以主动联系",
                "施肥": "我可以培养新的兴趣爱好",
                "除虫": "我可以主动创造相聚机会",
            },
            "我什么都做不好": {
                "浇水": "每个人都有长处，我可以慢慢学习",
                "施肥": "我可以专注擅长的事情",
                "除虫": "我可以从简单事情开始",
            },
        },
        2: {
            "我记忆力变差了": {
                "浇水": "我可以使用记忆技巧帮助自己",
                "施肥": "我可以记录重要的事情",
                "除虫": "我可以保持大脑活跃",
            },
            "我害怕孤独": {
                "浇水": "我可以培养新的社交圈子",
                "施肥": "我可以参加社区活动",
                "除虫": "我可以学会享受独处时光",
            },
            "我是家人的负担": {
                "浇水": "我的存在有价值，为家人带来温暖",
                "施肥": "我可以为家庭做贡献",
                "除虫": "我可以保持独立",
            },
        },
        3: {
            "我失去生活意义": {
                "浇水": "我可以重新发现生活美好",
                "施肥": "我可以设定新的生活目标",
                "除虫": "我可以帮助他人找到价值",
            },
            "我担心身体": {
                "浇水": "我可以通过健康生活方式照顾自己",
                "施肥": "我可以定期体检预防疾病",
                "除虫": "我可以保持积极心态",
            },
            "我被社会遗忘": {
                "浇水": "我可以主动参与社区活动",
                "施肥": "我可以学习新技能跟上时代",
                "除虫": "我可以成为志愿者",
            },
        },
        4: {
            "我羡慕年轻人": {
                "浇水": "我可以用自己方式保持活力",
                "施肥": "我可以学习年轻人新观念",
                "除虫": "我可以发挥自己优势",
            },
            "我害怕死亡": {
                "浇水": "我可以珍惜当下每一刻",
                "施肥": "我可以为家人留下美好回忆",
                "除虫": "我可以接受生命自然规律",
            },
            "我觉得自己没用": {
                "浇水": "我的经验和智慧很宝贵",
                "施肥": "我可以成为年轻人导师",
                "除虫": "我可以为社会传承文化",
            },
        },
        5: {
            "我怀念过去": {
                "浇水": "我可以创造新的美好回忆",
                "施肥": "我可以把过去经验运用现在",
                "除虫": "我可以活在当下珍惜时光",
            },
            "我担心拖累子女": {
                "浇水": "我可以保持独立接受适度帮助",
                "施肥": "我可以为子女减轻负担",
                "除虫": "我可以与子女建立更好关系",
            },
            "我觉得没希望": {
                "浇水": "希望就在身边，我可以找到快乐",
                "施肥": "我可以设定小目标逐步实现",
                "除虫": "我可以寻求帮助重新找到意义",
            },
        },
        6: {
            "我害怕生病": {
                "浇水": "我可以找到内心的力量",
                "施肥": "我可以练习冥想和放松",
                "除虫": "我可以培养内心的平静",
            },
            "我跟不上时代": {
                "浇水": "我可以学习新事物保持开放心态",
                "施肥": "我可以向年轻人学习新技术",
                "除虫": "我可以保持好奇心终身学习",
            },
            "我担心没人关心": {
                "浇水": "我可以主动关心他人，爱是相互的",
                "施肥": "我可以培养深厚友谊",
                "除虫": "我可以学会自爱不依赖他人",
            },
        },
        7: {
            "我害怕失去独立": {
                "浇水": "我可以通过锻炼保持身体机能",
                "施肥": "我可以学习新的生活技能",
                "除虫": "我可以保持积极心态延缓衰老",
            },
            "我拖累家人": {
                "浇水": "我的存在让家庭更完整",
                "施肥": "我可以为家庭创造价值",
                "除虫": "我可以与家人建立更好关系",
            },
            "我担心未来": {
                "浇水": "我可以活在当下珍惜每一刻",
                "施肥": "我可以为未来做好准备",
                "除虫": "我可以保持希望相信美好未来",
            },
        },
    }

    THINKING_TOOL_KEYWORDS = ("浇水", "施肥", "除虫")

    STUDIO_ACTION_KEYWORDS = (
        "想不好画什么",
        "想不到画什么",
        "画笔",
        "橡皮擦",
        "完成绘画",
        "下载绘画",
        "清空画布",
    )

    MEMORY_PERIODS = [
        {
            "period_index": 1,
            "period_name": "1950s 建国初期",
            "aliases": ["1950s", "建国初期", "1950年代", "建国初期历史照片"],
        },
        {
            "period_index": 2,
            "period_name": "1960s 激情岁月",
            "aliases": ["1960s", "激情岁月", "1960年代", "激情岁月历史照片"],
        },
        {
            "period_index": 3,
            "period_name": "1970s 转折年代",
            "aliases": ["1970s", "转折年代", "1970年代", "转折年代历史照片"],
        },
        {
            "period_index": 4,
            "period_name": "1980s 黄金时代",
            "aliases": ["1980s", "黄金时代", "1980年代", "黄金时代历史照片"],
        },
        {
            "period_index": 5,
            "period_name": "1990s 飞速发展",
            "aliases": ["1990s", "飞速发展", "1990年代", "飞速发展历史照片"],
        },
        {
            "period_index": 6,
            "period_name": "2000s 崭新纪元",
            "aliases": ["2000s", "崭新纪元", "2000年代", "崭新纪元历史照片"],
        },
        {
            "period_index": 7,
            "period_name": "2010s 美好生活",
            "aliases": ["2010s", "美好生活", "2010年代", "美好生活历史照片"],
        },
    ]

    MEMORY_PHOTOS = {
        1: [
            "开国大典",
            "人民大会堂建成",
            "长春一汽建成",
        ],
        2: [
            "第一颗原子弹",
            "知青下乡",
            "长江大桥通车",
        ],
        3: [
            "乒乓外交",
            "恢复高考",
            "深圳渔村",
        ],
        4: [
            "奥运首金",
            "首届春晚",
            "肯德基进京",
        ],
        5: [
            "QQ诞生",
            "亚运会盼盼",
            "双休开始",
        ],
        6: [
            "神舟五号",
            "北京奥运会",
            "汶川大地震",
        ],
        7: [
            "辽宁舰",
            "广场舞",
            "移动支付",
        ],
    }

    MEMORY_PHOTO_ALIASES = {
        "开国大典": ["毛主席宣布新中国成立", "街头欢庆的秧歌队", "新中国成立历史时刻"],
        "人民大会堂建成": ["人民大会堂只用了10个月建成", "这个标志性建筑"],
        "长春一汽建成": ["中国第一辆解放牌卡车落地", "参加过工业建设"],
        "第一颗原子弹": ["原子弹", "戈壁滩上的蘑菇云", "蘑菇云让中国挺直腰杆", "您那时如何听到这个消息的"],
        "知青下乡": ["年轻人奔赴农村", "亲友有这段特别经历"],
        "长江大桥通车": ["万里长江第一桥", "保留过它的邮票或照片"],
        "乒乓外交": ["小小乒乓球打开中美大门", "街头乒乓球热潮"],
        "恢复高考": ["关闭十年的考场重启", "改变命运的考试"],
        "深圳渔村": ["小渔村即将巨变", "第一次听说经济特区"],
        "奥运首金": ["奥运会首金", "零的突破"],
        "首届春晚": ["李谷一唱响了乡恋", "电视机是黑白的还是彩色的"],
        "肯德基进京": ["第一家洋快餐", "第一次吃炸鸡是什么时候"],
        "QQ诞生": ["小企鹅改变了沟通方式", "第一次上网是哪年"],
        "亚运会盼盼": ["北京第一次办国际赛事", "收集过盼盼周边"],
        "双休开始": ["周末终于有两天了", "第一个双休日怎么过的"],
        "神舟五号": ["杨利伟成为太空第一人", "中国科技腾飞"],
        "北京奥运会": ["北京欢迎你", "2008年北京奥运会"],
        "汶川大地震": ["512汶川大地震", "万众一心的时刻"],
        "辽宁舰": ["向海图强", "这艘舰的名字"],
        "广场舞": ["常去的广场舞据点在哪里"],
        "移动支付": ["买菜不用带钱包", "学会扫码用了多久"],
    }

    MEMORY_QUESTIONS = {
        1: "您小时候最难忘的集体活动是什么？",
        2: "那个年代最让您自豪的一件事是什么？",
        3: "70年代您家添置的最贵的物品是什么？",
        4: "80年代您最爱的影视剧是哪部？",
        5: "您学会用的第一个电子设备是什么？",
        6: "新世纪最让您惊讶的变化是什么？",
        7: "最近十年，您最想感谢的人是谁？",
    }

    MEMORY_QUESTION_ALIASES = {
        1: ["你小时候最难忘的集体活动是什么"],
        2: ["那个年代最让你自豪的一件事是什么"],
        3: ["70年代你家添置的最贵的物品是什么"],
        4: ["80年代你最爱的影视剧是哪部"],
        5: ["你学会用的第一个电子设备是什么"],
        6: ["新世纪最让你惊讶的变化是什么"],
        7: ["最近十年你最想感谢的人是谁"],
    }

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self._module_patterns = self._build_module_patterns()
        self._thought_entries = self._build_thought_entries()
        self._thinking_positive_entries = self._build_thinking_positive_entries()
        self._memory_period_entries = self._build_memory_period_entries()
        self._memory_photo_entries = self._build_memory_photo_entries()
        self._memory_question_entries = self._build_memory_question_entries()
        self._game_marker_norms = self._build_game_marker_norms()

    def analyze(self, ocr_result: dict, analyzed_frames: list[dict]) -> dict:
        timeline = []
        if isinstance(ocr_result, dict):
            maybe_timeline = ocr_result.get("timeline")
            if isinstance(maybe_timeline, list):
                timeline = maybe_timeline

        samples = self._prepare_samples(timeline)
        if not samples:
            return {
                "summary": {
                    "analysis_version": "v2",
                    "samples_total": 0,
                    "samples_with_text": 0,
                    "samples_with_raw_text": 0,
                    "noise_filtered_samples": 0,
                    "noise_filtered_ratio": 0.0,
                    "raw_line_total": 0,
                    "effective_line_total": 0,
                    "noise_line_total": 0,
                    "keyword_samples_total": 0,
                    "keyword_hits_total": 0,
                    "segments_total": 0,
                    "modules_detected": [],
                    "module_duration_sec": {},
                    "module_segment_count": {},
                    "dominant_module": None,
                },
                "segments": [],
                "modules": [],
                "sample_labels": [],
            }

        analyzed_samples = [self._analyze_sample(sample) for sample in samples]
        segments = self._detect_segments(analyzed_samples)
        self._attach_segment_emotions(segments, analyzed_frames)
        modules = self._aggregate_modules(segments, analyzed_frames)
        summary = self._build_summary(analyzed_samples, segments, modules)

        return {
            "summary": summary,
            "segments": segments,
            "modules": modules,
            "sample_labels": self._build_sample_labels(analyzed_samples),
            "keyword_timeline": self._build_keyword_timeline(analyzed_samples),
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        compact = re.sub(r"\s+", "", str(text or "")).strip().lower()
        if not compact:
            return ""
        compact = re.sub(r"[，。！？；：、“”‘’\"'`~·\-—_（）()【】\[\]{}<>《》/\\|,.!?:;]", "", compact)
        return compact

    @staticmethod
    def _safe_float(value) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return float(number)

    @staticmethod
    def _safe_int(value) -> int | None:
        if value is None:
            return None
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return int(number)

    @classmethod
    def _build_alias_norms(cls, text: str, aliases: list[str] | tuple[str, ...] | None = None) -> list[str]:
        norms: set[str] = set()

        candidates = [str(text or "")]
        if aliases:
            candidates.extend(str(item or "") for item in aliases)

        for candidate in candidates:
            norm = cls._normalize_text(candidate)
            if norm:
                norms.add(norm)

        return sorted(norms)

    def _prepare_samples(self, timeline: list[dict]) -> list[dict]:
        samples: list[dict] = []
        for order_index, row in enumerate(timeline):
            if not isinstance(row, dict):
                continue

            sample_index = self._safe_int(row.get("sample_index"))
            if sample_index is None:
                sample_index = order_index

            timestamp = self._safe_float(row.get("timestamp_sec"))
            if timestamp is None:
                timestamp = float(order_index)

            text_payload = self._prepare_sample_text_payload(row)
            text = str(text_payload.get("clean_text") or "")
            text_raw = str(text_payload.get("raw_text") or "")
            linked_analysis_index = self._safe_int(row.get("linked_analysis_index"))

            samples.append(
                {
                    "sample_index": int(sample_index),
                    "order_index": int(order_index),
                    "timestamp_sec": float(timestamp),
                    "text": text,
                    "text_raw": text_raw,
                    "text_norm": self._normalize_text(text),
                    "line_count": int(text_payload.get("clean_line_count") or 0),
                    "raw_line_count": int(text_payload.get("raw_line_count") or 0),
                    "noise_line_count": int(text_payload.get("noise_line_count") or 0),
                    "linked_analysis_index": linked_analysis_index,
                }
            )

        samples.sort(key=lambda item: (item["sample_index"], item["order_index"]))
        return samples

    @staticmethod
    def _extract_sample_text_lines(row: dict) -> list[str]:
        texts = row.get("texts")
        lines: list[str] = []

        if isinstance(texts, list):
            for raw in texts:
                text = str(raw or "").strip()
                if text:
                    lines.append(text)

        if lines:
            return lines

        fallback = str(row.get("text") or "").strip()
        return [fallback] if fallback else []

    def _prepare_sample_text_payload(self, row: dict) -> dict:
        raw_lines = self._extract_sample_text_lines(row)
        raw_text = " ".join(raw_lines).strip()
        if not raw_text:
            raw_text = str(row.get("text") or "").strip()

        clean_lines: list[str] = []
        seen_norms: set[str] = set()
        noise_line_count = 0

        for line in raw_lines:
            text_norm = self._normalize_text(line)
            if not text_norm:
                continue

            if (not self._contains_game_marker(text_norm)) and self._looks_like_browser_noise(line, text_norm):
                noise_line_count += 1
                continue

            if text_norm in seen_norms:
                continue

            clean_lines.append(line)
            seen_norms.add(text_norm)
            if len(clean_lines) >= self.MAX_TEXT_LINES_PER_SAMPLE:
                break

        clean_text = " ".join(clean_lines).strip()

        return {
            "raw_text": raw_text,
            "clean_text": clean_text,
            "raw_line_count": int(len(raw_lines)),
            "clean_line_count": int(len(clean_lines)),
            "noise_line_count": int(max(0, noise_line_count)),
        }

    def _contains_game_marker(self, text_norm: str) -> bool:
        if not text_norm:
            return False
        for marker in self._game_marker_norms:
            if marker in text_norm:
                return True
        return False

    def _looks_like_browser_noise(self, text: str, text_norm: str) -> bool:
        raw_lower = str(text or "").strip().lower()
        if not raw_lower and not text_norm:
            return True

        if raw_lower and self.OCR_NOISE_URL_REGEX.search(raw_lower):
            return True

        for marker in self.OCR_NOISE_RAW_MARKERS:
            if marker in raw_lower:
                return True

        for marker in self.OCR_NOISE_NORM_MARKERS:
            if marker and marker in text_norm:
                return True

        return False

    def _build_module_patterns(self) -> dict:
        pattern_source = {
            "thinking": {
                "name": ["思维苗圃"],
                "strong": [
                    "请选择一个种子开始培育",
                    "点击工具按钮开始认知重构",
                    "思维苗圃已完成",
                    "请选择其他模块继续旅程",
                ],
                "normal": [
                    "负面念头",
                    "种子",
                    "培育",
                    "认知重构",
                    "浇水",
                    "施肥",
                    "除虫",
                    "返回选择",
                ],
            },
            "lake": {
                "name": ["静心湖畔"],
                "strong": [
                    "跟随圆环的节奏呼吸",
                    "呼吸练习完成",
                    "呼吸训练完成",
                    "训练完成",
                    "静心湖畔已完成",
                ],
                "normal": [
                    "准备",
                    "吸气",
                    "屏息",
                    "呼气",
                    "完成0/3次呼吸训练",
                    "完成第0/3次呼吸训练",
                    "第0/3次呼吸训练",
                    "完成1/3次呼吸训练",
                    "完成第1/3次呼吸训练",
                    "第1/3次呼吸训练",
                    "完成2/3次呼吸训练",
                    "完成第2/3次呼吸训练",
                    "第2/3次呼吸训练",
                    "完成3/3次呼吸训练",
                    "完成第3/3次呼吸训练",
                    "第3/3次呼吸训练",
                ],
            },
            "attic": {
                "name": ["回忆阁楼", "回忆楼阁"],
                "strong": [
                    "时光长廊",
                    "历史照片",
                    "年代照片",
                    "互动提问",
                ],
                "normal": [
                    "互动提问",
                    "建国初期",
                    "激情岁月",
                    "转折年代",
                    "黄金时代",
                    "飞速发展",
                    "崭新纪元",
                    "美好生活",
                ],
            },
            "studio": {
                "name": ["共创画室"],
                "strong": [
                    "想不好画什么",
                    "想不到画什么",
                    "完成绘画",
                    "共创画室已完成",
                ],
                "normal": [
                    "下载绘画",
                    "清空画布",
                    "画笔",
                    "橡皮擦",
                    "参考图片",
                ],
            },
        }

        output = {}
        for module_key, groups in pattern_source.items():
            output[module_key] = {}
            for group_name, texts in groups.items():
                entries = []
                for text in texts:
                    entries.append(
                        {
                            "text": text,
                            "norm": self._normalize_text(text),
                            "alias_norms": self._build_alias_norms(text),
                        }
                    )
                output[module_key][group_name] = entries
        return output

    def _build_game_marker_norms(self) -> set[str]:
        markers: set[str] = set()

        for module_groups in self._module_patterns.values():
            if not isinstance(module_groups, dict):
                continue
            for entries in module_groups.values():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    alias_norms = entry.get("alias_norms") or [entry.get("norm")]
                    if not isinstance(alias_norms, list):
                        alias_norms = [entry.get("norm")]
                    for alias_norm in alias_norms:
                        norm = str(alias_norm or "").strip()
                        if norm:
                            markers.add(norm)

        for entry in self._thought_entries:
            for alias_norm in entry.get("alias_norms") or []:
                norm = str(alias_norm or "").strip()
                if norm:
                    markers.add(norm)

        for entry in self._thinking_positive_entries:
            norm = str(entry.get("norm") or "").strip()
            if norm:
                markers.add(norm)

        for entry in self._memory_period_entries:
            for alias_norm in entry.get("alias_norms") or []:
                norm = str(alias_norm or "").strip()
                if norm:
                    markers.add(norm)

        for entry in self._memory_photo_entries:
            for alias_norm in entry.get("alias_norms") or []:
                norm = str(alias_norm or "").strip()
                if norm:
                    markers.add(norm)

        for entry in self._memory_question_entries:
            for alias_norm in entry.get("alias_norms") or []:
                norm = str(alias_norm or "").strip()
                if norm:
                    markers.add(norm)

        for tool in self.THINKING_TOOL_KEYWORDS:
            norm = self._normalize_text(tool)
            if norm:
                markers.add(norm)

        return markers

    def _build_thought_entries(self) -> list[dict]:
        entries = []
        for day, thoughts in self.THINKING_DAILY_THOUGHTS.items():
            for thought in thoughts:
                aliases = self.THINKING_THOUGHT_ALIASES.get(thought) or []
                entries.append(
                    {
                        "day": int(day),
                        "thought": thought,
                        "norm": self._normalize_text(thought),
                        "alias_norms": self._build_alias_norms(thought, aliases),
                    }
                )
        return entries

    def _build_thinking_positive_entries(self) -> list[dict]:
        entries = []
        for day, thought_map in self.THINKING_POSITIVE_THOUGHTS.items():
            if not isinstance(thought_map, dict):
                continue
            for thought, tool_map in thought_map.items():
                if not isinstance(tool_map, dict):
                    continue
                for tool, positive_text in tool_map.items():
                    text = str(positive_text or "").strip()
                    if not text:
                        continue
                    entries.append(
                        {
                            "day": int(day),
                            "thought": str(thought or "").strip(),
                            "tool": str(tool or "").strip(),
                            "positive_text": text,
                            "norm": self._normalize_text(text),
                        }
                    )
        return entries

    def _build_memory_period_entries(self) -> list[dict]:
        entries = []
        for period in self.MEMORY_PERIODS:
            alias_norms = []
            for alias in period.get("aliases") or []:
                alias_norm = self._normalize_text(alias)
                if alias_norm:
                    alias_norms.append(alias_norm)
            label_norm = self._normalize_text(period["period_name"])
            if label_norm:
                alias_norms.append(label_norm)

            entries.append(
                {
                    "period_index": int(period["period_index"]),
                    "period_name": period["period_name"],
                    "alias_norms": sorted(set(alias_norms)),
                }
            )
        return entries

    def _build_memory_photo_entries(self) -> list[dict]:
        entries = []
        for period_index, titles in self.MEMORY_PHOTOS.items():
            for title in titles:
                aliases = self.MEMORY_PHOTO_ALIASES.get(title) or []
                entries.append(
                    {
                        "period_index": int(period_index),
                        "title": title,
                        "norm": self._normalize_text(title),
                        "alias_norms": self._build_alias_norms(title, aliases),
                    }
                )
        return entries

    def _build_memory_question_entries(self) -> list[dict]:
        entries = []
        for period_index, question in self.MEMORY_QUESTIONS.items():
            aliases = self.MEMORY_QUESTION_ALIASES.get(int(period_index)) or []
            entries.append(
                {
                    "period_index": int(period_index),
                    "question": question,
                    "norm": self._normalize_text(question),
                    "alias_norms": self._build_alias_norms(question, aliases),
                }
            )
        return entries

    @staticmethod
    def _collect_hits(text_norm: str, entries: list[dict]) -> list[str]:
        hits: list[str] = []
        if not text_norm:
            return hits
        for entry in entries:
            markers = entry.get("alias_norms") or [entry.get("norm")]
            if not isinstance(markers, list):
                markers = [entry.get("norm")]

            for marker in markers:
                marker_text = str(marker or "").strip()
                if not marker_text:
                    continue
                if marker_text in text_norm:
                    hits.append(str(entry.get("text") or marker_text))
                    break
        return hits

    @staticmethod
    def _append_keyword_hit(keyword_hits: list[dict], module_key: str, keyword: str, source: str) -> None:
        key = str(module_key or "").strip()
        text = str(keyword or "").strip()
        source_text = str(source or "").strip()
        if not key or not text:
            return

        dedup_key = (key, text)
        for item in keyword_hits:
            exists = (
                str(item.get("module_key") or "") == dedup_key[0]
                and str(item.get("keyword") or "") == dedup_key[1]
            )
            if exists:
                return

        keyword_hits.append(
            {
                "module_key": key,
                "keyword": text,
                "source": source_text if source_text else "pattern",
            }
        )

    @classmethod
    def _is_output_keyword_hit(cls, item: dict) -> bool:
        if not isinstance(item, dict):
            return False
        source = str(item.get("source") or "").strip()
        return source in cls.OUTPUT_KEYWORD_SOURCES

    def _match_thinking_contents(self, text_norm: str) -> list[dict]:
        matches: list[dict] = []
        if not text_norm:
            return matches

        seen = set()

        def _append(match: dict) -> None:
            key = (
                str(match.get("content_type") or ""),
                str(match.get("label") or ""),
                self._safe_int(match.get("day")),
                str(match.get("tool") or ""),
                str(match.get("thought") or ""),
            )
            if key in seen:
                return
            seen.add(key)
            matches.append(match)

        for entry in self._thought_entries:
            alias_norms = entry.get("alias_norms") or [entry.get("norm")]
            if not isinstance(alias_norms, list):
                alias_norms = [entry.get("norm")]

            hit = False
            for marker in alias_norms:
                marker_text = str(marker or "").strip()
                if marker_text and marker_text in text_norm:
                    hit = True
                    break
            if not hit:
                continue

            _append(
                {
                    "module_key": "thinking",
                    "content_type": "thought",
                    "label": entry.get("thought"),
                    "day": entry.get("day"),
                    "weight": 3,
                    "source": "ocr",
                }
            )

        for tool in self.THINKING_TOOL_KEYWORDS:
            marker = self._normalize_text(tool)
            if marker and marker in text_norm:
                _append(
                    {
                        "module_key": "thinking",
                        "content_type": "tool",
                        "label": tool,
                        "weight": 4,
                        "source": "ocr",
                    }
                )

        for entry in self._thinking_positive_entries:
            marker = str(entry.get("norm") or "").strip()
            if not marker or marker not in text_norm:
                continue

            _append(
                    {
                        "module_key": "thinking",
                        "content_type": "reframe",
                        "label": entry.get("positive_text"),
                        "day": entry.get("day"),
                        "tool": entry.get("tool"),
                        "thought": entry.get("thought"),
                        "weight": 4,
                        "source": "ocr",
                    }
                )

        return matches

    def _match_lake_contents(self, text_norm: str) -> list[dict]:
        matches: list[dict] = []
        if not text_norm:
            return matches

        phase_aliases = {
            "准备": ["准备"],
            "吸气": ["吸气"],
            "屏息": ["屏息"],
            "呼气": ["呼气"],
            "进度": [
                "完成03次呼吸训练",
                "完成第03次呼吸训练",
                "第03次呼吸训练",
                "完成13次呼吸训练",
                "完成第13次呼吸训练",
                "第13次呼吸训练",
                "完成23次呼吸训练",
                "完成第23次呼吸训练",
                "第23次呼吸训练",
                "完成33次呼吸训练",
                "完成第33次呼吸训练",
                "第33次呼吸训练",
            ],
            "完成": ["训练完成", "呼吸练习完成", "呼吸训练完成"],
        }

        for label, alias_list in phase_aliases.items():
            for alias in alias_list:
                marker = self._normalize_text(alias)
                if marker and marker in text_norm:
                    matches.append(
                        {
                            "module_key": "lake",
                            "content_type": "breath_phase",
                            "label": label,
                            "weight": 2 if label in {"准备", "吸气", "屏息", "呼气"} else 1,
                            "source": "ocr",
                        }
                    )
                    break

        return matches

    def _match_attic_contents(self, text_norm: str) -> list[dict]:
        matches: list[dict] = []
        if not text_norm:
            return matches

        for entry in self._memory_period_entries:
            for alias_norm in entry.get("alias_norms") or []:
                if alias_norm and alias_norm in text_norm:
                    matches.append(
                        {
                            "module_key": "attic",
                            "content_type": "period",
                            "label": entry.get("period_name"),
                            "period_index": entry.get("period_index"),
                            "weight": 3,
                            "source": "ocr",
                        }
                    )
                    break

        for entry in self._memory_photo_entries:
            alias_norms = entry.get("alias_norms") or [entry.get("norm")]
            if not isinstance(alias_norms, list):
                alias_norms = [entry.get("norm")]

            has_hit = False
            for marker in alias_norms:
                marker_text = str(marker or "").strip()
                if marker_text and marker_text in text_norm:
                    has_hit = True
                    break

            if has_hit:
                matches.append(
                    {
                        "module_key": "attic",
                        "content_type": "photo",
                        "label": entry.get("title"),
                        "period_index": entry.get("period_index"),
                        "weight": 4,
                        "source": "ocr",
                    }
                )

        for entry in self._memory_question_entries:
            alias_norms = entry.get("alias_norms") or [entry.get("norm")]
            if not isinstance(alias_norms, list):
                alias_norms = [entry.get("norm")]

            has_hit = False
            for marker in alias_norms:
                marker_text = str(marker or "").strip()
                if marker_text and marker_text in text_norm:
                    has_hit = True
                    break

            if has_hit:
                matches.append(
                    {
                        "module_key": "attic",
                        "content_type": "question",
                        "label": entry.get("question"),
                        "period_index": entry.get("period_index"),
                        "weight": 4,
                        "source": "ocr",
                    }
                )

        return matches

    def _match_studio_contents(self, text_norm: str) -> list[dict]:
        matches: list[dict] = []
        if not text_norm:
            return matches

        for keyword in self.STUDIO_ACTION_KEYWORDS:
            marker = self._normalize_text(keyword)
            if marker and marker in text_norm:
                matches.append(
                    {
                        "module_key": "studio",
                        "content_type": "studio_action",
                        "label": keyword,
                        "weight": 2,
                        "source": "ocr",
                    }
                )

        return matches

    def _analyze_sample(self, sample: dict) -> dict:
        text_norm = sample.get("text_norm") or ""
        module_scores: dict[str, int] = {key: 0 for key in self.MODULE_ORDER}
        module_hits: dict[str, list[str]] = {key: [] for key in self.MODULE_ORDER}
        name_hit_count: dict[str, int] = {key: 0 for key in self.MODULE_ORDER}
        keyword_hits: list[dict] = []

        for module_key in self.MODULE_ORDER:
            groups = self._module_patterns[module_key]

            for hit in self._collect_hits(text_norm, groups.get("name") or []):
                module_scores[module_key] += 3
                module_hits[module_key].append(f"模块名:{hit}")
                name_hit_count[module_key] += 1
                self._append_keyword_hit(keyword_hits, module_key, hit, "module_name")

            for hit in self._collect_hits(text_norm, groups.get("strong") or []):
                module_scores[module_key] += 2
                module_hits[module_key].append(f"强特征:{hit}")
                self._append_keyword_hit(keyword_hits, module_key, hit, "strong")

            for hit in self._collect_hits(text_norm, groups.get("normal") or []):
                module_scores[module_key] += 1
                module_hits[module_key].append(f"特征:{hit}")
                self._append_keyword_hit(keyword_hits, module_key, hit, "normal")

        content_matches = []
        content_matches.extend(self._match_thinking_contents(text_norm))
        content_matches.extend(self._match_lake_contents(text_norm))
        content_matches.extend(self._match_attic_contents(text_norm))
        content_matches.extend(self._match_studio_contents(text_norm))

        for match in content_matches:
            module_key = str(match.get("module_key") or "")
            if module_key not in module_scores:
                continue
            weight = self._safe_int(match.get("weight"))
            if weight is None:
                weight = 1
            module_scores[module_key] += max(1, weight)

            content_type = str(match.get("content_type") or "内容")
            label = str(match.get("label") or "-")
            module_hits[module_key].append(f"{content_type}:{label}")
            self._append_keyword_hit(keyword_hits, module_key, label, content_type)

        home_hits = [
            self.MODULE_NAMES.get(module_key, module_key)
            for module_key in self.MODULE_ORDER
            if int(name_hit_count.get(module_key, 0) or 0) > 0
        ]
        home_hit_count = len(home_hits)

        predicted_module = self._choose_module(
            module_scores=module_scores,
            home_hit_count=home_hit_count,
            name_hit_count=name_hit_count,
        )
        forced_module = self._pick_content_forced_module(content_matches)
        if forced_module:
            current_score = int(module_scores.get(predicted_module or "", 0) or 0)
            forced_score = int(module_scores.get(forced_module, 0) or 0)
            predicted_name_hits = int(name_hit_count.get(predicted_module or "", 0) or 0)
            predicted_is_weak = predicted_name_hits <= 0
            if (
                (not predicted_module)
                or (forced_score >= current_score + 1)
                or (predicted_is_weak and forced_score >= current_score)
            ):
                predicted_module = forced_module

        is_home = home_hit_count >= 2
        if is_home:
            predicted_module = None

        if predicted_module:
            event_type = "module_active"
            scene_label = self.MODULE_NAMES.get(predicted_module, predicted_module)
        elif is_home:
            event_type = "home_screen"
            scene_label = "主界面"
        else:
            event_type = "unknown"
            scene_label = "未知"

        scored = sorted(module_scores.items(), key=lambda item: item[1], reverse=True)
        top_module = scored[0][0] if scored else None
        top_score = scored[0][1] if scored else 0
        output_keyword_hits = [item for item in keyword_hits if self._is_output_keyword_hit(item)]
        keyword_texts = [
            str(item.get("keyword") or "")
            for item in output_keyword_hits
            if str(item.get("keyword") or "").strip()
        ]

        return {
            **sample,
            "module_scores": module_scores,
            "module_hits": module_hits,
            "content_matches": content_matches,
            "keyword_hits": keyword_hits,
            "output_keyword_hits": output_keyword_hits,
            "keyword_texts": keyword_texts,
            "predicted_module": predicted_module,
            "is_home": bool(is_home),
            "home_hits": home_hits,
            "top_module": top_module,
            "top_score": int(top_score),
            "event_type": event_type,
            "scene_label": scene_label,
        }

    def _pick_content_forced_module(self, content_matches: list[dict]) -> str | None:
        force_scores: dict[str, int] = {}
        for item in content_matches:
            if not isinstance(item, dict):
                continue

            module_key = str(item.get("module_key") or "").strip()
            content_type = str(item.get("content_type") or "").strip()
            if not module_key or not content_type:
                continue

            base_weight = self._safe_int(item.get("weight"))
            if base_weight is None:
                base_weight = 0

            forced_weight = self.CONTENT_FORCED_ENTRY_WEIGHTS.get((module_key, content_type))
            if forced_weight is None:
                continue

            score = max(int(base_weight), int(forced_weight))
            force_scores[module_key] = int(force_scores.get(module_key, 0) + score)

        if not force_scores:
            return None

        ranked = sorted(force_scores.items(), key=lambda row: row[1], reverse=True)
        top_module, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1
        if top_score <= 0 or top_score == second_score:
            return None
        return top_module

    @staticmethod
    def _choose_module(module_scores: dict[str, int], home_hit_count: int, name_hit_count: dict[str, int]) -> str | None:
        if not module_scores:
            return None

        sorted_scores = sorted(module_scores.items(), key=lambda item: item[1], reverse=True)
        best_module, best_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0

        if best_score <= 0:
            return None

        if home_hit_count >= 2 and best_score <= 3:
            return None

        if best_score >= 4:
            return best_module

        if best_score >= 2 and best_score >= second_score + 1:
            return best_module

        best_name_hits = int(name_hit_count.get(best_module, 0) or 0)
        if best_score >= 3 and best_name_hits > 0 and home_hit_count < 2:
            return best_module

        return None

    def _detect_segments(self, samples: list[dict]) -> list[dict]:
        segments: list[dict] = []
        active = None

        for pos, sample in enumerate(samples):
            module_key_raw = sample.get("predicted_module")
            module_key = str(module_key_raw or "")
            is_home = bool(sample.get("is_home"))

            if module_key:
                if active is None:
                    active = self._start_segment(module_key, pos, sample)
                elif active.get("module_key") == module_key:
                    active["end_pos"] = int(pos)
                    active["last_confirmed_pos"] = int(pos)
                    active["closed_by"] = ""
                else:
                    self._finalize_segment(active, samples, closed_by="next_module")
                    if self._keep_segment(active):
                        segments.append(active)
                    active = self._start_segment(module_key, pos, sample)

                self._ingest_segment_sample(active, sample, module_key)
                continue

            if active is None:
                continue

            if is_home:
                active["end_pos"] = max(int(active.get("end_pos") or 0), int(pos))
                self._finalize_segment(active, samples, closed_by="home_screen")
                if self._keep_segment(active):
                    segments.append(active)
                active = None
                continue

            active["end_pos"] = max(int(active.get("end_pos") or 0), int(pos))
            active_module_key = str(active.get("module_key") or "")
            self._ingest_segment_sample(active, sample, active_module_key, allow_weak=True)

        if active is not None:
            self._finalize_segment(active, samples, closed_by="video_end")
            if self._keep_segment(active):
                segments.append(active)

        for seg_index, segment in enumerate(segments):
            segment["segment_index"] = int(seg_index)

        return segments

    def _start_segment(self, module_key: str, pos: int, sample: dict) -> dict:
        scene_label = self.MODULE_NAMES.get(module_key, module_key)
        return {
            "module_key": module_key,
            "module_name": scene_label,
            "scene_label": scene_label,
            "start_reason": self._infer_segment_start_reason(sample, module_key),
            "start_pos": int(pos),
            "end_pos": int(pos),
            "last_confirmed_pos": int(pos),
            "closed_by": "",
            "_content_counter": Counter(),
            "_content_payload": {},
            "evidence_samples": [],
        }

    @staticmethod
    def _infer_segment_start_reason(sample: dict, module_key: str) -> str:
        module_hits = sample.get("module_hits") or {}
        hit_rows = module_hits.get(module_key) if isinstance(module_hits, dict) else []
        if not isinstance(hit_rows, list):
            hit_rows = []

        for hit in hit_rows:
            text = str(hit or "").strip()
            if text.startswith("模块名:"):
                return "module_name_hit"

        for hit in hit_rows:
            text = str(hit or "").strip()
            if text.startswith("强特征:"):
                return "strong_feature_hit"

        content_matches = sample.get("content_matches") or []
        if isinstance(content_matches, list):
            for item in content_matches:
                if not isinstance(item, dict):
                    continue
                if str(item.get("module_key") or "") == module_key:
                    return "content_keyword_hit"

        return "score_inference"

    @staticmethod
    def _build_content_key(payload: dict) -> tuple:
        module_key = str(payload.get("module_key") or "")
        content_type = str(payload.get("content_type") or "")
        label = str(payload.get("label") or "")
        day = payload.get("day")
        period_index = payload.get("period_index")
        tool = str(payload.get("tool") or "")
        thought = str(payload.get("thought") or "")
        source = str(payload.get("source") or "ocr")
        return module_key, content_type, label, day, period_index, tool, thought, source

    def _ingest_segment_sample(self, segment: dict, sample: dict, module_key: str, allow_weak: bool = False) -> None:
        if module_key not in self.MODULE_ORDER:
            return

        module_hits = sample.get("module_hits") or {}
        sample_hits = module_hits.get(module_key) or []
        content_matches = [
            item
            for item in (sample.get("content_matches") or [])
            if str(item.get("module_key") or "") == module_key
        ]
        module_keyword_hits = [
            item
            for item in (sample.get("output_keyword_hits") or [])
            if str(item.get("module_key") or "") == module_key
        ]
        module_keywords = [
            str(item.get("keyword") or "").strip()
            for item in module_keyword_hits
            if str(item.get("keyword") or "").strip()
        ]

        if sample_hits or content_matches or module_keywords:
            evidence = {
                "sample_index": int(sample.get("sample_index") or 0),
                "timestamp_sec": float(sample.get("timestamp_sec") or 0.0),
                "hits": (sample_hits[:6] if isinstance(sample_hits, list) else []),
                "keywords": module_keywords[:8],
            }
            if not segment["evidence_samples"] or segment["evidence_samples"][-1].get("sample_index") != evidence["sample_index"]:
                segment["evidence_samples"].append(evidence)

        for match in content_matches:
            key = self._build_content_key(match)
            segment["_content_counter"][key] += 1
            if key not in segment["_content_payload"]:
                segment["_content_payload"][key] = {
                    "content_type": str(match.get("content_type") or "内容"),
                    "label": str(match.get("label") or "-"),
                    "day": self._safe_int(match.get("day")),
                    "period_index": self._safe_int(match.get("period_index")),
                    "tool": str(match.get("tool") or "").strip() or None,
                    "thought": str(match.get("thought") or "").strip() or None,
                    "source": str(match.get("source") or "ocr").strip() or "ocr",
                }

        if allow_weak and not sample_hits and not content_matches:
            return

        segment["end_pos"] = max(int(segment["end_pos"]), int(sample.get("order_index") or segment["end_pos"]))

    def _finalize_segment(self, segment: dict, samples: list[dict], closed_by: str) -> None:
        start_pos = int(segment.get("start_pos") or 0)
        end_pos = int(segment.get("end_pos") or segment.get("last_confirmed_pos") or start_pos)
        start_pos = max(0, min(start_pos, len(samples) - 1))
        end_pos = max(start_pos, min(end_pos, len(samples) - 1))

        segment_samples = samples[start_pos : end_pos + 1]
        if not segment_samples:
            segment.update(
                {
                    "start_sample_index": 0,
                    "end_sample_index": 0,
                    "start_sec": 0.0,
                    "end_sec": 0.0,
                    "duration_sec": 0.0,
                    "sample_count": 0,
                    "start_linked_analysis_index": None,
                    "end_linked_analysis_index": None,
                    "content_labels": [],
                    "closed_by": closed_by,
                    "end_reason": str(closed_by),
                    "scene_label": str(segment.get("scene_label") or "未知"),
                    "start_reason": str(segment.get("start_reason") or "score_inference"),
                }
            )
            return

        start_sample = segment_samples[0]
        end_sample = segment_samples[-1]

        start_sec = self._safe_float(start_sample.get("timestamp_sec"))
        end_sec = self._safe_float(end_sample.get("timestamp_sec"))
        if start_sec is None:
            start_sec = 0.0
        if end_sec is None:
            end_sec = start_sec

        if end_sec < start_sec:
            end_sec = start_sec

        duration_sec = float(max(0.0, end_sec - start_sec))
        start_linked = self._resolve_linked_index(segment_samples, pick_first=True)
        end_linked = self._resolve_linked_index(segment_samples, pick_first=False)

        content_labels = []
        for key, hit_count in segment.get("_content_counter", {}).most_common(24):
            payload = dict(segment.get("_content_payload", {}).get(key) or {})
            payload["hit_count"] = int(hit_count)
            content_labels.append(payload)

        segment.update(
            {
                "start_sample_index": int(start_sample.get("sample_index") or 0),
                "end_sample_index": int(end_sample.get("sample_index") or 0),
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "duration_sec": duration_sec,
                "sample_count": int(len(segment_samples)),
                "start_linked_analysis_index": start_linked,
                "end_linked_analysis_index": end_linked,
                "content_labels": content_labels,
                "closed_by": str(closed_by),
                "end_reason": str(closed_by),
                "scene_label": str(segment.get("scene_label") or segment.get("module_name") or "未知"),
                "start_reason": str(segment.get("start_reason") or "score_inference"),
            }
        )

        evidence = segment.get("evidence_samples")
        if not isinstance(evidence, list):
            evidence = []
        segment["evidence_samples"] = evidence[:12]

        segment.pop("_content_counter", None)
        segment.pop("_content_payload", None)
        segment.pop("start_pos", None)
        segment.pop("end_pos", None)
        segment.pop("last_confirmed_pos", None)

    @staticmethod
    def _resolve_linked_index(segment_samples: list[dict], pick_first: bool) -> int | None:
        iterable = segment_samples if pick_first else reversed(segment_samples)
        for sample in iterable:
            value = GameplayTimelineService._safe_int(sample.get("linked_analysis_index"))
            if value is not None:
                return value
        return None

    def _keep_segment(self, segment: dict) -> bool:
        sample_count = int(segment.get("sample_count") or 0)
        duration_sec = self._safe_float(segment.get("duration_sec")) or 0.0
        has_content = bool(segment.get("content_labels"))
        has_evidence = bool(segment.get("evidence_samples"))

        if sample_count <= 0:
            return False
        if duration_sec >= self.MIN_SEGMENT_DURATION_SEC:
            return True
        return has_content or has_evidence

    def _attach_segment_emotions(self, segments: list[dict], analyzed_frames: list[dict]) -> None:
        for segment in segments:
            start_sec = self._safe_float(segment.get("start_sec"))
            end_sec = self._safe_float(segment.get("end_sec"))
            frames = self._collect_frames_by_time(analyzed_frames, start_sec, end_sec)

            if not frames:
                start_idx = self._safe_int(segment.get("start_linked_analysis_index"))
                end_idx = self._safe_int(segment.get("end_linked_analysis_index"))
                frames = self._collect_frames_by_index(analyzed_frames, start_idx, end_idx)

            segment["emotion"] = self._build_emotion_payload(frames)

    @staticmethod
    def _collect_frames_by_time(analyzed_frames: list[dict], start_sec: float | None, end_sec: float | None) -> list[dict]:
        if start_sec is None or end_sec is None:
            return []

        start = float(start_sec)
        end = float(end_sec)
        if end < start:
            start, end = end, start

        output = []
        for row in analyzed_frames:
            if not isinstance(row, dict):
                continue
            ts = GameplayTimelineService._safe_float(row.get("timestamp"))
            if ts is None:
                continue
            if start - 1e-9 <= ts <= end + 1e-9:
                output.append(row)
        return output

    @staticmethod
    def _collect_frames_by_index(analyzed_frames: list[dict], start_idx: int | None, end_idx: int | None) -> list[dict]:
        if start_idx is None and end_idx is None:
            return []

        if start_idx is None:
            start_idx = end_idx
        if end_idx is None:
            end_idx = start_idx
        if start_idx is None or end_idx is None:
            return []

        begin = min(start_idx, end_idx)
        finish = max(start_idx, end_idx)

        output = []
        for row in analyzed_frames:
            if not isinstance(row, dict):
                continue
            idx = GameplayTimelineService._safe_int(row.get("index"))
            if idx is None:
                continue
            if begin <= idx <= finish:
                output.append(row)
        return output

    def _build_emotion_payload(self, frames: list[dict]) -> dict:
        frame_count = len(frames)
        macro_count = sum(1 for row in frames if bool(row.get("macro_inference")))
        return {
            "frame_count": int(frame_count),
            "macro_frame_count": int(macro_count),
            "deepface": self._aggregate_deepface(frames),
            "emonet": self._aggregate_emonet(frames),
        }

    @classmethod
    def _normalize_probability_dict(cls, scores: dict) -> dict[str, float] | None:
        if not isinstance(scores, dict) or not scores:
            return None

        cleaned = {}
        total = 0.0
        for emotion, value in scores.items():
            number = cls._safe_float(value)
            if number is None or number < 0:
                continue
            key = str(emotion).strip().lower()
            if not key:
                continue
            cleaned[key] = number
            total += number

        if total <= 1e-9:
            return None
        return {emotion: value / total for emotion, value in cleaned.items()}

    @classmethod
    def _aggregate_deepface(cls, frames: list[dict]) -> dict:
        probs_sum = defaultdict(float)
        used = 0

        for row in frames:
            deepface = row.get("deepface")
            if not isinstance(deepface, dict):
                continue

            scores = deepface.get("scores")
            if not isinstance(scores, dict):
                continue

            normalized = cls._normalize_probability_dict(scores)
            if not normalized:
                continue

            used += 1
            for emotion, value in normalized.items():
                probs_sum[emotion] += value

        if used <= 0:
            return {
                "frames_used": 0,
                "emotion_distribution": {},
                "top_emotions": [],
                "dominant_emotion": None,
                "tone_index": None,
            }

        distribution = {
            emotion: float(round(value / used * 100.0, 4))
            for emotion, value in sorted(probs_sum.items(), key=lambda item: item[1], reverse=True)
            if value > 1e-9
        }

        top_emotions = [
            {"emotion": emotion, "percent": float(round(percent, 3))}
            for emotion, percent in list(distribution.items())[:3]
        ]
        dominant_emotion = top_emotions[0]["emotion"] if top_emotions else None

        positive = (
            float(distribution.get("happy", 0.0))
            + float(distribution.get("surprise", 0.0))
            + float(distribution.get("surprised", 0.0))
            + 0.5 * float(distribution.get("neutral", 0.0))
        )
        negative = (
            float(distribution.get("sad", 0.0))
            + float(distribution.get("fear", 0.0))
            + float(distribution.get("angry", 0.0))
            + float(distribution.get("disgust", 0.0))
            + float(distribution.get("contempt", 0.0))
            + float(distribution.get("contemptuous", 0.0))
        )
        tone_index = (positive - negative) / 100.0

        return {
            "frames_used": int(used),
            "emotion_distribution": distribution,
            "top_emotions": top_emotions,
            "dominant_emotion": dominant_emotion,
            "tone_index": float(round(tone_index, 6)),
        }

    @classmethod
    def _aggregate_emonet(cls, frames: list[dict]) -> dict:
        valence_values = []
        arousal_values = []

        for row in frames:
            emonet = row.get("emonet")
            if not isinstance(emonet, dict):
                continue

            valence = cls._safe_float(emonet.get("valence"))
            arousal = cls._safe_float(emonet.get("arousal"))
            if valence is None or arousal is None:
                continue

            valence_values.append(max(-1.0, min(1.0, valence)))
            arousal_values.append(max(-1.0, min(1.0, arousal)))

        used = len(valence_values)
        if used <= 0:
            return {
                "frames_used": 0,
                "valence_mean": None,
                "arousal_mean": None,
                "valence_min": None,
                "valence_max": None,
                "arousal_min": None,
                "arousal_max": None,
                "positive_valence_ratio": None,
                "high_arousal_ratio": None,
            }

        valence_mean = sum(valence_values) / used
        arousal_mean = sum(arousal_values) / used
        positive_ratio = sum(1 for item in valence_values if item > 0.10) / used
        high_arousal_ratio = sum(1 for item in arousal_values if item > 0.25) / used

        return {
            "frames_used": int(used),
            "valence_mean": float(round(valence_mean, 6)),
            "arousal_mean": float(round(arousal_mean, 6)),
            "valence_min": float(round(min(valence_values), 6)),
            "valence_max": float(round(max(valence_values), 6)),
            "arousal_min": float(round(min(arousal_values), 6)),
            "arousal_max": float(round(max(arousal_values), 6)),
            "positive_valence_ratio": float(round(positive_ratio, 6)),
            "high_arousal_ratio": float(round(high_arousal_ratio, 6)),
        }

    def _aggregate_modules(self, segments: list[dict], analyzed_frames: list[dict]) -> list[dict]:
        modules = []
        for module_key in self.MODULE_ORDER:
            module_segments = [segment for segment in segments if segment.get("module_key") == module_key]
            if not module_segments:
                continue

            duration_sec = 0.0
            module_content_counter = Counter()
            module_content_payload = {}

            frame_by_index = {}
            fallback_frames = []
            for segment in module_segments:
                duration_sec += float(segment.get("duration_sec") or 0.0)

                for item in segment.get("content_labels") or []:
                    content_type = str(item.get("content_type") or "内容")
                    label = str(item.get("label") or "-")
                    day = self._safe_int(item.get("day"))
                    period_index = self._safe_int(item.get("period_index"))
                    tool = str(item.get("tool") or "").strip() or None
                    thought = str(item.get("thought") or "").strip() or None
                    source = str(item.get("source") or "ocr").strip() or "ocr"
                    key = (content_type, label, day, period_index, tool, thought, source)
                    module_content_counter[key] += int(item.get("hit_count") or 0)
                    if key not in module_content_payload:
                        module_content_payload[key] = {
                            "content_type": content_type,
                            "label": label,
                            "day": day,
                            "period_index": period_index,
                            "tool": tool,
                            "thought": thought,
                            "source": source,
                        }

                start_sec = self._safe_float(segment.get("start_sec"))
                end_sec = self._safe_float(segment.get("end_sec"))
                seg_frames = self._collect_frames_by_time(analyzed_frames, start_sec, end_sec)
                if not seg_frames and (start_sec is None or end_sec is None):
                    start_idx = self._safe_int(segment.get("start_linked_analysis_index"))
                    end_idx = self._safe_int(segment.get("end_linked_analysis_index"))
                    seg_frames = self._collect_frames_by_index(analyzed_frames, start_idx, end_idx)

                for frame in seg_frames:
                    frame_index = self._safe_int(frame.get("index"))
                    if frame_index is None:
                        fallback_frames.append(frame)
                    else:
                        frame_by_index[frame_index] = frame

            merged_frames = list(frame_by_index.values())
            if not merged_frames and fallback_frames:
                merged_frames = fallback_frames

            content_labels = []
            for key, hit_count in module_content_counter.most_common(24):
                payload = dict(module_content_payload.get(key) or {})
                payload["hit_count"] = int(hit_count)
                content_labels.append(payload)

            modules.append(
                {
                    "module_key": module_key,
                    "module_name": self.MODULE_NAMES.get(module_key, module_key),
                    "segment_count": int(len(module_segments)),
                    "duration_sec": float(round(duration_sec, 6)),
                    "emotion": self._build_emotion_payload(merged_frames),
                    "content_labels": content_labels,
                }
            )

        return modules

    def _build_summary(self, samples: list[dict], segments: list[dict], modules: list[dict]) -> dict:
        samples_total = len(samples)
        samples_with_text = sum(1 for sample in samples if bool(str(sample.get("text") or "").strip()))
        samples_with_raw_text = sum(1 for sample in samples if bool(str(sample.get("text_raw") or "").strip()))
        keyword_samples_total = sum(1 for sample in samples if bool(sample.get("keyword_texts")))
        keyword_hits_total = sum(len(sample.get("keyword_texts") or []) for sample in samples)
        noise_filtered_samples = sum(
            1
            for sample in samples
            if bool(str(sample.get("text_raw") or "").strip()) and not bool(str(sample.get("text") or "").strip())
        )
        raw_line_total = sum(int(self._safe_int(sample.get("raw_line_count")) or 0) for sample in samples)
        effective_line_total = sum(int(self._safe_int(sample.get("line_count")) or 0) for sample in samples)
        noise_line_total = sum(int(self._safe_int(sample.get("noise_line_count")) or 0) for sample in samples)
        noise_filtered_ratio = float(noise_filtered_samples / samples_total) if samples_total > 0 else 0.0

        module_duration_sec = {}
        module_segment_count = {}
        for item in modules:
            key = str(item.get("module_key") or "")
            if not key:
                continue
            module_duration_sec[key] = float(item.get("duration_sec") or 0.0)
            module_segment_count[key] = int(item.get("segment_count") or 0)

        modules_detected = [
            key
            for key in self.MODULE_ORDER
            if float(module_duration_sec.get(key, 0.0)) > 0.0 or int(module_segment_count.get(key, 0)) > 0
        ]

        dominant_module = None
        if modules_detected:
            dominant_module = max(
                modules_detected,
                key=lambda key: (
                    float(module_duration_sec.get(key, 0.0)),
                    int(module_segment_count.get(key, 0)),
                ),
            )

        return {
            "analysis_version": "v2",
            "samples_total": int(samples_total),
            "samples_with_text": int(samples_with_text),
            "samples_with_raw_text": int(samples_with_raw_text),
            "keyword_samples_total": int(keyword_samples_total),
            "keyword_hits_total": int(keyword_hits_total),
            "noise_filtered_samples": int(noise_filtered_samples),
            "noise_filtered_ratio": float(noise_filtered_ratio),
            "raw_line_total": int(raw_line_total),
            "effective_line_total": int(effective_line_total),
            "noise_line_total": int(noise_line_total),
            "segments_total": int(len(segments)),
            "modules_detected": modules_detected,
            "module_duration_sec": module_duration_sec,
            "module_segment_count": module_segment_count,
            "dominant_module": dominant_module,
            "dominant_module_name": self.MODULE_NAMES.get(dominant_module) if dominant_module else None,
        }

    def _build_sample_labels(self, samples: list[dict]) -> list[dict]:
        output = []
        for sample in samples:
            labels = [str(item or "").strip() for item in (sample.get("keyword_texts") or []) if str(item or "").strip()][:6]

            predicted_module_raw = sample.get("predicted_module")
            predicted_module = str(predicted_module_raw or "")
            predicted_module_name = self.MODULE_NAMES.get(predicted_module) if predicted_module else None
            output.append(
                {
                    "sample_index": int(sample.get("sample_index") or 0),
                    "timestamp_sec": float(sample.get("timestamp_sec") or 0.0),
                    "predicted_module": predicted_module if predicted_module else None,
                    "predicted_module_name": predicted_module_name,
                    "event_type": str(sample.get("event_type") or "unknown"),
                    "scene_label": str(sample.get("scene_label") or "未知"),
                    "is_home": bool(sample.get("is_home")),
                    "top_module": sample.get("top_module"),
                    "top_score": int(sample.get("top_score") or 0),
                    "line_count": int(sample.get("line_count") or 0),
                    "raw_line_count": int(sample.get("raw_line_count") or 0),
                    "noise_line_count": int(sample.get("noise_line_count") or 0),
                    "keyword_hits": labels,
                    "matched_contents": labels,
                    "linked_analysis_index": self._safe_int(sample.get("linked_analysis_index")),
                }
            )

        return output

    def _build_keyword_timeline(self, samples: list[dict]) -> list[dict]:
        output = []
        for sample in samples:
            keyword_texts = [str(item or "").strip() for item in (sample.get("keyword_texts") or []) if str(item or "").strip()]
            if not keyword_texts:
                continue

            output.append(
                {
                    "sample_index": int(sample.get("sample_index") or 0),
                    "timestamp_sec": float(sample.get("timestamp_sec") or 0.0),
                    "predicted_module": sample.get("predicted_module"),
                    "predicted_module_name": self.MODULE_NAMES.get(str(sample.get("predicted_module") or ""))
                    if sample.get("predicted_module")
                    else None,
                    "event_type": str(sample.get("event_type") or "unknown"),
                    "scene_label": str(sample.get("scene_label") or "未知"),
                    "linked_analysis_index": self._safe_int(sample.get("linked_analysis_index")),
                    "keywords": keyword_texts[:8],
                }
            )

        return output
