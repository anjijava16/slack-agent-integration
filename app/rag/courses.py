"""
Course catalogue (config/courses.yaml).

Each course gets its own Qdrant collection and its own Slack channels, exactly
like the original faq-slack-bot kept four query engines and routed by channel id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_REPO_INCLUDE = [".md", ".ipynb", ".py", ".txt", ".sql", ".sh", ".yaml", ".yml", ".toml", ".dockerfile"]
DEFAULT_REPO_EXCLUDE_DIRS = [".github", "images", "img", "assets"]


@dataclass
class RepoSpec:
    owner: str
    repo: str
    branch: str = "main"
    include_ext: list[str] = field(default_factory=lambda: list(DEFAULT_REPO_INCLUDE))
    exclude_dirs: list[str] = field(default_factory=lambda: list(DEFAULT_REPO_EXCLUDE_DIRS))
    local_path: str | None = None  # use an existing checkout instead of downloading

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


@dataclass
class Course:
    key: str
    name: str
    short_name: str = ""
    cohort_year: int | None = None
    course_start_date: str = ""
    channels: list[str] = field(default_factory=list)
    slack_ingest: list[str] = field(default_factory=list)
    slack_days: int = 90
    faq_dir: str | None = None
    repos: list[RepoSpec] = field(default_factory=list)
    youtube_videos: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.short_name or self.name


class CourseCatalog:
    def __init__(self, courses: dict[str, Course], default_course: str | None = None) -> None:
        self.courses = courses
        self.default_course = default_course or None
        if self.default_course and self.default_course not in courses:
            raise ValueError(f"RAG_DEFAULT_COURSE={default_course!r} is not in the course list {sorted(courses)}")
        self._by_channel = {ch: c for c in courses.values() for ch in c.channels}

    def __iter__(self):
        return iter(self.courses.values())

    def __contains__(self, key: str) -> bool:
        return key in self.courses

    def get(self, key: str) -> Course:
        try:
            return self.courses[key]
        except KeyError as exc:
            raise KeyError(f"Unknown course {key!r}. Known: {', '.join(sorted(self.courses))}") from exc

    def keys(self) -> list[str]:
        return sorted(self.courses)

    def for_channel(self, channel_id: str | None) -> Course | None:
        """The course whose Slack channels include this one, else RAG_DEFAULT_COURSE, else None."""
        if channel_id and channel_id in self._by_channel:
            return self._by_channel[channel_id]
        return self.courses.get(self.default_course) if self.default_course else None

    @classmethod
    def load(cls, path: str | Path, default_course: str | None = None, extra_channels: str = "") -> CourseCatalog:
        """Load YAML; `extra_channels` ("C123=llm,C456=de") adds channel routes without editing the file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        courses: dict[str, Course] = {}
        for key, raw in (data.get("courses") or {}).items():
            raw = dict(raw or {})
            repos = [RepoSpec(**r) for r in raw.pop("repos", []) or []]
            courses[key] = Course(key=key, repos=repos, **raw)
        for pair in filter(None, (p.strip() for p in extra_channels.split(","))):
            channel, _, course_key = pair.partition("=")
            if course_key not in courses:
                raise ValueError(f"RAG_CHANNEL_MAP: {course_key!r} is not a course ({sorted(courses)})")
            courses[course_key].channels.append(channel.strip())
        return cls(courses, default_course=default_course)
