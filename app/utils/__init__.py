# app/utils/__init__.py
"""유틸리티 함수"""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> str:
    """텍스트에서 JSON 부분 추출"""
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


def safe_json_loads(text: str, default: Any = None) -> Any:
    """안전한 JSON 파싱"""
    try:
        json_text = extract_json(text)
        return json.loads(json_text)
    except (json.JSONDecodeError, IndexError):
        return default


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """텍스트 자르기"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """텍스트 정제"""
    # 연속된 공백/줄바꿈 정리
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()
