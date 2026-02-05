# app/__init__.py
"""
VP-Web-Search Application

범용 데이터 분석 및 웹 검색 시스템
"""
from .agents import ResearchAgent, build_research_agent
from .schemas import AnalysisRequest, AnalysisResponse

__version__ = "2.0.0"
__all__ = [
    "ResearchAgent",
    "build_research_agent",
    "AnalysisRequest",
    "AnalysisResponse",
]
