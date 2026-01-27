from typing import Any, Dict, List, Optional
from langchain_community.tools import TavilySearchResults

def search_tool(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Tavily Search API를 LangChain Tool로 호출.
    include_raw_content를 켜면, 저장용 원문 근거를 같이 챙기기 좋음.
    """
    tool = TavilySearchResults(
        max_results=max_results,
        include_answer=True,
        include_raw_content=True,
    )
    # tool.invoke는 보통 list[dict] 형태를 반환
    return tool.invoke({"query": query})
