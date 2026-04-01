import os

from langchain_openai import ChatOpenAI


def get_llm(
    model: str | None = None,
    temperature: float = 0.3,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or os.environ.get("LLM_MODEL", "google/gemini-2.5-flash"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=temperature,
    )
