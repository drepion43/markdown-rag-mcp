import os
import asyncio
from functools import wraps
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any, Optional
from rag_chain import RAGChain
from dotenv import load_dotenv
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API 키 로드 (글로벌 로드 최소화)
load_dotenv(override=True)

# 비동기 타임아웃 데코레이터
def async_timeout(seconds):
    def decorator(f):
        @wraps(f)
        async def decorated(*args, **kwargs):
            async with asyncio.timeout(seconds):
                return await f(*args, **kwargs)
        return decorated
    return decorator

# FastMCP 서버 초기화
mcp = FastMCP("markdown_rag_KR", timeout=300)  # 타임아웃 옵션 추가 가능 여부 확인 필요

@mcp.tool()
async def get_weather(location: str) -> str:
    logger.info(f"get_weather called for location: {location}")
    return f"It's always Sunny in {location}"

@mcp.tool()
@async_timeout(60)  # 60초 타임아웃
async def run_rag(question: str, context: str, history: Optional[str]):
    logger.info(f"run_rag invoked with question: {question}")
    
    # API 키 로드 (툴 실행 시에만)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set")
        return "Error: OPENAI_API_KEY is not set"

    try:
        # OpenAI 설정에 타임아웃 추가
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=OPENAI_API_KEY,
            timeout=30
        )
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            openai_api_key=OPENAI_API_KEY,
            timeout=30
        )

        # RAGChain 생성
        rag_chain = RAGChain(llm=llm, embeddings=embeddings)

        # 문서 검색기 생성
        retriever = await rag_chain.create_retriever(context)
        relevant_docs = await retriever.get_relevant_documents(question)

        # RAG 실행
        response = await rag_chain.invoke({
            "question": question,
            "context": relevant_docs,
            "chat_history": history
        })
        return response
    except asyncio.TimeoutError:
        logger.error("run_rag timed out after 60 seconds")
        return "Error: RAG processing timed out"
    except Exception as e:
        logger.error(f"run_rag failed: {str(e)}")
        return f"Error: {str(e)}"

def main():
    try:
        logger.info("Listing registered tools")
        print("Registered tools:", mcp.list_tools())
    except Exception as e:
        logger.error(f"Failed to list tools: {str(e)}")
        print(f"Failed to list tools: {e}")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()