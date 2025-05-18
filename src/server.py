import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any, Optional
from rag_chain import RAGChain

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FastMCP server with configuration
mcp = FastMCP("markdown_rag_KR")

@mcp.tool()
async def get_weather(location: str) -> str:
    # Return a mock weather response
    # In a real implementation, this would call a weather API
    return f"It's always Sunny in {location}"


# RAG 실행 함수
@mcp.tool()
async def run_rag(question: str,
                  context: str,
                  history: Optional[str]):
    # LLM과 Embeddings 설정
    print("🛠️ run_rag tool invoked")

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",
                                  openai_api_key=OPENAI_API_KEY)
    llm = ChatOpenAI(model="gpt-4o-mini",
                     temperature=0.5,
                     openai_api_key=OPENAI_API_KEY)

    # RAGChain 생성
    rag_chain = RAGChain(llm=llm,
                         embeddings=embeddings)

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

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()