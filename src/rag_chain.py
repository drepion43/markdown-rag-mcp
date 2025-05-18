import os
import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChain:
    def __init__(self, llm: object, embeddings: Optional[object] = None):
        self.llm = llm
        # embeddings 초기화 시 API 키 확인
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is not set in RAGChain")
            raise ValueError("OPENAI_API_KEY is not set")
        self.embeddings = embeddings or OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=OPENAI_API_KEY,
            timeout=30
        )

    async def create_retriever(self, markdown_text: str):
        """Markdown 텍스트 → Retriever 생성"""
        try:
            logger.info("Creating retriever from markdown text")
            doc = Document(page_content=markdown_text)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = splitter.split_documents([doc])
            vector = FAISS.from_documents(split_docs, self.embeddings)
            return vector.as_retriever()
        except Exception as e:
            logger.error(f"Failed to create retriever: {str(e)}")
            raise

    def format_docs(self, docs: List[Document]) -> str:
        """검색된 문서 리스트를 하나의 문자열로 병합"""
        return "\n".join([doc.page_content for doc in docs])

    async def invoke(self, inputs: Dict[str, Any]) -> str:
        try:
            async with asyncio.timeout(45):  # 45초 타임아웃
                question: str = inputs.get("question", "")
                context_docs: List[Document] = inputs.get("context", [])
                history: List[str] = inputs.get("chat_history", [])

                logger.info(f"Invoking RAG chain with question: {question}")
                context_text = self.format_docs(context_docs) if isinstance(context_docs, list) else context_docs
                history_text = "\n".join(history)

                prompt = PromptTemplate.from_template(
                    """
                    당신은 주어진 문서를 기반으로 질문에 답변하는 AI입니다.

                    🔹 과거 대화 내용 (참고용):
                    {chat_history}
                    (과거 대화는 사용자가 이전에 물어본 질문과 그에 대한 답변들입니다. 이 대화는 사용자의 의도를 이해하는 데 중요하며, 사용자가 계속해서 진행하고자 하는 대화의 흐름을 추적해야 합니다.)

                    🔹 검색된 문서 내용 (반드시 참고):
                    {context}
                    (검색된 문서는 질문에 대한 답을 제공할 때 가장 중요한 정보입니다. 문서에서 제공된 내용만을 사용하여 답변해야 합니다.)

                    💬 사용자 질문:
                    {question}
                    (이 질문은 사용자가 현재 원하는 정보입니다. 답변은 이 질문을 직접적으로 다뤄야 합니다.)

                    📌 지침:
                    - 과거 대화 내용은 현재 질문과 관련된 맥락을 제공합니다. 대화의 흐름을 놓치지 않도록 주의하세요.
                    - 반드시 제공된 문서 내용에만 기반하여 답변하세요.
                    - 문서에 없는 내용은 "문서에 해당 내용이 없습니다."라고 답하세요.
                    - 간결하고 전문적인 문장으로 대답하세요.

                    🧠 답변:
                    """
                )

                chain: Runnable = prompt | self.llm | StrOutputParser()
                response = await chain.ainvoke({
                    "chat_history": history_text,
                    "context": context_text,
                    "question": question
                })
                return response
        except asyncio.TimeoutError:
            logger.error("RAG chain invocation timed out after 45 seconds")
            return "Error: RAG processing timed out"
        except Exception as e:
            logger.error(f"RAG chain invocation failed: {str(e)}")
            return f"Error: {str(e)}"