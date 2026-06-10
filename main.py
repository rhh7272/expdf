# pip install --upgrade langchain langchain-community langchain-text-splitters langchain-openai langchain-chroma pypdf python-dotenv
# pip install langchain

import os  # 내부에 들어가 있는 코어 라이브러리 (설치 없이 사용 가능)
import streamlit as st
import tempfile
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

st.title("📄 PDF 문서 질문 봇")

def init_rag_chain(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,           # 하나의 청크가 가질 최대 글자 수
        chunk_overlap  = 20,        # 청크 간 문맥 연결을 위해 겹칠 글자 수
        length_function = len,      # 길이 측정 기준 (기본 문자열 길이)
        is_separator_regex = False, # 구분 기호의 정규표현식 해석 여부
    )
    texts = text_splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    db = Chroma.from_documents(texts, embeddings_model)

    # 대화형 LLM 모델 초기화 (예: GPT-4o-mini, 디폴트는 무료인 gpt-3.5-turbo)
    # 멀티 쿼리 리트리버 생성 및 LLM과 설정, 모델을 명시적으로 지정할 수 있음 (예: gpt-4o-mini)
    llma = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # temperature=0은 답변의 일관성을 높이기 위해 사용 (2로 세팅하면 창의성이 증가)

    # 사용자의 질문을 다양한 각도에서 재해석해서 검색 확률을 높이는 멀티 쿼리 리트리버(MultiQueryRetriever) 개체를 생성
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever = db.as_retriever(), 
        llm = llma
    )

    # RAG 체인 생성
    # LLM(시스템) 프롬프트 정의: LLM이 질문에 답할 때 사용할 지침과 맥락을 포함
    system_prompt = (
        "너는 질문-답변을 돕는 유능한 비서야. "
        "아래 제공된 맥락(context)만을 사용하여 질문에 답해줘. "
        "답을 모르면 모른다고 하고, 절대 답변을 지어내지 마.\n\n"
        "{context}"
    )
    # 대화형 프롬프트 템플릿 생성: 시스템(system) 프롬프트와 사용자(human or user) 입력을 결합하여 LLM이 이해할 수 있는 형식으로 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 검색된 문서를 활용하여 질문에 답변하는 체인 생성
    question_answer_chain = create_stuff_documents_chain(llma, prompt)

    # RAG 체인에서 사용할 리트리버로 Chroma db를 참조하여, 검색된 문서를 활용할 수 있도록 준비
    # 크로마 데이터베이스에서 검색된 문서를 활용하여 질문에 답변하는 체인 생성
    retriver = db.as_retriever()  # 추가

    # RAG 체인 생성: 멀티 쿼리 리트리버와 질문-답변 체인을 결합하여, 사용자의 질문에 대해 검색된 문서를 활용하여 답변을 생성하는 체인
    rag_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)
    
    return rag_chain

# 파일 업로더 추가
uploaded_file = st.file_uploader("PDF 파일을 업로드해주세요", type=["pdf"])

if uploaded_file is not None:
    # 새로운 파일이 업로드되었거나 아직 처리되지 않은 경우
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("PDF 문서를 읽고 분석 중입니다. 잠시만 기다려주세요..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            st.session_state.rag_chain = init_rag_chain(tmp_file_path)
            st.session_state.current_file = uploaded_file.name
            
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass
            
            st.success(f"'{uploaded_file.name}' 문서 분석 완료!")

    # 질문 입력란 (Streamlit Text Input)
    question = st.text_input("질문을 입력하세요:", placeholder="예: 문서의 핵심 내용은 무엇인가요?")

    # 버튼 생성
    if st.button("답변 생성", type="primary"):
        if question.strip():  # 질문이 비어 있지 않을 경우
            with st.spinner("답변을 생각하는 중입니다..."):
                # RAG 체인에 질문을 입력하여 답변과 함께 검색된 참조 문서(context)를 반환
                response = st.session_state.rag_chain.invoke({"input": question})
                
                # 결과 화면에 출력
                st.subheader("--- [ 최종 답변 ] ---")
                st.write(response['answer'])
        else:
            st.warning("먼저 질문을 입력해 주세요.")
else:
    st.info("분석할 PDF 문서를 먼저 업로드해 주세요.")
    # 파일이 없을 경우 기존 세션 초기화
    if "rag_chain" in st.session_state:
        del st.session_state["rag_chain"]
    if "current_file" in st.session_state:
        del st.session_state["current_file"]
