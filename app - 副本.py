import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
#rom langchain_community.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms.openai import OpenAI
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub, __all__
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
#from openai import OpenAI
import os

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
#上面的代码为了解决 pycharm无法链接vpn 的问题


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    #llm = OpenAI(#这里要是用chatopenai否则是直接使用openai的接口，到时候就会是直接封禁的样子如下：The API you are accessing is not open' 。
    #    api_key = "key",
    #    base_url = "https://api.moonshot.cn/v1")
    llm = ChatOpenAI(#调用kimi的接口
    openai_api_base="https://api.moonshot.cn/v1/",
    openai_api_key="key",
    model_name="moonshot-v1-8k",)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})


    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        #ConversationalRetrievalChain是用于处理包含聊天历史记录的对话链路的类。
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()# todo 如何说明在哪里显示结果
    st.set_page_config(page_title="使用多个pdf进行reg"
                       )
    st.write(css, unsafe_allow_html=True)

    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("使用多个pdf进行reg")
    user_question = st.text_input("请准对pdfs提出你的问题")
    if user_question:
        handle_userinput(user_question)
#with 类似于try finally 最后一定会退出
    with st.sidebar:
        st.subheader("你的文件")
        pdf_docs = st.file_uploader(
            "上传 PDFs 并点击  'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                st.write(raw_text)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
# todo 上传的问题解决 streamlit run D:\wechat-agant\ask-multiple-pdfs\app.py --server.enableXsrfProtection=false
