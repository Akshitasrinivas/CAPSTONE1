from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import pickle
import sqlite3
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
import streamlit as st
from langchain.callbacks import get_openai_callback
import tempfile

def apply_custom_css():
    st.markdown(
        """
        <style>
            body {
                background-color: #f4f4f4;
                font-family: 'Arial', sans-serif;
            }
            .st-ax {
                background-color: #dff0d8;
                padding: 3px;
                border-radius: 10px;
                color: black;
            }
            .userbox {
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 10px;
                margin: 10px 0;
                background-color: #dff0d8;
                color: #3c763d;
            }
            .responsebox {
                border: 2px solid #2196F3;
                border-radius: 10px;
                padding: 10px;
                margin: 10px 0;
                background-color: #d9edf7;
                color: #31708f;
            }
            .sidebar .sidebar-content {
                background-color: #f8f9fa;
            }
            h1 {
                color: #4a4a4a;
            }
            .st-bb {
                border-bottom: 1px solid #e6e6e6;
            }
            .st-bb .st-bv {
                padding-top: 15px;
                padding-bottom: 15px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def database():
    conn = sqlite3.connect('bot.db')
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(chat_bot)")
    columns = [info[1] for info in cursor.fetchall()]
    if "user_input" in columns and "chatbot_id" not in columns:
       
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS new_chat_bot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chatbot_id TEXT,
                user_input TEXT,
                chatbot_response TEXT
            )
        ''')
        conn.commit()

        cursor.execute('''
            INSERT INTO new_chat_bot (user_input, chatbot_response)
            SELECT user_input, chatbot_response FROM chat_bot
        ''')
        conn.commit()

        cursor.execute("DROP TABLE chat_bot")
        cursor.execute("ALTER TABLE new_chat_bot RENAME TO chat_bot")
        conn.commit()

    else:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_bot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chatbot_id TEXT,
                user_input TEXT,
                chatbot_response TEXT
            )
        ''')
        conn.commit()

    return conn, cursor


def file_vector_store(pdf_file):
    loader = PyPDFLoader(file_path=pdf_file)
    data = loader.load()

    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.from_documents(data, embeddings)
    return vectordb

def save_vector_store(vectordb, filename="store.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(vectordb, f)

def load_vector_store(filename="store.pkl"):
    with open(filename, "rb") as f:
        vectordb = pickle.load(f)
    return vectordb

prompt_template="""Given the context provided in the following pdf file,
                       generate an answer based solely on this context. In your response, 
                       aim to include as much text as possible from the "answer" 
                       section in the source document context without any errors.
                       If the answer is not present in the provided context,
                       please respond with "I don't know." Do not attempt to create an answer.


                CONTEXT: {context}
                QUESTION: {question}
                """
    
PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

def get_qa_chain(mybot,vectordb):


    chain=RetrievalQA.from_chain_type(llm=mybot,
                                  chain_type="stuff",
                                  retriever=vectordb.as_retriever(),
                                  input_key="query",
                                  return_source_documents=True,
                                  chain_type_kwargs={"prompt":PROMPT})
    
    return chain


def main():

    apply_custom_css()

    st.title(" CHATBOT🤖!!")
    st.sidebar.header("Chatbot Settings")

    conn, cursor = database()

    if 'chatbots' not in st.session_state:
        st.session_state.chatbots = {}

    new_id = st.sidebar.text_input("Enter new chatbot ID:")
    if st.sidebar.button("Add Chatbot"):
        if new_id not in st.session_state.chatbots:
            st.session_state.chatbots[new_id] = {}

    selected_chatbot_id = st.sidebar.selectbox("Select a chatbot:", list(st.session_state.chatbots.keys()))

    if st.sidebar.button("Delete Selected Chatbot"):
        if selected_chatbot_id and selected_chatbot_id in st.session_state.chatbots:
            cursor.execute("DELETE FROM chat_bot WHERE chatbot_id = ?", (selected_chatbot_id,))
            conn.commit()

            del st.session_state.chatbots[selected_chatbot_id]
            st.experimental_rerun() 

    if selected_chatbot_id:
        chatbot(selected_chatbot_id, conn, cursor)

    conn.close()

def chatbot(selected_id, conn, cursor):
    st.sidebar.markdown(f"## Chatbot: {selected_id}")

    load_dotenv('key.env')

    if selected_id not in st.session_state.chatbots:
        st.session_state.chatbots[selected_id] = {"model": None, "temperature": 0.7, "max_tokens": 500, "vectordb": None}

    chatbot = st.session_state.chatbots[selected_id]

    chatbot['model'] = st.sidebar.radio(f"Select model for {selected_id}", ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-0301", "gpt-4-0314"], key=f"model_{selected_id}")
    chatbot['temperature'] = st.sidebar.slider(f"Select temperature for {selected_id}", min_value=0.20, max_value=1.00, value=0.40, step=0.20, key=f"temperature_{selected_id}")
    chatbot['max_tokens'] = st.sidebar.slider(f"Select max_tokens for {selected_id}", min_value=200, max_value=1200, value=400, step=200, key=f"max_tokens_{selected_id}")

    if 'file_name' not in chatbot:
        chatbot['file_name'] = None

    uploaded_file = st.sidebar.file_uploader(f"Upload a PDF document for {selected_id}", type="pdf", key=f"file_uploader_{selected_id}")
    if uploaded_file is not None:
        file_name = uploaded_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        chatbot['vectordb'] = file_vector_store(tmp_file_path)
        save_vector_store(chatbot['vectordb'])
        chatbot['file_name'] = file_name

    if chatbot['file_name']:
        st.sidebar.markdown(f"Uploaded file: `{chatbot['file_name']}`")

    mybot=ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model=chatbot['model'], temperature=chatbot['temperature'], max_tokens=chatbot['max_tokens'])

    user_input = st.text_input("You:")
    prompt = f"{user_input}\n"

    if os.path.exists("store.pkl"):
        vectordb = load_vector_store()

        st.sidebar.subheader("Tokens:")
        with get_openai_callback() as cb:
            chain = get_qa_chain(mybot,vectordb)
            answer = chain(prompt)
        st.sidebar.write(cb)
        answer=answer['result']

        cursor.execute("INSERT INTO chat_bot (chatbot_id, user_input, chatbot_response) VALUES (?, ?, ?)",(selected_id, prompt, answer))
        conn.commit()
        message_placeholder = st.empty()
        message_placeholder.markdown(f"<div class='userbox'>Chatbot {selected_id}: {answer}</div>", unsafe_allow_html=True)

    st.header("Chat History")
    if st.button("🗑️"):
        cursor.execute('DELETE FROM chat_bot WHERE chatbot_id = ?', (selected_id,))
        conn.commit()
    history = cursor.execute("SELECT user_input, chatbot_response FROM chat_bot WHERE chatbot_id = ?", (selected_id,)).fetchall()
    for user_input, chatbot_response in history[::-1]:
        st.markdown(f"<div class='userbox'>User: {user_input}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='responsebox'>Chatbot: {chatbot_response}</div>", unsafe_allow_html=True)


    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)



if __name__ == "__main__":
    main()
