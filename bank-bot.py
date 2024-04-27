# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import base64
import gc
import random
import tempfile
import time
import uuid

from IPython.display import Markdown, display
import chromadb

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore

import streamlit as st

# Import the necessary module
from dotenv import load_dotenv
import os

# Load environment variables from the .env file (if present)
# Load environment variables from the .env file (if present)
load_dotenv(override=True)
docs_path = os.getenv('DOCS_PATH')
ollama_host = os.getenv('OLLAMA_HOST_URL')
vectordb_path = os.getenv('VECTOR_DB')
print(f"""
    Environment variable are
    docs_path = {docs_path}
    ollama_host = {ollama_host}
    vectordb_path = {vectordb_path}
""")


if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


llm=Ollama(model="llama3", request_timeout=120.0, base_url=ollama_host)
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
local_model_dir = "./bge_onnx"
embed_model = OptimumEmbedding(folder_name=local_model_dir)

# # load from disk
collections_name="rbi-master-directions-new"
chroma_client = chromadb.PersistentClient(path=vectordb_path)
#chroma_client.delete_collection(name=collections_name)
chroma_collection = chroma_client.get_or_create_collection(name=collections_name)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)


# Create the query engine, where we use a cohere reranker on the fetched nodes
Settings.llm = llm
query_engine = index.as_query_engine(streaming=True)

# ====== Customise prompt template ======
qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
    "Query: {query_str}\n"
    "Answer: "
    )
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Banking Bot ( using Llama-3)")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate stream of response with milliseconds delay
        streaming_response = query_engine.query(prompt)
        
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        # full_response = query_engine.query(prompt)

        message_placeholder.markdown(full_response)
        # st.session_state.context = ctx

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})