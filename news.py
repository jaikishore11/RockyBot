import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables (especially for the OpenAI API key)
load_dotenv()

# Streamlit UI components
st.title("Research Helper Bot 🔍")
st.sidebar.title("Upload Text Files")

# Upload files
uploaded_files = st.sidebar.file_uploader("Choose up to 3 text files", accept_multiple_files=True, type=["txt"])

process_files_clicked = st.sidebar.button("Process Files")
file_path = "faiss_store_text.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_files_clicked and uploaded_files:
    # Load data from text files
    main_placeholder.text("Data Loading... Started... ✅✅✅")
    texts = [file.read().decode("utf-8") for file in uploaded_files]

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter... Started... ✅✅✅")
    docs = text_splitter.split_documents(texts)
    
    # Create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_text = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building... ✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_text, f)

query = main_placeholder.text_input("Enter your question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # Display answer
            st.header("Answer")
            st.write(result["answer"])

            # Display sources if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
else:
    st.write("Please upload files and click 'Process Files' to initialize the bot.")
