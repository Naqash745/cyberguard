import pandas as pd
import numpy as np
import streamlit as st
import requests
import json
import sys
import os
import colorama
from time import sleep 
import json
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import sentence_transformers
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser








st.title("CYBER-GUARD")

# loader=PyPDFLoader("guide-to-the-general-data-protection-regulation-gdpr-1-1.pdf")
# data = loader.load()
# #split the extracted data into text chunks using the text_splitter, which splits the text based on the specified number of characters and overlap
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# text_chunks = text_splitter.split_documents(data)
# #download the embeddings to use to represent text chunks in a vector space, using the pre-trained model "sentence-transformers/all-MiniLM-L6-v2"
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# # # create embeddings for each text chunk using the FAISS class, which creates a vector index using FAISS and allows efficient searches between vectors
# vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
# # Retrieve and generate using the relevant snippets of the blog.
# retriever = vector_store.as_retriever()

# from langchain_groq import ChatGroq
# GROQ_API_KEY=os.getenv("GROQ_API_KEY")
# llm = ChatGroq(
#     temperature=0,
#     model="llama3-70b-8192",
#     api_key=GROQ_API_KEY
# )

# prompt = hub.pull("rlm/rag-prompt")
# from langchain_groq import ChatGroq

# llm = ChatGroq(
#     temperature=0,
#     model="llama3-70b-8192",
#     api_key=GROQ_API_KEY
# )

# prompt = hub.pull("rlm/rag-prompt")


# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )


#######################################################################
# colorama.init()
# def type(words: str):
#     for char in words:
#         sleep(0.015)
#         sys.stdout.write(char)
#         sys.stdout.flush()
#     # print()

# url = r'https://www.virustotal.com/vtapi/v2/file/scan'
# api= os.getenv("VT_API_KEY")
#######################################################################


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Sample data

selection=st.sidebar.selectbox("Select",("Dashboard","Cyber Awareness Chatbot","Malicious File Scanner","Education Portal"))

if selection=="Dashboard":
    st.subheader("Welcome to Dashboard")
    sheet_name = 'Cyber Quiz (Responses)' # replace with your own sheet name
    sheet_id = '1QE9qW7DxaYp44RvTM0YUtpRFoe4GPt9i0WX-_OruXHM' # replace with your sheet's ID
    
    url=f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df=pd.read_csv(url,names=["Timestamp","Q1","Q2"])
    # df.values
    st.write(df.iloc[-1,1:].values)
    responses=df.iloc[-1,1:].values
    # correct_answers = ["A", "A"]
    correct=[]
    # wrong=[]
    for i,j in zip(responses,["A", "A"]):
        # st.write(i,j)
        if i==j:
            correct.append(1)
        else:
            st.write(i,j)
            correct.append(0)
    
    col1,col2,col3=st.columns(3)
    with col1:
        # Display pie chart
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie([1,0], labels=["Correct","Wrong"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")  # Equal aspect ratio ensures the pie is drawn as a circle.
        st.write("### Answer Validation Results:")
        st.pyplot(fig)






    
    st.write("Latest Cyber Attacks")
    col1,col2,col3=st.columns(3)
    with col1:
        st.markdown('''
        <a href="https://www.securityweek.com/starbucks-grocery-stores-hit-by-blue-yonder-ransomware-attack/">
            <img src="https://www.securityweek.com/wp-content/uploads/2024/01/Supply-Chain-Software-Attack.jpg" width="500" height="200" />
        </a>''',
        unsafe_allow_html=True
        )
        st.caption("Starbucks, Grocery Stores Hit by Blue Yonder Ransomware Attack")
    with col2:
        st.markdown('''
        <a href="https://www.securityweek.com/hackers-stole-1-49-billion-in-cryptocurrency-to-date-in-2024/">
            <img src="https://www.securityweek.com/wp-content/uploads/2024/01/cryptocurrency.jpeg" width="500" height="200" />
        </a>''',
        unsafe_allow_html=True
        )
        st.caption("Hackers Stole $1.49 Billion in Cryptocurrency to Date in 2024")
    with col3:
        st.markdown('''
        <a href="https://www.securityweek.com/new-google-project-aims-to-become-global-clearinghouse-for-scam-fraud-data/">
            <img src="https://www.securityweek.com/wp-content/themes/zoxpress-child/assets/img/posts/security-week-post-0.jpg" width="500" height="200" />
        </a>''',
        unsafe_allow_html=True
        )
        st.caption("New Google Project Aims to Become Global Clearinghouse for Scam, Fraud Data")




if selection=="Cyber Awareness Chatbot":
    st.subheader("Cyber Awareness Chatbot")
#     query=st.text_input("Write Query Here")
#     if st.button("Submit"):
#         st.write(rag_chain.invoke(query))
# if selection=="Malicious File Scanner":
#     st.subheader("Malicious File Scanner")
#     file=st.file_uploader("Select a File")
#     if file!=None and st.button("Analyze"):     
#         with open(file.name, mode='wb') as w:
#                 w.write(file.getvalue())
        
#         file_to_upload = {"file": open(file.name, "rb")}
        
#         response = requests.post(url,files = file_to_upload , params=params)
#         file_url = f"https://www.virustotal.com/api/v3/files/{(response.json())['sha1']}"
        
#         headers = {"accept": "application/json", "x-apikey": api}
#         type(colorama.Fore.YELLOW + "Analysing....")
        
#         response = requests.get(file_url,headers=headers)
        
#         report = response.text
#         report = json.loads(report)
#         # json_string = json.dumps(report)
        
#         st.write(response)

if selection=="Education Portal":
    st.subheader("Welcome to Education Portal")
    url="https://docs.google.com/forms/d/e/1FAIpQLSeRVC8WVGSqDHN5B9_kX18RbRnS0gOFyMbKYZFqzBSGyP5rLA/viewform?usp=header"
    st.write("Test Your Cyber Knowledge [Here](https://docs.google.com/forms/d/e/1FAIpQLSeRVC8WVGSqDHN5B9_kX18RbRnS0gOFyMbKYZFqzBSGyP5rLA/viewform?usp=header)")
    
# st.markdown("Test your Knowledge" %url)
     
