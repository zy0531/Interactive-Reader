import os
import streamlit as st
# load api
from dotenv import load_dotenv, find_dotenv
# embed text
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
# chat model
from langchain.chat_models import ChatOpenAI
# prompt chain
from langchain.chains import ConversationalRetrievalChain
# create database for pdf embeddings
from langchain.vectorstores import Chroma
# load pdf
from langchain.document_loaders import PyPDFLoader
# extract sections of the pdf file
from PyPDF2 import PdfReader, PdfWriter
# create temporary pdf file
from tempfile import NamedTemporaryFile
# encoding binary files as text, display it on web
import base64
# 
from htmlTemplates import expander_css, css, bot_template, user_template

# load api key
dotenv_path = find_dotenv()
if not dotenv_path:
    st.error(".env file not found. Please ensure it is in the correct location.")
    st.stop()

load_dotenv(dotenv_path)
print(dotenv_path)

openai_api_key = os.getenv('OPENAI_API_K')
huggingfacehub_api_token= os.getenv('HUGGINGFACEHUB_API_TOKEN')

# process pdf
def process_file(doc):
    # embedding object
    model_name = "thenlper/gte-small"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)  
    # vector store
    pdfsearch = Chroma.from_documents(doc, embeddings)
    # conversation chain
    # 
    chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.3, openai_api_key=openai_api_key, model_name= 'gpt-4o-mini'), 
        retriever=pdfsearch.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True)
    return chain


# Task 6: Method for Handling User Input
# def handle_userinput(query):
    
#     if st.session_state.conversation is None:
#         st.error("Conversation chain is not initialized. Please process the PDF first.")
#         return
#     try:

#         response = st.session_state.conversation({"question": query, 'chat_history':st.session_state.chat_history}, return_only_outputs=True)
#         st.session_state.chat_history += [(query, response['answer'])]
#         # Debugging: Inspect the response
#         print("Debug - Response:", response)

#         st.session_state.N = list(response['source_documents'][0])[1][1]['page']
        
        
#         for i, message in enumerate(st.session_state.chat_history): 
#             st.session_state.expander1.write(user_template.replace("{{MSG}}", message[0]), unsafe_allow_html=True)
#             st.session_state.expander1.write(bot_template.replace("{{MSG}}", message[1]), unsafe_allow_html=True)


#     except Exception as e:
#         st.error(f"An error occurred: {e}")

# Task 6: Method for Handling User Input
def handle_userinput(query):
    if st.session_state.conversation is None:
        st.error("Conversation chain is not initialized. Please process the PDF first.")
        return
    try:
        response = st.session_state.conversation({"question": query, 'chat_history': st.session_state.chat_history}, return_only_outputs=True)
        st.session_state.chat_history += [(query, response['answer'])]
        print("Debug - Response:", response)

        # Capture and display page numbers
        for i, doc in enumerate(response['source_documents']):
            page_number = doc.metadata.get('page', 'N/A')
            st.session_state[f"N_{i}"] = page_number
            print('Debug -- N:', st.session_state[f"N_{i}"])
            st.session_state.chat_history[-1] = (query, f"{response['answer']} (Page {page_number})")
            
            st.session_state.N = page_number

        for i, message in enumerate(st.session_state.chat_history[::-1]):
            st.session_state.expander1.write(user_template.replace("{{MSG}}", message[0]), unsafe_allow_html=True)
            st.session_state.expander1.write(bot_template.replace("{{MSG}}", message[1]), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")



def main():
    
    # "My app"

    # page layout
    st.set_page_config(layout="wide", 
                       page_title="Interactive Reader",
                       page_icon=":books:")
    # insert css
    st.write(css, unsafe_allow_html=True)
    # initialize the conversation, chat history, pdf page number
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "N" not in st.session_state:
        print('\"N\" not in st.session_state')
        st.session_state.N = 0
    # web has two columns
    st.session_state.col1, st.session_state.col2 = st.columns([1, 1])
    # page title
    st.session_state.col1.header("Interactive Reader :books:")
    
    # Task 5: Load and Process the PDF 
    st.session_state.col1.subheader("Your documents")
    st.session_state.pdf_doc = st.session_state.col1.file_uploader("Upload your PDF here and click on 'Process'")

    
    # if st.session_state.col1.button("Process", key='a'):
        # with st.spinner("Processing"):
    if st.session_state.pdf_doc is not None:
        with NamedTemporaryFile(suffix="pdf", delete=False) as temp:
            temp.write(st.session_state.pdf_doc.getvalue())
            temp.seek(0)
            loader = PyPDFLoader(temp.name)
            pdf = loader.load()
            st.session_state.conversation = process_file(pdf)
            st.session_state.col1.markdown("Done processing. You may now ask a question.")    

        # text box
        user_question = st.session_state.col1.text_input("Ask a question on the contents of the uploaded PDF:")
        
        # scrollale area to display chat
        st.session_state.expander1 = st.session_state.col1.expander('Your Chat', expanded=False)
        st.session_state.col1.markdown(expander_css, unsafe_allow_html=True) 
        
        # Task 7: Handle Query and Display Pages
        if user_question:
            handle_userinput(user_question)
            with NamedTemporaryFile(suffix="pdf", delete=False) as temp:
                temp.write(st.session_state.pdf_doc.getvalue())
                temp.seek(0)
                reader = PdfReader(temp.name)
                
                pdf_writer = PdfWriter()
                start = max(st.session_state.N, 0)
                end = min(st.session_state.N+1, len(reader.pages)-1) 
                
                while start <= end:
                    pdf_writer.add_page(reader.pages[start])
                    start+=1
                with NamedTemporaryFile(suffix="pdf", delete=False) as temp2:
                    pdf_writer.write(temp2.name)
                    with open(temp2.name, "rb") as f:
                        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

                        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}#page={3}"\
                            width="100%" height="900" type="application/pdf frameborder="0"></iframe>'
                    
                        st.session_state.col2.markdown(pdf_display, unsafe_allow_html=True)
            
       



if __name__ == '__main__':
    main()
