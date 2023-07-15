from langchain.document_loaders import PyPDFLoader #pip install pypdf==3.12.1
from IPython.display import display, Markdown
import openai
import os
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv()) # read local .env file with OpenAI key
# openai.api_key = openai_api_key=os.environ['OPENAI_API_KEY']
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet')

from langchain import OpenAI
from langchain.text_splitter import TokenTextSplitter
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

# A function that will be called only if the environment's openai_api_key isn't set
def get_openai_api_key():
    input_text = st.text_input(label="OpenAI API Key (or set it as .env variable)",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text

# Start Of Streamlit page
st.set_page_config(page_title="Paper review assistant", page_icon=":robot:")

# Start Top Information
st.header("LLM Assisted Paper review")

col1, col2 = st.columns(2)

with col1:
    st.markdown("Have a manuscript to review? Takes a lot of time? This tool is meant to help you generate \
                questions for the manuscript that can help improving it.\
                \n\nThis tool is made  by [Nikos Sourlos](www.linkedin.com/in/nsourlos). \n\n View Source Code on [Github](https://github.com/nsourlos/review_manuscript_streamlit/blob/main/review_manuscript_streamlit.py)")

with col2:
    st.image(image='paper_review.jpg', width=300, caption='https://www.nature.com/articles/d41586-018-06991-0')
# End Top Information

st.markdown("## :muscle: Upload PDF documents")
	# :older_man:
# Output type selection by the user
# output_type = st.radio(
#     "Output Type:",
#     ('Interview Questions', '1-Page Summary'))

uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file.name.endswith(".pdf")==0:
#     st.write("Only accepts PDF files. Please select another file")
# else:

# # Collect information about the person you want to research
# person_name = st.text_input(label="Person's Name",  placeholder="Ex: Elad Gil", key="persons_name")
# twitter_handle = st.text_input(label="Twitter Username",  placeholder="@eladgil", key="twitter_user_input")
# youtube_videos = st.text_input(label="YouTube URLs (Use , to seperate videos)",  placeholder="Ex: https://www.youtube.com/watch?v=c_hO_fjmMnk, https://www.youtube.com/watch?v=c_hO_fjmMnk", key="youtube_user_input")
# webpages = st.text_input(label="Web Page URLs (Use , to seperate urls. Must include https://)",  placeholder="https://eladgil.com/", key="webpage_user_input")

button_ind = st.button("*Generate Output*", type='secondary', help="Click to generate review questions")
if button_ind:
    if uploaded_file is None:
        st.warning('Please provide a PDF file', icon="⚠️")
        st.stop()
    
    if uploaded_file.name.endswith('.pdf')==0:
        st.warning('Only accepts PDF files. Please select another file', icon="🚨")
        st.stop()

    if not OPENAI_API_KEY:
        st.warning('Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', icon="⚠️")
        st.stop()

    if OPENAI_API_KEY == 'YourAPIKeyIfNotSet':
        # If the openai key isn't set in the env, put a text box out there
        OPENAI_API_KEY = get_openai_api_key()




    st.write("Loading PDF...")
    manuscript_path='paper.pdf'
    # Load PDF
    loaders = [
        PyPDFLoader(manuscript_path),
    ]

    docs = []
    for loader in loaders: #Add all documents to one
        docs.extend(loader.load())
    paper=[docs[i].page_content for i in range(len(docs))] #Get only document content
    paper=''.join(paper)

    #Calculate token usage and price for CV prompt - Same as sent to OpenAI but free
    st.write("Calculating price...")
    text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0,model_name='gpt-3.5-turbo-16k')
    ind_tokens=text_splitter.split_text(paper)
    print("Total tokens:",len(ind_tokens))
    print("Price for them:",round(len(ind_tokens)*0.003/1000,4),"$")

    review_prompt='Below is a manuscript of a scientific publication. Act as a reviewer for the manuscript and provide at least 10 points for improvement. \
        These points should provide clear instructions on how to improve the manuscript. The manuscript is: '
    llm=OpenAI(openai_api_key=openai_api_key,temperature=0,model_name='gpt-3.5-turbo-16k') #Initialize LLM - 16k context length to fit the paper
    job_final=llm.predict(review_prompt+paper) #Predict response using LLM 
    display(Markdown(job_final))
    st.markdown(f"#### Output:")
    st.write(job_final)




