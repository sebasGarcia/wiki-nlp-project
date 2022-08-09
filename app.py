import torch 
import wikipedia 
import transformers
from transformers import pipeline, Pipeline
import streamlit as st
import requests
from streamlit_lottie import st_lottie

#This demo is from https://www.youtube.com/watch?v=wVF0-ZalYmk&t=35s
def main():
    # Adding title and description
    st.title("Demo Wikipedia App")
    st.write("Search topic, ask questions, get answers")

    #Only for lottie animation as a bonus..

    # create a variable to load lottie animation using url
    lottie_animation = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_yhetm7ld.json")
    # finally to show it
    st_lottie(lottie_animation,
                                height=192,
                                width=192)

    #Display topic input slot
    topic = st.text_input("SEARCH TOPIC", "")

    #Display article paragraph:

    article_paragraph = st.empty()

    #Display question input slot
    question = st.text_input("QUESTION", "")

    if topic: 
        #load wikipedia summary of topic
        summary = load_wiki_summary(topic)

        #display article summary in paragraph 
        article_paragraph.markdown(summary)

        #perform question answering
        if question != "": 
            qa_pipeline = load_qa_pipeline()

            #answer query question using article summary
            result = answer_question(qa_pipeline, question, summary)
            answer = result["answer"]

            #display answer
            st.write(answer)

#Lottie files for animations --> Only if we have enough time 
# 1. pip install streamlit_lottie and then from streamlit_lottie import st_lottie
# 2. pip install requests and import requests
def load_lottieurl(url:str):
    """
    This function is used to show an animation on the webpage
    """
    r = requests.get(url)
    if r.status_code != 200:
        return None 

    return r.json()

#Function for load question answering pipeline
#returns a pipeline 
#Definition of a pipeline --> https://huggingface.co/docs/transformers/main_classes/pipelines

def load_qa_pipeline() -> Pipeline:
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    return qa_pipeline

def answer_question(pipeline: Pipeline, question: str, paragraph: str) -> dict:
    input = {
        "question": question, 
        "context": paragraph
    }

    output = pipeline(input)

    return output

def load_wiki_summary(query: str ) -> str:
    """
    receives a string as a query and returns a summary 
    """
    results = wikipedia.search(query)
    summary = wikipedia.summary(results[0], sentences = 10)
    return summary




#Main app engine
if __name__ == '__main__':

    main()