import torch 
import wikipedia 
import transformers
from transformers import pipeline, Pipeline
import streamlit as st


def main():
    st.title("Demo Wikipedia App")
    st.write("Search topic, ask questions, get answers")

    topic = st.text_input("SEARCH TOPIC", "")

    #Display article paragraph:

    article_paragraph = st.empty()
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

    



if __name__ == '__main__':

    main()