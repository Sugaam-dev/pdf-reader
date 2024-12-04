import openai
import streamlit as st

# Load environment variables
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=openai_api_key)

def query_refiner(conversation, query):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}
        ]
    )
    return response.choices[0].message.content  # Correct way to access content

def find_match(input_text, model, index):
    input_em = model.embed_query(input_text)
    result = index.query(vector=input_em, top_k=2, include_metadata=True)  # Use keyword arguments
    
    # Check if there are enough matches
    if 'matches' in result and len(result['matches']) > 0:
        # Collect matches
        matched_texts = [match['metadata']['text'] for match in result['matches'][:2]]  # Get up to 2 matches
        return "\n".join(matched_texts)
    else:
        return "No relevant matches found."

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string
