# app.py
import streamlit as st
import requests
import time
import yaml

TEST_MODE = False

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

general_config = config['general']
owner_port = general_config["owner_port"]
API_URL = "http://localhost:"+str(owner_port)+"/"


st.title("Watch me kill my language!")

if "last_typing" not in st.session_state:
    st.session_state["last_typing"] = 0

def on_start_typing():
    # Send "typing" event
    now = time.time()
    st.session_state["last_typing"] = now
    requests.post(API_URL + "typing_event", json={"status": "typing"})

now = time.time()
if st.session_state["last_typing"] != 0 and now - st.session_state["last_typing"] > 0.5:
    requests.post(API_URL + "typing_event", json={"status": "stopped"})
    st.session_state["last_typing"] = 0

prompt = st.text_area("Enter your prompt:", key="prompt_area", on_change=on_start_typing)
if st.button("Send"):
    payload = {"prompt": prompt}
    if TEST_MODE:
    #response = requests.post(API_URL, json=payload)
        answer = "Hello World!" #response.json()["text"]
    else:
        response = requests.post(API_URL+"generate_gui", json=payload)
        answer = response.json()["text"]
        iteration = response.json()["iteration"]
        st.write("We are currently in iteration ", iteration)
    st.write(answer)
