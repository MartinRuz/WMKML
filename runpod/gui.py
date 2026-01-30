import streamlit as st
import requests
import time
from utility import API_URL_INFERENCE, API_URL_OWNER
from pythonosc import udp_client

# This is a dummy gui, that can be used for debugging. No effort will be spent making this gui better, more intuitive or more usable.
# Known limitations include that prompts can be sent at unsafe times, that the typing detection does not work properly,
# and that CTRL-Enter does not send the prompt, contrary to what the gui indicates.
# The user must ensure proper use of this gui themselves.

st.title("Watch me kill my language!")

if "last_typing" not in st.session_state:
    st.session_state["last_typing"] = 0 #track typing times for the start and stop detection

def on_start_typing():
    # Send "typing" event
    now = time.time()
    st.session_state["last_typing"] = now
    requests.post(API_URL_OWNER + "typing_event", json={"status": "typing"})

now = time.time()
if st.session_state["last_typing"] != 0 and now - st.session_state["last_typing"] > 0.5:
    requests.post(API_URL_OWNER + "typing_event", json={"status": "stopped"})
    st.session_state["last_typing"] = 0

prompt = st.text_area("Enter your prompt:", key="prompt_area", on_change=on_start_typing)
if st.button("Send"):
    payload = {"prompt": prompt}
    response = requests.post(API_URL_OWNER+"generate_gui", json=payload)
    answer = response.json()["text"]
    iteration = response.json()["iteration"]
    st.write("We are currently in iteration ", iteration)
    st.write(answer)

# Changing the language must only be clicked once, when starting a run. This will start the training, and reset all systems.
if st.button("Wechsel auf Deutsch"):
    requests.post(API_URL_OWNER+"language", json={"language": "de"})
    requests.post(API_URL_INFERENCE+"language", json={"language": "de"})

if st.button("Change to English"):
    requests.post(API_URL_OWNER+"language", json={"language": "en"})
    requests.post(API_URL_INFERENCE+"language", json={"language": "en"})

# This method is kept for reference, what a reset function might look like. Now, a language change resets the system instead.
#if st.button("Reset"):
#    client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
#    client.send_message("/restart", "requested")
#    open(restart_flag, "w").close()
#    if not os.path.exists(restart_flag):
#        response_owner = requests.post(API_URL_OWNER + "restart_system")
#        response_inference = requests.post(API_URL_INFERENCE + "restart_system")
#        st.write("Trainer, Owner and Inference confirmed the restart, everything is reset and you are ready to go again.")
#        client.send_message("/restart", "completed")