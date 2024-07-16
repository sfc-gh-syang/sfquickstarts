import os
import sys
import json
import streamlit as st
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# WRITE WELCOME DIRECTLY TO THE APP
title = "Experiment on Text2Image"
page_desc = """ This is a place for you to quickly generate images to express thoughts with client (powered by SDXL Lightening)"""

st.set_page_config(page_title=title
    , layout='wide'
    , initial_sidebar_state='expanded'
)

from tools.utils import *
session = create_session_object()
add_colored_header(header=f"{title}"
    , description=f"{page_desc}"
)
# Add box for prompt request
st.subheader(":balloon: Type in a short prompt to generate image")
default_text2image_q = ["<select a question>"]
preselect_text2image_q_lst = ['An astronaut riding a horse'
    , 'A young girl smiling, multi-colored'
    , 'Photo of a monkey drinking a cup of coffee'
    , 'Sketch a salesperson reading a lot of client company files in front of his laptop, pulling hair'
    , 'Sketch a salesperson reading the laptop, displaying the word "Streamlit" on the laptop screen']

# Setup the form for user question input and setup session_state variables
if "text_input_text2image_q" not in st.session_state:
    st.session_state["text_input_text2image_q"] = default_text2image_q[0]
if "text_select_text2image_q" not in st.session_state:
    st.session_state["text_select_text2image_q"] = default_text2image_q[0]
input_preselect_text2image_q = default_text2image_q + preselect_text2image_q_lst
selection_text2image = setup_form_universal(input_preselect=input_preselect_text2image_q,
                                            default_selection=default_text2image_q,
                                            question_name='_text2image_q')
# Import diffuser related packages
pipe = load_and_create_text2image_model_pipe()

if selection_text2image=='' or selection_text2image=='<select a question>':
    st.write('Please input your question to get answer.')
else:
    generator = [torch.Generator(device="cuda").manual_seed(0)]
    pipe(prompt=selection_text2image,generator=generator, num_inference_steps=8, guidance_scale=0).images[0].save("../stage/text2image/output.png")
    st.image("../stage/text2image/output.png",width=600)