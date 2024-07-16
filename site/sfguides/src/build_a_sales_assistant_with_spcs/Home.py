import jwt
import streamlit as st
from streamlit_timeline import st_timeline
from streamlit.web.server.websocket_headers import _get_websocket_headers

st.set_page_config(
    layout='wide'
    , initial_sidebar_state='expanded'
)

from tools.utils import *
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.mode.copy_on_write = True

title = "Sales Assistant Powered by AI"

def main():
    """Main page execution starts here."""
    if "token" in st.session_state:
        # prod:
        # okta_user_context = jwt.decode(st.session_state["token"]["access_token"], options={"verify_signature": False})
        # okta_sub=okta_user_context["sub"]
        # test:
        okta_sub = 'stella.yang@snowflake.com'
    else:
        st.error("Okta Token is missing from session state, please check")
        okta_sub = "None"
        st.stop()

    load_dotenv()
    session = create_session_object()

    default_q = ["<select a question>"]
    # # Initiate Session State
    if "text_input_general_q" not in st.session_state:
        st.session_state["text_input_general_q"] = default_q[0]
    if "text_select_general_q" not in st.session_state:
        st.session_state["text_select_general_q"] = default_q[0]
    if "text_input_context_q" not in st.session_state:
        st.session_state["text_input_context_q"] = default_q[0]
    if "text_select_context_q" not in st.session_state:
        st.session_state["text_select_context_q"] = default_q[0]
    if "text_input_text2image_q" not in st.session_state:
        st.session_state["text_input_text2image_q"] = default_q[0]
    if "text_select_text2image_q" not in st.session_state:
        st.session_state["text_select_text2image_q"] = default_q[0]
    # Clear the session state value for forecast page user input if user switch page
    st.session_state["text_input_general_q"] = default_q[0]
    st.session_state["text_select_general_q"] = default_q[0]
    st.session_state["text_input_context_q"] = default_q[0]
    st.session_state["text_select_context_q"] = default_q[0]
    st.session_state["text_input_text2image_q"] = default_q[0]
    st.session_state["text_select_text2image_q"] = default_q[0]

    st.title(f"{title}")
    st.write(f" Welcome to this Streamlit-based LLM AI assistant. {okta_sub}")
    st.markdown(
        """ 
        This is  your one-stop shop for obtaining customer status information. Here, you can engage in general Q&A about your account status; skip all complex data but access actionable insights, and retrieve detailed account information through Context Q&A.  
        """, unsafe_allow_html=True)
    st.warning(f" Disclaimer: For demonstration purposes at the Summit, we replaced all clients' account names and competitor names with animal names. Additionally, we adjusted point-of-contact names, key numbers, etc., based on our clients' real cases.")

if __name__ == "__main__":
    st.session_state["token"] = {"access_token" : None}
    headers = _get_websocket_headers()
    access_token = headers.get("X-Auth-Request-Access-Token")
    if access_token:
        st.session_state["token"]["access_token"] = access_token
    main()