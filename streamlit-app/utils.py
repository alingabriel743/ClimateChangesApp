import streamlit as st

def show_logout():
    if st.session_state['logged_in']:
        col1, col2 = st.columns([0.9, 0.])  # Adjust columns ratio based on your layout
        with col2:  # This places the button towards the right
            if st.button("Logout", key="logout_button"):
                st.session_state['logged_in'] = False
                st.experimental_rerun()