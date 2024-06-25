import streamlit as st
import hashlib
import json
from pathlib import Path
from pymongo import MongoClient
from streamlit.source_util import _on_pages_changed, get_pages
from extra_streamlit_components import CookieManager
from validations import *

# MongoDB Setup
client = MongoClient("mongodb://localhost:27017/")
db = client['climate_changes']
users = db['users']

# Initialize cookie manager
cookie_manager = CookieManager()

DEFAULT_PAGE = "pages/01_🌐Caracterizare generala.py"
SECOND_PAGE_NAME = "Home"

def get_all_pages():
    default_pages = get_pages(DEFAULT_PAGE)
    pages_path = Path("pages.json")
    if pages_path.exists():
        saved_default_pages = json.loads(pages_path.read_text())
    else:
        saved_default_pages = default_pages.copy()
        pages_path.write_text(json.dumps(default_pages, indent=4))
    return saved_default_pages

def clear_all_but_first_page():
    current_pages = get_pages(DEFAULT_PAGE)
    if len(current_pages.keys()) == 1:
        return
    get_all_pages()
    key, val = list(current_pages.items())[0]
    current_pages.clear()
    current_pages[key] = val
    _on_pages_changed.send()

def show_all_pages():
    current_pages = get_pages(DEFAULT_PAGE)
    saved_pages = get_all_pages()
    for key in saved_pages:
        if key not in current_pages:
            current_pages[key] = saved_pages[key]
    _on_pages_changed.send()

def hide_page(name: str):
    current_pages = get_pages(DEFAULT_PAGE)
    for key, val in current_pages.items():
        if val["page_name"] == name:
            del current_pages[key]
            _on_pages_changed.send()
            break

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def login_user(username, password):
    if not username or not password:
        st.error("Numele de utilizator și parola sunt necesare.")
        return False 

    user = users.find_one({"username": username, "password": make_hashes(password)})
    if user:
        cookie_manager.set(cookie='logged_in', val='true', max_age=86400) 
        st.session_state['logged_in'] = True
        logged_in_cookie = cookie_manager.get('logged_in')
        if logged_in_cookie:
            print("Cookie retrieved successfully:", logged_in_cookie)
        else:
            print("Failed to retrieve cookie after setting")
        st.rerun()  
        
    else:
        st.error("Nume de utilizator sau parolă incorecte.")
        return False

def logout_user():
    # Check if the cookie exists before trying to delete it
    if 'logged_in' in cookie_manager.cookies:
        cookie_manager.delete('logged_in',)
    st.session_state['logged_in'] = False
    st.session_state.clear() 
    clear_all_but_first_page()
    st.rerun()

def create_user(username, password, email):
    if users.find_one({"username": username}) or users.find_one({"email": email}):
        return False, "User already exists."
    hashed_password = make_hashes(password)
    users.insert_one({"username": username, "email": email, "password": hashed_password})
    return True, "Account created successfully. Please log in."

def verify_user(username, password):
    user = users.find_one({
        "username": username,
        "password": make_hashes(password)
    })
    return user is not None

def display_login_form():
    """Display the login form and handle user authentication."""
    with st.form("Login Form"):
        username = st.text_input("Nume de utilizator")
        password = st.text_input("Parola", type="password")
        login_button = st.form_submit_button("Login")
        if login_button:
            if verify_user(username, password):
                login_user(username, password)
            else:
                st.error("Nume de utilizator sau parolă greșite.")
    
    if not st.session_state.get('logged_in', False):
        if st.button("Nu aveți un cont? Înregistrați-vă!"):
            st.session_state['show_signup'] = True
            st.rerun()

def display_validation_errors(valid_email, valid_pass, valid_user):
    if not valid_email:
        st.error("Formatul adresei de email este invalid.")
    if not valid_pass:
        st.error("Parola trebuie să aibă cel puțin 8 caractere, o cifră și un caracter special.")
    if not valid_user:
        st.error("Numele de utilizator trebuie să fie alfanumeric, cu o lungime cuprinsă între 4 și 20 de caractere.")

def display_signup_form():
    with st.form("Signup Form"):
        new_user = st.text_input("Nume de utilizator")
        new_user_email = st.text_input("Email")
        new_password = st.text_input("Parola", type="password")
        submit_button = st.form_submit_button("Creaza cont")

        if submit_button:
            valid_email = validate_email(new_user_email)
            valid_pass = validate_password(new_password)
            valid_user = validate_username(new_user)

            if valid_email and valid_pass and valid_user:
                result, message = create_user(new_user, new_password, new_user_email)
                if result:
                    st.success(message)
       
                    st.session_state['last_action'] = "signup_successful"
                else:
                    st.error(message)
            else:
                display_validation_errors(valid_email, valid_pass, valid_user)

    if st.button("Înapoi la conectare"):
        st.session_state['show_signup'] = False
        st.rerun()

def main():
    st.title("Bine ați venit!")
    if 'logged_in' not in st.session_state:
        logged_in_cookie = cookie_manager.get('logged_in')
        st.session_state['logged_in'] = (logged_in_cookie == 'true')

    if st.session_state['logged_in']:
        st.markdown("""
        În cadrul acestei aplicații, ne propunem să explorăm și să analizăm impactul schimbărilor climatice la nivelul României. Aplicația servește ca suport pentru lucrarea mea de disertație, intitulată "Aplicație web pentru studiul schimbărilor climatice la nivelul României", și este structurată în mai multe secțiuni care reflectă diferitele faze și metodologii ale cercetării noastre.

        ### Metodologia de colectare a datelor
        Datele au fost obținute folosind o strategie duală:
        - **Date cu parametri atmosferici** (cantitativi): Acestea includ temperatura, presiunea atmosferică, umiditatea relativă și structura chimică a aerului la momentul t;
        - **Fenomene meteorologice** (calitative): Acestea cuprind evenimentele meteorologice de la momentul t mai sus menționat.

        Colectarea datelor a implicat tehnici de web scraping, folosind framework-ul Scrapy pentru a extrage informații din articole publicate online, îmbogățind astfel setul de date cu perspective calitative asupra evenimentelor meteorologice.

        ### Descrierea meniurilor și analizelor implementate
        Aplicația include mai multe meniuri care facilitează analiza datelor:
        - **Caracterizare generală**: Vom folosi hărți interactive realizate cu Folium pentru a oferi o perspectivă generală asupra schimbărilor climatice observate.
        - **Analiza exploratorie de date**: Aceasta secțiune permite o investigație detaliată a seturilor de date, identificând tendințe și modele.
        - **Analiza de cluster**: Folosind algoritmii K-means și K-prototypes, vom clasifica datele pentru a descoperi grupuri semnificative.
        - **Detecția anomaliilor**: Vom implementa metodele de detecție a anomaliilor Autoencoders (AE) și Isolation Forest (IF) pentru a identifica instanțe ce reprezintă anomalii.

        ### Scop
        Scopul principal al acestei aplicații este de a oferi o platformă interactivă care să faciliteze înțelegerea fenomenelor climatice și să contribuie la cercetarea schimbărilor climatice la nivel regional în România. Prin analize detaliate și vizualizări intuitive, dorim să oferim o bază solidă pentru decizii informate în domeniul meteorologiei și climatologiei.
    """)
        if st.button("Deconectare"):
            logout_user()  
        show_all_pages() 
    else:
        if st.session_state.get('show_signup', False):
            display_signup_form()
        else:
            display_login_form()
        clear_all_but_first_page() 

if __name__ == "__main__":
    main()
