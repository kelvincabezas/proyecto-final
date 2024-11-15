import os
import streamlit as st
from streamlit_navigation_bar import st_navbar
import pages as pg

st.set_page_config(initial_sidebar_state="collapsed", page_title="ETL PRODUCE", page_icon="🇵🇪", layout="wide")

pages = ["Inicio", "Carga", "Dashboard", "Descripcion", "Evaluacion", "Modelo", "Documentacion"]
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "escudo.svg")
urls = {"Documentacion": "https://github.com/jersonalvr/etl_produce"}
styles = {
    "nav": {
        "background-color": "#860000",
        "justify-content": "left",
    },
    "img": {
        "padding-right": "14px",
    },
    "span": {
        "color": "white",
        "padding": "14px",
    },
    "active": {
        "background-color": "#494948",
        "color": "var(--text-color)",
        "font-weight": "normal",
        "padding": "14px",
    }
}
options = {
    "show_menu": False,
    "show_sidebar": False,
}

page = st_navbar(
    pages,
    logo_path=logo_path,
    urls=urls,
    styles=styles,
    options=options,
)

if page not in pages:
    page = "Inicio"

functions = {
    "Inicio": pg.show_inicio,
    "Carga": pg.show_carga,
    "Dashboard": pg.show_dashboard,
    "Descripcion": pg.show_descripcion,
    "Evaluacion": pg.show_evaluacion,
    "Modelo": pg.show_modelo,
}

go_to = functions.get(page)
if go_to:
    go_to()
