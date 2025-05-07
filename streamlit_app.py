import os

import streamlit as st
from PIL import Image

from utils.ee_utils import initialize_ee, vulnerability_analysis

# Set page configuration
st.set_page_config(
    layout="wide", page_title="Riesgo de inundación en La Plata, Berisso y Ensenada"
)

# Load logo
logo_path = os.path.join("assets", "logo.png")
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.sidebar.image(logo, width=150)

# Sidebar information
st.sidebar.title("¿Qué?")
st.sidebar.info(
    """
    Esta aplicación analiza el riesgo de inundación en los municipios de La Plata, Berisso y Ensenada
    basándose en población, exposición, vulnerabilidad, superficies impermeables y humedales.
    """
)

st.sidebar.title("¿Quién?")
st.sidebar.info(
    """
    Hecho por [Nissim Lebovits](https://nlebovits.github.io/)
    """
)


def app():
    # Initialize Earth Engine
    initialize_ee()

    st.title("Riesgo de inundación en La Plata, Berisso y Ensenada")

    # Run the main analysis directly without selection
    vulnerability_analysis()


if __name__ == "__main__":
    app()
