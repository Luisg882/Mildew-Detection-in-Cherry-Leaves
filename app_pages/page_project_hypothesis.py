import streamlit as st
import matplotlib.pyplot as plt

def page_project_hypothesis_body():
    st.write('### Project Hypotesis')

    st.success(
        f'* Our main hypothesis was to confirm clear patters that could differentiate'
        f"healty to powedery mildew leaves, mainly the presence of white cover, and "
        f"damaged edges of the leafe.\n"
        f"* We see that powdery mildel leafes lack of pigmentation showing a darker green "
        f"color, a inclination of the edges to retract to the center of the talus.\n"
        f"* Comparing the Average Images and Variability images between healty and "
        f"powdery mildew leaves show a wider oval shape with distributed white spots "
        f"in powdery mildew leaves. This difference alone weren't enough to visually "
        f"differentiate bouth classes"
    )