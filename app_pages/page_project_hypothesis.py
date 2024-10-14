import streamlit as st
import matplotlib.pyplot as plt

def page_project_hypothesis_body():
    st.write('### Project Hypotesis')

    st.success(
        f'Our first hypothesis was to confirm clear patterns that could differentiate'
        f" healthy to powdery mildew leaves, mainly the presence of white cover, and "
        f"damaged edges of the leafe.\n\n"
        f"We see that powdery mildew leaves lack pigmentation showing a darker green "
        f"color, and an inclination of the edges to retract to the center of the talus.\n\n"
        f"Comparing the Average Images and Variability images between healthy and "
        f"powdery mildew leaves show a wider oval shape with distributed white spots "
        f"in powdery mildew leaves. This difference alone wesn't enough to visually "
        f"differentiate both classes\n"
    )
    st.image('outputs/v1/avg_diff.png', caption='Difference between average images', use_column_width=True)

    st.success(
        f"The second hypothesis was to create a machine learning model that can predict whether a leaf is healthy or " 
        f"powdery mildew. The model was made accomplish the business target accuracy of "
        f"97%, getting a result of 99.17% and a loss of 5.1%. Resulting in a successful predictor "
        f"healthy and powdery mildew leaves, helping the business to save money in human visual checks"
        f" (more information about the ML performance can be seeng in the Machine Learning Performance page)."
    )