import streamlit as st
import matplotlib.pyplot as plt

def page_project_hypothesis_body():
    st.write('### Project Hypotesis')

    st.success(
        f'* Our main hypothesis was to confirm clear patters that could differentiate'
        f" healty to powedery mildew leaves, mainly the presence of white cover, and "
        f"damaged edges of the leafe.\n\n"
        f"We see that powdery mildew leafes lack of pigmentation showing a darker green "
        f"color, and a inclination of the edges to retract to the center of the talus.\n\n"
        f"Comparing the Average Images and Variability images between healty and "
        f"powdery mildew leaves show a wider oval shape with distributed white spots "
        f"in powdery mildew leaves. This difference alone weren't enough to visually "
        f"differentiate bouth classes\n"
    )
    st.image('outputs/v1/avg_diff.png', caption='Difference between average images', use_column_width=True)

    st.success(
        f"* The creation of the Machine Learning model accomplish the business target of accuracy of "
        f"97%, getting a result of 99.17% and a loss of 5.1%. Resulting a a successfull predictor "
        f"healthy and powdery mildew leaves, helping the bussines to save money in human visual checks"
        f" (more information abouth the ML performace can be seing in the Machine Learning Performance page)."
    )