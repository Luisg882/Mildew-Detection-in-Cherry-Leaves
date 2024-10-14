import streamlit as st
import matplotlib.pyplot as plt

def page_project_sumary_body():
    st.write('### Project Summary')

    st.info(
        f"**About Mildew**\n\n"
        f"Mildew is a type of fungus that appears as a white powder that consumes organic matter like plants.\n"
        f"In cherry leaves, it generates a coat preventing the sunlight from reaching the plant, obstructing photosynthesis, "
        f"compromising the tree's health, and reducing the quality of the cherries.\n\n"
        f"It not only has repercussions on tree health, but it can also cause respiratory irritation in people sensitive to it. "
        f"Our client Farmy & Foods spend approximally 30 minutes per tree to check if the tree is infected or not, in case "
        f"of infection will take an extra minute to kill the fungus. This process takes a lot of time and resources when the client "
        f"have thousands of trees to check\n\n"
        f"The purpose of this project is to visually identify healthy vs. powdery mildew leaves and create a "
        f"Machine Learning model that can classify healthy and powdery mildew leaves to save time and resources.\n\n"
        f"**Project Dataset**\n\n"
        f"A dataset of 2,104 images was used, containing healthy and powdery mildew cherry leaves."
    )

    st.write(
        f"For more information, please visit the "
        f"[Project README file](https://github.com/Luisg882/mildew-detection-in-cherry-leaves/blob/main/README.md)."
    )

    st.success(
        f"This project has two business requirements:\n\n"
        f"1 - The client is interested in a study showing how to visually differentiate a cherry leaf "
        f"that is healthy from one that contains powdery mildew.\n\n"
        f"2 - The client is interested in determining if a cherry leaf is healthy or affected by powdery mildew."
    )