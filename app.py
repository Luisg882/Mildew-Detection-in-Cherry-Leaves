import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_project_summary import page_project_sumary_body
from app_pages.page_leaves_visualizer import page_leaves_visualizer_body


app = MultiPage(app_name="Mildew Detector")  # Create an instance of the app

# Add your app pages here using .add_page()

app.add_page("Project Summary", page_project_sumary_body)
app.add_page("Leaves Visualizer", page_leaves_visualizer_body)

app.run()