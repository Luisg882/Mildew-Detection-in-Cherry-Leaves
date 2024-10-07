import streamlit as st
from app_pages.multipage import MultiPage
import sys
import os

# Assuming 'app_pages' is in the current working directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'app_pages'))

from app_pages.page_mildew_detector import page_mildew_detector_body

# load pages scripts
from app_pages.page_project_summary import page_project_sumary_body
from app_pages.page_leaves_visualizer import page_leaves_visualizer_body
from app_pages.page_mildew_detector import page_mildew_detector_body



app = MultiPage(app_name="Mildew Detector")  # Create an instance of the app

# Add your app pages here using .add_page()

app.add_page("Project Summary", page_project_sumary_body)
app.add_page("Leaves Visualizer", page_leaves_visualizer_body)
app.add_page("Mildew Detector", page_mildew_detector_body)

app.run()