import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread


def page_ml_performance_body ():
    version = 'v1'

    st.write(
        f"### Train, Validation, and Test Set: Labels Frequenccies"
        )
    st.image('outputs/v1/labels_distribution.png', caption='Labels Frequency', use_column_width=True)