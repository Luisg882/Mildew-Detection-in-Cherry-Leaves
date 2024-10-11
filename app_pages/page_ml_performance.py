import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))  

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from app_pages.src.machine_learning.evaluate import load_test_evaluation  

def page_ml_performance_body():
    version = 'v1'

    st.write(f"### Train, Validation, and Test Set: Labels Frequencies")
    st.image('outputs/v1/labels_distribution.png', caption='Labels Frequency', use_column_width=True)

    st.write("### Model History")
    col1, col2 = st.beta_columns(2)
    with col1:
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
