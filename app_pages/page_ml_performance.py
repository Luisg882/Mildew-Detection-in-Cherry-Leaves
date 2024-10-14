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

    st.write(
        f"### Model Learning Curve\n\n"
        f"The training accuracy starts high and slightly improves over epochs, while loss decreases "
        f"over epochs. Validation accuracy fluctuates, but remains consistently high "
        f"(~98-99%) after the initial epochs, while the loss shows the same pattern of flutuation "
        f"but decreases, suggesting the model learns well without significant overfitting.\n\n"
        )
    col1, col2 = st.beta_columns(2)
    with col1:
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write(
        f"### Generalised Performance on Test Set\n\n"
        f"With a very low test set loss and a high test accuracy, the model generalizes well "
        f"to unseen data. This is an excellent result, suggesting the model is highly accurate in identifying"
        )
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
