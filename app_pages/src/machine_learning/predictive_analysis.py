import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from PIL import Image 
from ..data_management import load_pkl_file


from tensorflow.keras.optimizers import Adam

# Define the custom Adam optimizer
class CustomAdam(Adam):
    def __init__(self, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)

    def get_config(self):
        # Get the configuration from the parent class
        config = super().get_config()
        return config


def plot_predictions_probabilities(pred_proba, pred_class):
    """Plot prediction probability results."""
    prob_per_class = pd.DataFrame(
        data=[0, 0],
        index=['Powdery Mildew', 'Healthy'],
        columns=['Probability']
    )
    prob_per_class.loc[pred_class] = pred_proba
    prob_per_class.loc[prob_per_class.index != pred_class] = 1 - pred_proba
    prob_per_class = prob_per_class.round(3)
    prob_per_class['Diagnostic'] = prob_per_class.index

    fig = px.bar(
        prob_per_class,
        x='Diagnostic',
        y='Probability',
        range_y=[0, 1],
        width=600,
        height=300,
        template='seaborn'
    )
    st.plotly_chart(fig)

def resize_input_image(img, version):
    """Reshape image to average image size of 100x100."""
    target_size = (100, 100)  # Set the target size to 100x100
    img_resized = img.resize(target_size, Image.LANCZOS)  # Resize the image
    my_image = np.expand_dims(img_resized, axis=0) / 255.0  # Normalize the image and expand dimensions

    return my_image


def load_model_and_predict(my_image, version):
    """Load and perform ML prediction over live images."""
    model = load_model(
        f"outputs/{version}/mildew_detection_model.h5",
        custom_objects={'CustomAdam': CustomAdam}  # Correctly reference CustomAdam
    )

    pred_proba = model.predict(my_image)[0, 0]
    target_map = {v: k for k, v in {'Powdery Mildew': 1, 'Healthy': 0}.items()}
    pred_class = target_map[pred_proba > 0.5]

    if pred_class == target_map[0]:
        pred_proba = 1 - pred_proba

    st.write(
        f"The predictive analysis indicates the sample leaf is "
        f"**{pred_class.lower()}**."
    )

    return pred_proba, pred_class

