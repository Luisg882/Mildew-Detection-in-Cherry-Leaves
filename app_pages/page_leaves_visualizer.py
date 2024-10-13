import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random

def page_leaves_visualizer_body():
    st.write('### Leaves Visualizer')
    st.info(
        f"Here you will be able to see visual study made to visually differentiate "
        f"healthy and powdery mildew cherry leaves"
    )

    if st.checkbox("Average and variability image in both classes"):

        st.warning(
            f"The average variability image between healthy and powdery mildew leaves "
            f"shows a small difference in the shape of the leaves, with healthy ones having "
            f"a small circular shape, while powdery mildew leaves have an oval shape.\n\n"
            f"It also shows spread pigmentation in the mildew leaves around the center, "
            f"which is characteristic of the mildew fungus. Nonetheless, this difference is not "
            f"sufficient to differentiate between the two classes."
        )

        st.image('outputs/v1/avg_var_healthy.png', caption='Average Healthy Leaf', use_column_width=True)
        st.image('outputs/v1/avg_var_powdery_mildew.png', caption='Average Powdery Mildew Leaf', use_column_width=True)

    if st.checkbox("Differences between average healthy and powdery mildew leaves"):
        st.warning(
            f"With these patterns, we can't visually demonstrate the difference between the two classes."
        )

        st.image('outputs/v1/avg_diff.png', caption='Difference between average images', use_column_width=True)

    # Function to create image montage
    def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15, 10)):
        sns.set_style("white")
        labels = os.listdir(dir_path)

        # Check if label exists in the directory
        if label_to_display in labels:
            images_list = os.listdir(dir_path + '/' + label_to_display)
            if nrows * ncols <= len(images_list):
                img_idx = random.sample(images_list, nrows * ncols)
            else:
                st.error(
                    f"Decrease nrows or ncols to create your montage. \n"
                    f"There are {len(images_list)} images in your subset. "
                    f"You requested a montage with {nrows * ncols} spaces")
                return

            # Create a Figure and display images
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            for x in range(nrows * ncols):
                img = imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
                img_shape = img.shape
                axes[x // ncols, x % ncols].imshow(img)
                axes[x // ncols, x % ncols].set_title(f"Width {img_shape[1]}px x Height {img_shape[0]}px")
                axes[x // ncols, x % ncols].set_xticks([])
                axes[x // ncols, x % ncols].set_yticks([])
            plt.tight_layout()

            st.pyplot(fig=fig)
        else:
            st.error("The label you selected doesn't exist.")
            st.write(f"The existing options are: {labels}")

    if st.checkbox("Image Montage"):
        st.write("To refresh the montage, click 'Create Montage' button")
        my_data_dir = 'inputs/mildew_detection_in_cherry_leaves/cherry-leaves'
        
        # Check if validation directory exists
        validation_dir = my_data_dir + '/validation'
        if os.path.exists(validation_dir):
            labels = os.listdir(validation_dir)

            if not labels:
                st.error("No labels found in the validation directory.")
            else:
                # Format labels to be human-readable
                labels_readable = [label.capitalize().replace('_', ' ') for label in labels]
                label_mapping = dict(zip(labels_readable, labels))

                label_to_display_readable = st.selectbox(label="Select label", options=labels_readable, index=0)

                if st.button("Create Montage"):
                    selected_label = label_mapping[label_to_display_readable]
                    montage_dir_path = os.path.join(validation_dir, selected_label)
                    if os.path.exists(montage_dir_path):
                        image_montage(dir_path=validation_dir,
                                      label_to_display=selected_label,
                                      nrows=8, ncols=3, figsize=(10, 25))
                    else:
                        st.error(f"Directory {montage_dir_path} does not exist.")
        else:
            st.error(f"Validation directory {validation_dir} does not exist.")
