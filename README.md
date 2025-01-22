# Mildew Detection in Cherry Leaves
Live version [here](https://mildew-detector-in-leaves-1c41d59dd8ae.herokuapp.com/)

## Business Requirements

The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes on each tree, taking a few samples of tree leaves and verifying visually if the leaf is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual inspection process.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

### Key Business Requirements:
- 1  The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
- 2  The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

## Hypothesis and how to validate?

### Hypothesis 1:
There are distinguishable visual patterns between healthy leaves and leaves infected with powdery mildew.
- **Validation:** We will use image analysis techniques such as averaging, variability, and difference images to identify potential visual patterns. Comparing these visual patterns will allow us to visually differentiate the two classes of leaves.

### Hypothesis 2:
A machine learning model that can predict whether a leaf is healthy or infected with high accuracy.
- **Validation:** We will build a classification model using a dataset of labeled cherry leaf images, healthy vs powdery mildew, and measure its accuracy and loss on predictions.

### Business Requirement 1: 
The client is interested in conducting a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.

**Rationale:** To create the visual study we are going to generate several visualizations:
- **Average Image:** Creating average images of healthy and infected leaves to identify general patterns.
- **Variability Images:** Displaying the differences between each category to highlight consistent features.
- **Difference Image:** Creating a difference image between healthy and powdery mildew leaves to highlight the areas of the image that differ the most.

These visualizations will help answer the business question of whether healthy and infected leaves are visually distinct, and if so, how.

### Business Requirement 2: 
The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

**Rationale:** Develop a Machine Learning model that can automatically classify images of healthy or powedered mildew leaves. This involves:
- **Data Preprocessing:** Resizing, and augmenting the leaf images to ensure they are suitable for machine learning.
- **Model Development:** Building a Convolutional Neural Network (CNN) to classify the images.
- **Model Evaluation:** Evaluate the model performance. The agreed performance with the client is an accuracy of at least 97%.

## ML Business Case

**Current Process:** The company currently spends around 30 minutes per tree inspecting leaves manually. This process is not scalable for large farms, leading to increased labor costs and inefficiencies.

**Proposed Solution:** The ML model will provide real-time detection of powdery mildew from images of cherry leaves, reducing the time required for inspection and allowing the company to scale the inspection process.

**Key Benefits:** 
- **Cost Reduction:** Automated leaf inspection will save significant time and labor costs.
- **Improved Efficiency:** The process will become scalable across large cherry plantations.
- **Product Quality:** The system will ensure better quality control by identifying and treating powdery mildew in its early stages.

## Dashboard Design

### Project Summary Page:
- **Content:**
    - Overview of the project, including the business requirements and dataset description.
    - Key findings from the visual study of healthy vs. powdery mildew leaves.


### Visual Differentiation Study Page:
- **Content**
    - Average images, variability images, and difference images between healthy and infected leaves.
    - Image montage for both categories to give a holistic view of the dataset.
    - Interactive buttons or checkboxes to toggle between different visual analysis types.


### Prediction System Page:
- **Content:**
    - File uploader widget: Allows users to upload multiple cherry leaf images.
    - Predictions: The dashboard displays the uploaded image with a prediction of whether the leaf is healthy or infected. It also provides the probability score for each prediction.
    - Table: Displays a list of uploaded images along with their prediction results.
    - Download button: Allows users to download the table of prediction results.

### Project Hypothesis and Validation Page:
- **Content:**
    - List the project hypotheses and how each was validated.
    - Provides visual and statistical evidence supporting the differentiation of leaves and the performance of the ML model.

### Model Performace Page:
- **Content:**
    - Performance metrics of the ML model as learning curve, average accuracy and loss in predictions.
    - Insights into any improvements or optimizations made during the model development process.

## Ethical and Privacy Considerations
- **Data Confidentiality:** The dataset provided by Farmy & Foods is under an NDA, meaning it cannot be shared outside the project team.
- **Data Usage:** The data will only be used for the purposes of this project, and any outputs (such as visualizations or models) will not expose sensitive or identifying information. All training and test datasets won't be shared in project repo.

## Technical Considerations
- **Image Size:** The original dataset consists of 256x256 pixel images. To balance model performance and file size, we may consider resizing images to 100x100 or 50x50 pixels. This will help maintain the model's size under 100Mb, ensuring smooth integration with GitHub.
- **Model Type:** A binary classification model will be used to predict whether a cherry leaf is healthy or infected. For this project a Conventional Neural Network will be used.
- **Performance Goal:** The model is expected to achieve a minimum accuracy of 97%, as agreed upon with the client.

## Dashboard Features

### Project Summary Page
- **Project Overview:** Displays a brief summary about powdery mildew, its impact on cherry plantations, and the business need to detect this disease on leaves.
- **Data Information:** Offers details about the dataset, including the number of images and the categories (healthy or powdery mildew).
- **Business Requirements:** Lists the project’s objectives, including the need to visually differentiate healthy leaves from those with mildew and the prediction task.
- **Reference Link:** A link to a detailed README file hosted on GitHub for further information.

![image of the summary page](/static/images/project-summary-page.jpg)

### Leaves Visualizer Page
- **Image Comparisons:** Offers checkboxes for displaying various visual comparisons between healthy and powdery mildew leaves:
 - **Average and Variability Image:** Shows a comparison between the average and variability of healthy and mildew-affected leaves.

![image of average and variability image](/static/images/differences-between-average-healthy-and-powdery-mildew-leaves.jpg)

 - **Difference between Average Leaves:** Highlights subtle differences between healthy and mildew-affected leaves.

![image of Difference between Average Leaves](/static/images/differences-between-average-healthy-and-powdery-mildew-leaves.jpg)

- **Image Montage Feature:** Allows the user to generate a montage of images for visual comparison:
 - The user selects a label (Healthy or Powdery Mildew), and the system displays a montage of sample images from the validation set. This helps visualize multiple images at once to study differences.

![image of the image montage of healthy leaves](/static/images/image-montage-healthy-leaves.jpg)

![image of the image montage of infected leaves](/static/images/image-montage-mildew-leaves.jpg)


### Mildew Detection Page
- **Instructions for Prediction:** Explains the prediction process, including how to download cherry leaf samples for testing.
- **File Uploader:** A widget enabling the user to upload multiple leaf images for analysis.
Image Display & Predictions, for each uploaded image, it displays:
  - The image itself
  - The dimensions of the image
  - The prediction result (whether healthy or affected by mildew)
- **Downloadable Report:** A table summarizing the predictions for all uploaded images, with an option to download the report as a CSV file.

![image of the mildew detector](/static/images/mildew-detector.jpg)

![image of download report](/static/images/download-report-mildew-detector.jpg)

### Hypothesis Page
 - **Main Hypothesis:** The hypothesis focuses on visual differences between healthy leaves and those with powdery mildew, such as color, texture, and pigmentation patterns.
 - **Visual Comparisons:** Discusses the lack of significant visual difference between the two leaf types, which challenges the hypothesis.
 - **Machine Learning Task:** Mentions how the Machine Learning (ML) model is built to help detect these differences, achieving 97.13% accuracy, surpassing the business requirement.
 
![image of the hypothesis page](/static/images/project-hypotesis.jpg)

### ML Performance Page
 - **Labels Distribution:** A plot showing the distribution of leaf labels in the train, validation, and test sets, highlighting the balance of data.
 - **Model Training Progress:** Displays the loss and accuracy achieved during training and testing phases, with insights into the improvements made.
 - **Loss & Accuracy Curve:** Provides visual evidence of the model's performance, revealing an optimal balance between training and testing loss.


## Manual Testing

### Multipage Navigation
- **Multipage Navigation:** Successfully navigates between pages without any loading errors or crashes.

### Project Summary Page
- **Reference Link:** Successfully takes the user to the new page.

### Leaves Visualizer Page
- **Average and Variability Image:** When the checkbox is clicked, it successfully loads the description of the average and variability image, and displays both the Average and Variability images for healthy and powdery mildew leaves.
- **Difference between Average Leaves:** When the checkbox is clicked, it successfully loads the description of the differences between Average and Variability images, and displays the difference image.
- **Image Montage Feature:**
  - When the checkbox is clicked, it successfully shows the description text, the label selection with two options: Healthy and Powdery Mildew, and the "Create Montage" button.
  - After clicking "Create Montage," it successfully generates a set of random images based on the selected label.
  - Creating a new montage with the same label creates a fresh set of random leaves, rather than repeating the previously generated set, successfully refreshing the montage.
  - When a montage is generated and the label is changed, the previously generated montage is automatically removed.

### Mildew Detection Page
- **Download Link:** Successfully takes the user to a Kaggle dataset, allowing users to download a set of images to test predictions.
  
  ![image of link directing to new page](/static/images/download-report-mildew-detector.jpg)

- **File Uploader:**
  - Restricts uploads to only `.png` files.
  - After uploading a set of images, it generates copies of the images and produces an evaluation report with correct predictions.
  - Successfully downloads the report as a `.csv` file, with the correct date and time.
  - Able to generate predictions for images with sizes smaller or larger than 256x256 pixels.

  ![image of small size test](/static/images/small-size-test.jpg)

  ![image of large size test](/static/images/big-size-test.jpg)

### Hypothesis Page
- All content on this page is successfully loaded. No interactive features were tested.

### ML Performance Page
- All content on this page is successfully loaded. No interactive features were tested.

## Fixed Bugs

- **Prediction of images with different sizes (other than 256x256px):** Initially, predictions failed because the code expected a 3-channel RGB image but received 4-channel images. The fix involved modifying the code to ensure that uploaded images are correctly converted to the desired RGB format and the correct dimensions before passing them to the model for prediction.

  ![image of bug](/static/images/bug-error.jpg)
  ![image of code causing the error](/static/images/code-error.jpg)
  ![image of the fixed bug](/static/images/fixed-bug.jpg)
  ![image of the corrected code](/static/images/fixed-bug-code.png)

- **Heroku Stack Version:** The app was initially set to stack 22, which did not support Python 3.8.19. The stack was changed to version 20 to resolve this issue and successfully load the app.

  ![image of Heroku version change](/static/images/heroku-change-of-version.jpg)

## Unfixed Bugs
- No other bugs were found.

## Deployment

### Heroku Deployment Steps

1. Log in to Heroku and create an app.
2. Select GitHub as the deployment method.
3. Choose the `mildew-detection-in-cherry-leafs` repository.
4. Change heroku version from terminal to stack 20. 
5. Deploy the project from the main branch.

## Main Data Analysis and Machine Learning Libraries

1. **Pandas**
   - **Purpose**: Pandas is a powerful library for data manipulation and analysis, providing data structures like DataFrames.
   - **Usage Example**: Used Pandas to create a frequency DataFrame for counting images in different sets (train, validation, test) based on their labels.
     ```python
     df_freq = pd.DataFrame([])
     for folder in ['train', 'validation', 'test']:
         for label in labels:
             df_freq = df_freq.append(
                 pd.Series(data={'Set': folder,
                                 'Label': label,
                                 'Frequency': int(len(os.listdir(my_data_dir + '/' + folder + '/' + label)))}
                           ),
                 ignore_index=True
             )
     ```

2. **NumPy**
   - **Purpose**: NumPy is a library for numerical computations, providing support for large multi-dimensional arrays and matrices.
   - **Usage Example**: Used NumPy to handle and manipulate numerical data, particularly when calculating average dimensions of resized images.
     ```python
     dim1_mean = int(np.array(dim1).mean())
     dim2_mean = int(np.array(dim2).mean())
     ```

3. **Matplotlib**
   - **Purpose**: Matplotlib is a plotting library that enables the creation of static, animated, and interactive visualizations in Python.
   - **Usage Example**: Used to create scatter plots for visualizing resized image dimensions and bar plots for the distribution of images in different sets.
     ```python
     plt.figure(figsize=(8, 5))
     sns.barplot(data=df_freq, x='Set', y='Frequency', hue='Label')
     plt.savefig(f'{file_path}/labels_distribution.png', bbox_inches='tight', dpi=150)
     plt.show()
     ```

4. **Seaborn**
   - **Purpose**: Seaborn is a statistical data visualization library based on Matplotlib, providing a high-level interface for drawing attractive statistical graphics.
   - **Usage Example**: Used Seaborn to enhance your plots with better aesthetics, such as in the scatter plots and learning curves of the model’s performance.
     ```python
     sns.set_style("whitegrid")
     losses[['loss', 'val_loss']].plot(style='.-')
     plt.title("Loss")
     plt.savefig(f'{file_path}/model_training_losses.png', bbox_inches='tight', dpi=150)
     plt.show()
     ```

5. **TensorFlow/Keras**
   - **Purpose**: TensorFlow is an open-source machine learning framework, while Keras is a high-level API for building and training deep learning models.
   - **Usage Example**: Used to build a convolutional neural network (CNN) for image classification, defining the model architecture and training it on the augmented dataset.
     ```python
     model = Sequential()
     model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     ...
     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
     ```

6. **Scikit-learn**
   - **Purpose**: Scikit-learn is a library for machine learning in Python that provides simple and efficient tools for data mining and data analysis.
   - **Usage Example**: Used Scikit-learn for splitting your dataset into training and testing sets, which is crucial for evaluating your model's performance.
     ```python
     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     ```

7. **Joblib**
   - **Purpose**: Joblib is a library for lightweight pipelining in Python, especially useful for saving and loading Python objects efficiently.
   - **Usage Example**: Used Joblib to save model evaluation results and class indices to avoid recomputation.
     ```python
     joblib.dump(value=evaluation, filename=f"outputs/v1/evaluation_results.pkl")
     ```

8. **Skilearn**
    - **Purpose**: Used to create the frontend dashboard of the project.
    - **Usage Example:** Used to create the summary page of the dashboard.
    ```python
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
    ```

## Content
- How to convert the image into RGB [in this article](https://www.geeksforgeeks.org/python-pil-image-convert-method/)
- How to change the version of [Heroku](https://devcenter.heroku.com/articles/upgrading-to-the-latest-stack#rolling-back)

## Credits

- From walkthrough [Maleria Detector Project](https://github.com/Luisg882/Malaria-Detector) helped to structure the project coding sequences to achieve the business requirements.

### Media

- The logo was taken from [Flavicon](https://www.flaticon.com/free-icon/leaves_6959474)

