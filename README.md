# Mildew Detection in Cherry Leaves

## Business Requirements

The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

### Key Business Requirements:
- 1  The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
- 2  The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

## Hypothesis and how to validate?

### Hypothesis 1:
There are distinguishable visual patterns between healthy leaves and leaves infected with powdery mildew.
- **Validation:** We will use image analysis techniques such as averaging, variability, and difference images to identify potential visual patterns. Comparing these visual patterns will allow us to visually differentiate the two classes of leaves.

### Hypothesis 2:
A machine learning model that can predict wheter a leaf is healthy or infected with powdery mildew with powdery mildew with high accuracy.
- **Validation:** We will build a classification model using a dataset of labeled cherry leaf images, healthy vs powdery mildew, and measure its accuracy and loss on predictions.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

### Business Requirement 1: The client is interested in conducting a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.

**Rational:** To create the visual study we are going to generate several visualizations:
- **Average Image:** Creating average images of healthy and infected leaves to identify general patterns.
- **Variability Images:** Displaying the differences between each category to highlight consistent features.
- **Difference Image:** Creating a differene image between healthy and powdery mildew leaves to highlight the areas of the image that differ the most.

These visualizations will help answer the business question of whether healthy and infected leaves are visually distinct, and if so, how.

### Business Requirement 2: The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

**Rationales:** Develop a Machine Learning model that can automatically classify images of healthy or powedery mildew leaves. This involves:
- **Data Preprocessing:** Resizing, and augmenting the leaf images to ensure they are suitable for machine learning.
- **Model Development:** Building a Convolutional Neural Network (CNN) to classify the images.
- **Model Evaluation:** Evaluate the model performance. The agreed performace with the client is a accuracy of at least 97%.

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

### Visual Differentioation Study Page:
- **Content**
    - Average images, variability images, and difference images between healthy and infected leaves.
    - Image montage for both categories to give a holistic view of the dataset.
    - Interactive buttons or checkboxes to toggle between different visual analysis types.

### Prediction SYStem Page:
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

## Ethical and Pruvacy Considerations
- **Data Confidentiality:** The dataset provided by Farmy & Foods is under an NDA, meaning it cannot be shared outside the project team.
- **Data Usage:** The data will only be used for the purposes of this project, and any outputs (such as visualizations or models) will not expose sensitive or identifying information. All training and test datasets won't be shared in project repo.

## Technical Considerations
- **Image Size:** The original dataset consists of 256x256 pixel images. To balance model performance and file size, we may consider resizing images to 100x100 or 50x50 pixels. This will help maintain the model's size under 100Mb, ensuring smooth integration with GitHub.
- **Model Type:** A binary classification model will be used to predict whether a cherry leaf is healthy or infected. For this project a Conventional Neural Network will be used.
- **Performance Goal:** The model is expected to achieve a minimum accuracy of 97%, as agreed upon with the client.



## Unfixed Bugs

- You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment

### Heroku

- The App live link is: `https://YOUR_APP_NAME.herokuapp.com/`
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

- Here, you should list the libraries used in the project and provide an example(s) of how you used these libraries.

## Credits

- In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism.
- You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

### Media

- The photos used on the home and sign-up page are from This Open-Source site.
- The images used for the gallery page were taken from this other open-source site.

## Acknowledgements (optional)

- Thank the people who provided support throughout this project.
