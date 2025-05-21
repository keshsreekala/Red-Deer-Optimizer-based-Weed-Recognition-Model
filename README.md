# Red Deer Optimizer with Deep Learning Driven Weed Recognition Model Using Agricultural Image Analysis
Red Deer Optimizer with Deep Learning Driven Weed Recognition Model Using Agricultural Image Analysis. The project combines image data and text data to train a model that can classify images as either "weed" or "non-weed". It provides an end-to-end pipeline, including image preprocessing, text preprocessing, data combination, model training, evaluation.


# Note 
The dataset in a raw dataset without any preprocessing being done before on it. Due to which the images had to be labelled manually and may be prone to human error. This affected the model accuracy and maybe the predictions too. Due to which the model accuracy was 54% on training as well as validation sets. After fine-tuning and hyperparameter tuning and some more data preprocessing the accuracy may subjected to increase.


# Dataset
The dataset used in this project consists of images and corresponding labels indicating whether the image contains a weed or not. The images are stored in the train_images directory, and the labels are provided in the labels.csv file.



# Preprocessing
Image Preprocessing : 
The image preprocessing includes resizing the images to a fixed size (e.g., 224x224 pixels) and converting them to an array format suitable for feeding into the CNN model. The cv2 library is used to read and preprocess the images.
The images are resized to a fixed size, typically to a square shape, such as 224x224 pixels. Resizing the images to a consistent size is important to ensure that all images have the same dimensions, as CNN models require inputs of fixed dimensions.
The cv2 (OpenCV) library is utilized for reading and preprocessing the images. OpenCV provides a wide range of image processing functions that can be used to manipulate and transform images. In this case, it is used to read the images from the file system and perform resizing.By performing these preprocessing steps, the images are prepared in a standardized format suitable for input into the CNN model. This ensures that the model receives consistent input data and allows it to learn meaningful patterns and features for weed detection.




# Model Architecture

Input Layer: Agricultural RGB images from the DeepWeeds dataset.

Feature Extraction: Utilizes ShuffleNet for fast and efficient feature extraction with low computational cost.

Attention Mechanism: Applies Convolutional Block Attention Module (CBAM) to enhance relevant spatial and channel features.

Hyperparameter Optimization: Employs Red Deer Optimizer (RDO) to fine-tune model parameters for improved accuracy.

Classifier: Uses Bi-Directional Long Short-Term Memory (Bi-LSTM) to capture spatial context and sequential dependencies.

Output Layer: Softmax activation for binary classification (Weed vs. Crop).




# Usage
To use this project, follow these steps:

Set Up Environment
Install Python 3.10 and create a virtual environment using Anaconda or venv.
conda create -n rdodl python=3.10
conda activate rdodl

Install Dependencies
Install the required libraries:
pip install torch torchvision numpy opencv-python matplotlib seaborn scikit-learn

Download Dataset
Obtain the DeepWeeds dataset from Kaggle and place it in the data/ directory.

Run Preprocessing
Execute the preprocessing script to clean, resize, and augment images:
python preprocess.py

Train the Model
Run the main training script which uses ShuffleNet + CBAM, tuned with Red Deer Optimizer, and classifies with BiLSTM:
python train_rdodl_wrmair.py

Evaluate the Model
Generate evaluation metrics and confusion matrix after training:
python evaluate.py

Visualize Results
Use included notebooks or scripts (e.g., plot_metrics.py) to visualize training loss, accuracy, and other performance graphs.




# Dependencies
This project requires the following dependencies:

The implementation of the RDODL-WRMAIR model was conducted using the Python 3.10 programming language due to its extensive support for deep learning and scientific computing. 
The model development and training were performed in the PyTorch deep learning framework, chosen for its dynamic computation graph and ease of model customization. 
Additional libraries such as NumPy for numerical operations, OpenCV for image processing, Matplotlib and Seaborn for visualization, and scikit-learn for auxiliary metrics and data preprocessing were utilized. 
The code was developed and executed in Visual Studio Code (v1.80), a lightweight and extensible IDE with support for Python and Git integration. 
The environment was managed using Anaconda to ensure package compatibility and reproducibility.




# Conclusion
The RDODL-WRMAIR model provides an accurate and efficient deep learning solution for automated weed detection using agricultural images. Its modular design ensures high performance and adaptability across diverse field conditions.

Feel free to contribute to this project by opening issues, suggesting improvements, or submitting pull requests. Happy coding!
