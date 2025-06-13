# NLP-Disaster-Classifier
This repository contains a Python script developed for the Kaggle competition "Natural Language Processing with Disaster Tweets." The primary objective is to build a machine learning model capable of distinguishing between tweets that describe real disasters and those that use disaster-related language metaphorically or ironically.

In times of crisis, Twitter has become a vital communication channel. However, ambiguity in language can make it difficult to programmatically identify genuine emergencies. This project proposes an automated solution for classifying these tweets, which can be of great utility for humanitarian aid organizations and news agencies in their real-time event monitoring.


# Features
- **Robust Text Preprocessing:** Implementation of a text cleaning and normalization pipeline that includes URL removal, lowercasing, punctuation removal, and extensive lemmatization (for nouns, verbs, and adjectives) to optimize language representation.
- **DistilBERT Modeling:** Utilizes DistilBERT, a pre-trained transformer model from keras_nlp library, which captures complex contextual relationships in text and offers state-of-the-art performance in text classification tasks.
- **Training Optimization:** The model is trained with an Adam optimizer and benefits from callbacks such as EarlyStopping (to prevent overfitting) and ReduceLROnPlateau (to dynamically adjust the learning rate and improve convergence).
- **Detailed Performance Evaluation:** The script calculates and displays the F1-Score, a key metric for classification problems where precision and recall are equally important, especially in datasets with potential class imbalances.
- **Output Generation:** Generates the submission.csv prediction file in the format required by Kaggle and a training_history.png plot to visualize model training progress.


# Technologies Used
All interactions and processing are performed using standard Python libraries for data science and deep learning:
- **Python 3.10**
- **Pandas:** For tabular data manipulation and analysis.
- **NumPy:** For efficient numerical operations.
- **TensorFlow & Keras:** For building, training, and evaluating the deep learning model.
- **KerasNLP:** Integration of cutting-edge Natural Language Processing models, specifically DistilBERT.
- **NLTK (Natural Language Toolkit):** For text preprocessing tasks like tokenization and lemmatization.
- **Scikit-learn:** For splitting the dataset into training and validation sets.
- **Matplotlib:** For visualizing the model's training history.


# Prerequisites
Before running this script, make sure you have the following:

- **Python 3.10:** Confirm that Python is installed on your system.
- **Python Libraries:** Install the necessary dependencies. You can do this by running:
```bash
pip install -r requirements.txt
```

- **Competition Data:** Download the train.csv, test.csv, and sample_submission.csv files from the Kaggle competition data page and place them in the same folder as the script, or update INPUT_PATH in the script to the correct path.
- **NLTK Resources:** The script uses WordNetLemmatizer and word_tokenize. Ensure you have the NLTK resources downloaded. If not, you can download them in an interactive Python environment:
```Python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

# Installation and Execution
1. Clone this repository:
```bash

git clone https://github.com/joseandres94/NLP-Disaster-Classifier.git
cd NLP-Disaster-Classifier
```

2. Install dependencies (if you haven't already in the prerequisites):
```bash

pip install -r requirements.txt
```

3. Execute the script from your terminal:
```bash

python main.py
```
The script will print its progress to the console, including data loading, preprocessing, model training, and evaluation stages.


# Output Files
Upon completion, the script will generate the following files in the output folder (defined by OUTPUT_PATH, which defaults to the same directory as the script if run locally):

- **submission.csv:** This file will contain the binary predictions (0 for "not disaster," 1 for "disaster") for the test dataset, ready for submission to Kaggle.
- **training_history.png:** A plot visualizing the model's loss and accuracy on the training and validation sets across epochs, which is useful for analyzing training behavior.


# Important Considerations
- **Dataset Content:** The Kaggle competition warns that the dataset may contain text considered profane, vulgar, or offensive.
- **Reproducibility:** DistilBERT's pre-trained model weights are automatically downloaded by keras_nlp. For exact reproducibility in the future, consider using a model versioning tool if weight files were stored locally.
- **Model Limitations:** While DistilBERT is powerful, prediction quality can depend on data cleanliness, dataset diversity, and hyperparameter fine-tuning.


# Acknowledgments
This project is an implementation of a solution for the "Natural Language Processing with Disaster Tweets" competition on Kaggle. The original dataset was created by Figure Eight.


# License
Distributed under the MIT License. See the LICENSE file for more information.


# Author
José Andrés Lorenzo.
https://github.com/joseandres94
