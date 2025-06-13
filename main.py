# Import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import keras_nlp
import matplotlib.pyplot as plt
import os

# Variables definition
INPUT_PATH = '/kaggle/input/nlp-getting-started/' # Original route from Kaggle.
OUTPUT_PATH = '/kaggle/working/' # Original route from Kaggle.

# Functions definition
def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing of input text.

    URLs are removed, text is converted to lowercase letter, some characters are removed
    and finally the resulting text is lemmatized for nouns, verbs and adjectives.

    :param data: Input DataFrame with raw input text
    :return: Cleaned and lemmatized text
    """

    out_data = data.copy()

    # Remove URLs
    out_data['text'] = out_data['text'].str.replace('http://\S+|https://\S+', '', regex=True)

    # Convert data to lowercase
    out_data['text'] = out_data['text'].str.lower()

    # Remove punctuation
    for punct in '?!.,"#$%\'''()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        out_data['text'] = out_data['text'].apply(lambda x: x.replace(punct, ''))

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    out_data['lemmatized'] = out_data['text'].apply(lambda x: word_tokenize(x))
    out_data['lemmatized'] = out_data['lemmatized'].apply(
        lambda words: [lemmatizer.lemmatize(word, 'n') for word in words])
    out_data['lemmatized'] = out_data['lemmatized'].apply(
        lambda words: [lemmatizer.lemmatize(word, 'v') for word in words])
    out_data['lemmatized'] = out_data['lemmatized'].apply(
        lambda words: [lemmatizer.lemmatize(word, 'a') for word in words])

    out_data['lemmatized'] = out_data['lemmatized'].apply(lambda x: ' '.join(x))

    return out_data


def f1_calculation(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Computes F1 metric for model performance evaluation.

    :param y_true: Series with true target values.
    :param y_pred: Series with predicted target values.
    :return: Calculated F1 score.
    """

    # Convert to array
    y_true_arr = y_true.values
    y_pred_arr = y_pred.values

    # Extract True Positive (TP), False Positive (FP) and False Negative (FN)
    TP = ((y_true_arr == 1) & (y_pred_arr == 1)).sum()
    FP = ((y_true_arr == 0) & (y_pred_arr == 1)).sum()
    FN = ((y_true_arr == 1) & (y_pred_arr == 0)).sum()

    # Compute precision and recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # Compute F1 score
    F1 = 2 * (precision * recall) / (precision + recall)

    return F1


def main():
    """
    Main function for model execution.
    """

    # -- 1. Read datasets --
    print('1. Loading input data')
    data_loaded = False
    try:
        train_data = pd.read_csv(os.path.join(INPUT_PATH, 'train.csv'))
        test_data = pd.read_csv(os.path.join(INPUT_PATH, 'test.csv'))
        data_loaded = True
        print('Data successfully loaded')
    except FileNotFoundError:
        print(f'Error: Files not found in {INPUT_PATH}')

    if data_loaded:
        # -- 2. Explore datasets --
        print('2. Dataset exporation')
        # Rate disaster
        count_disaster = (train_data['target'] == 1).sum()
        count_non_disaster = len(train_data) - count_disaster
        rate = count_disaster / len(train_data) * 100
        print(f'Number of disaster tweets on train datra: {count_disaster}')
        print(f'Number of non-disaster tweets on train datra: {count_non_disaster}')
        print(f'Rate of disaster tweets on train datra: {round(rate, 2)}%')

        # Sequences length
        num_words_train = train_data['text'].apply(lambda x: len(x.split()))
        num_words_test = test_data['text'].apply(lambda x: len(x.split()))

        # Print train data length
        print('\n Statistics from Train Dataset:')
        print(num_words_train.describe())

        # Print test data length
        print('\n Statistics from Test Dataset:')
        print(num_words_test.describe())

        # -- 3. Data preprocessing --
        print('3. Data preprocessing')
        train_data_prep = preprocessing(train_data)
        test_data_prep = preprocessing(test_data)

        # -- 4. Data preparation for model --
        print('4. Train/valid data split')
        # Features and target extraction
        X = train_data_prep['lemmatized']
        y = train_data_prep['target']

        # Split train/valid dataset
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

        # Test dataset
        X_test = test_data_prep['lemmatized']

        # -- 5. Loading model --
        print('5. Loading model')
        preset = "distil_bert_base_en_uncased"
        preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(preset,
                                                                           sequence_length=160,
                                                                           name="tweets_preprocessor")
        # Pretrained classifier.
        classifier = keras_nlp.models.DistilBertClassifier.from_preset(preset,
                                                                       preprocessor=preprocessor,
                                                                       num_classes=2)

        # Compile model
        adam = keras.optimizers.Adam(learning_rate=0.000001)
        classifier.compile(optimizer=adam,
                           loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        print('Model compiled')

        # -- 6. Callbacks definition --
        print('6. Defining callbacks (EarlyStopping and ReduceLROnPlateau)')
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=5,
                                       restore_best_weights=True)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.2,
                                      patience=2,
                                      min_lr=0.0000001)

        # -- 7. Model training --
        print('7. Model training')
        history_BERT = classifier.fit(X_train, y_train,
                                      validation_data=(X_valid, y_valid),
                                      batch_size=32, epochs=300, callbacks=[early_stopping, reduce_lr])

        # -- 8. Plot results --
        print('8. Plot results')
        pd.DataFrame(history_BERT.history).plot(figsize=(10, 6))
        plt.title('History of model training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

        # -- 9. Inference --
        print('9. Inference')
        valid_prediction_logits = classifier.predict(X_valid) # Valid data
        test_prediction_logits = classifier.predict(X_test) # Test data

        # Convert logits to probabilities
        valid_prob = tf.nn.softmax(valid_prediction_logits, axis=1).numpy()
        test_prob = tf.nn.softmax(test_prediction_logits, axis=1).numpy()

        # Take class with higher probability
        binary_pred_valid = np.argmax(valid_prob, axis=1)
        binary_pred_test = np.argmax(test_prob, axis=1)

        # Convert to data series
        binary_pred_valid = pd.Series(binary_pred_valid, name='valid_pred', index=X_valid.index)
        binary_pred_test = pd.Series(binary_pred_test, name='test_pred', index=X_test.index)

        # Concat with original y_valid
        y_valid_pred = pd.concat([y_valid.to_frame(), binary_pred_valid.to_frame()], axis=1)

        # Read submission file
        sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
        sample_submission['target'] = binary_pred_test[:, 1]

        # -- 10. Metrics computation --
        print('10. Metrics computation')
        F1 = f1_calculation(y_valid_pred)
        print(f'F1 metric: {F1}')

        # -- 11. Analysis of results --
        print('11. Analysis of results')
        disaster = (sample_submission['target'] == 1).sum()
        non_disaster = (sample_submission['target'] == 0).sum()
        print(f'Rate of disaster tweets predicted: {round(disaster / (disaster + non_disaster) * 100, 2)}%')

        # -- 12. Submission CSV --
        sample_submission.to_csv(os.path.join(OUTPUT_PATH, 'submission.csv'), index=False)


# Main
if __name__ == "__main__":
    main()
