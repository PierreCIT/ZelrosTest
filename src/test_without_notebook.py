import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import re
import heapq
from keras_tqdm import TQDMCallback

tqdm.pandas()
Input = tf.keras.layers.Input
Bidirectional = tf.keras.layers.Bidirectional
CuDNNLSTM = tf.compat.v1.keras.layers.CuDNNLSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
GlobalMaxPool1D = tf.keras.layers.GlobalMaxPool1D

print("Tensorflow version : {}".format(tf.__version__))
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
tf.config.experimental.set_memory_growth(gpus[0], True)

max_vocabulary_length = 300  # The maximum number of words in the vocabulary (171 000 words in english dictionary)
batch_size = 50  # Training batch size
validation_batch_size = 300  # Validation batch size
epochs = 5  # number of epoch

model = tf.keras.Sequential()
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True), input_shape=(1, max_vocabulary_length)))
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))
model.add(GlobalMaxPool1D())
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

data_df = pd.read_csv("../dataset/train.csv")
print("{} training data available".format(data_df.shape[0]))

def to_lower_case_and_rm_double_spaces_poncuation(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'\W', ' ', sentence)  # Remove punctuation (non-word caracters)
    sentence = re.sub(r'\s+', ' ', sentence)  # Remove multiple space
    if sentence[-1] == " ":  # Remove useless space at the end of the sentence
        sentence = sentence[:-1]
    return sentence


data_df["question_text"] = data_df["question_text"].progress_apply(to_lower_case_and_rm_double_spaces_poncuation)
print("Check the result on the first sentence : {}".format(data_df["question_text"][0]))


train_df, val_df = train_test_split(data_df, test_size=0.1)

percentage_in_train = train_df.groupby("target").count()["qid"][1] / train_df.shape[0]
percentage_in_val = val_df.groupby("target").count()["qid"][1] / val_df.shape[0]
print(f"Train dataset size: {train_df.shape[0]}, validation size: {val_df.shape[0]}, "
      f"{math.floor(val_df.shape[0] * 100 / train_df.shape[0])}% of the training dataset size")
print("Percentage of positives in train = {:.2f} and in val {:.2f}".format(percentage_in_train, percentage_in_val))


voc = {}  # Contain every word with their number of occurrences
for index, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
    question = row["question_text"]
    for word in question.split(" "):
        if word not in voc.keys():
            voc[word] = 1
        else:
            voc[word] += 1

print("The vocabulary contains {} words".format(len(voc)))


voc_most_freq = heapq.nlargest(max_vocabulary_length, voc, key=voc.get)
for i in range(50):  # print the 50 most used words
    print(voc_most_freq[i])


def vectorize_question(
        question):  # We assume the question as already been formatted (lower case, no punctuation, spaces)
    vector = np.zeros(max_vocabulary_length)  # Initial vector filled with zeros
    for _word in question.split(" "):
        try:
            _index = voc_most_freq.index(_word)
            vector[_index] += 1
        except ValueError:
            pass # If the word is not in the vocabulary we do nothing
    return vector


def create_vectors_from_dataframe(data):
    vectors = np.zeros((data.shape[0], 1, max_vocabulary_length), dtype=np.int)
    i = 0
    for _index, _row in data.iterrows():
        vectors[i][0] = vectorize_question(_row[1])
        i += 1
    return vectors




def training_generator(_train_df):
    nb_batches = math.ceil(_train_df.shape[0] / batch_size)

    while True:
        _train_df = _train_df.sample(frac=1)  # shuffle the data
        for i in range(nb_batches):
            vectors = create_vectors_from_dataframe(_train_df.iloc[i * batch_size:(i + 1) * batch_size])
            yield np.asarray(vectors), np.asarray(_train_df["target"][i * batch_size:(i + 1) * batch_size].values)

generator = training_generator(train_df)
a, b = generator.__next__()
print(a.shape)
print(b.shape)

model.fit_generator(generator, steps_per_epoch=train_df.shape[0] // batch_size, epochs=epochs, verbose=0,
                    callbacks=[TQDMCallback()])

