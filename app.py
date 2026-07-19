import json
import pickle

import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)


# Load configuration
with open("config.json", "r", encoding="utf-8") as file:
    config = json.load(file)

english_max_length = int(config["english_max_length"])
french_max_length = int(config["french_max_length"])


# Load tokenizers
with open("english_tokenizer.pkl", "rb") as file:
    english_tokenizer = pickle.load(file)

with open("french_tokenizer.pkl", "rb") as file:
    french_tokenizer = pickle.load(file)


# Recreate the same model architecture used during training
english_vocab_size = len(english_tokenizer.word_index) + 1
french_vocab_size = len(french_tokenizer.word_index) + 1

model = Sequential(
    [
        Input(shape=(english_max_length,)),
        Embedding(
            input_dim=english_vocab_size,
            output_dim=128,
        ),
        LSTM(256),
        RepeatVector(french_max_length),
        LSTM(256, return_sequences=True),
        TimeDistributed(
            Dense(french_vocab_size, activation="softmax")
        ),
    ]
)

# Load the trained weights from the H5 file
model.load_weights("model.h5")

print("Model architecture and trained weights loaded successfully.")


# Reverse French tokenizer dictionary
index_to_french = {
    index: word
    for word, index in french_tokenizer.word_index.items()
}


def translate_sentence(sentence):
    sequence = english_tokenizer.texts_to_sequences([sentence])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=english_max_length,
        padding="post",
    )

    prediction = model.predict(padded_sequence, verbose=0)
    predicted_indices = np.argmax(prediction, axis=-1)[0]

    translated_words = []

    for index in predicted_indices:
        index = int(index)

        if index != 0:
            word = index_to_french.get(index)

            if word:
                translated_words.append(word)

    return " ".join(translated_words)


@app.route("/", methods=["GET", "POST"])
def home():
    sentence = ""
    translation = ""
    error = ""

    if request.method == "POST":
        sentence = request.form.get("sentence", "").strip()

        if not sentence:
            error = "Please enter an English sentence."
        else:
            translation = translate_sentence(sentence)

            if not translation:
                error = "The model could not generate a translation."

    return render_template(
        "index.html",
        sentence=sentence,
        translation=translation,
        error=error,
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )