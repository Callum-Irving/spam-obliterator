import string

import numpy as np
import tensorflow as tf
import pandas
import matplotlib.pyplot as plt


def plot_graphs(history, metric: str):
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric], "")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, "val_" + metric])


def read_csv_to_tf(filename):
    # Read file
    csv = pandas.read_csv(filename, encoding="ISO-8859-1")
    csv = csv[["v1", "v2"]]
    csv.rename(columns={"v1": "spam", "v2": "text"}, inplace=True)
    csv.spam = csv.spam.apply(lambda x: x == "spam")

    # Detach features from answers
    labels = csv.pop("spam")
    messages = csv.pop("text")

    # Clean up messages
    messages = messages.apply(clean_message)

    # Convert to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((messages.values, labels.values))

    # Split into train and test sets
    length = len(dataset)
    train_size = int(length * 0.8)
    test_size = length - train_size
    train_set = dataset.take(train_size)
    test_set = dataset.skip(test_size)

    return train_set, test_set


def shuffle_set(dataset):
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    return dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def clean_message(message: str) -> str:
    return (
        message.lower()
        .replace("$", " dollar ")
        .translate(str.maketrans("", "", string.punctuation))
    )


def create_model(train_set):
    # Create encoder layer
    VOCAB_SIZE = 1000
    encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
    encoder.adapt(train_set.map(lambda text, _: text))

    return tf.keras.Sequential(
        [
            encoder,
            tf.keras.layers.Embedding(
                input_dim=len(encoder.get_vocabulary()),
                output_dim=64,
                mask_zero=True,
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )


if __name__ == "__main__":
    # Load the dataset
    train, test = read_csv_to_tf("data.csv")
    train = shuffle_set(train)
    test = shuffle_set(test)

    # Create the model
    model = create_model(train)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"]
    )

    # Train our model
    history = model.fit(
        train,
        epochs=7,
        validation_data=test,
        validation_steps=5,
    )

    # Show results of training
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plot_graphs(history, "accuracy")
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_graphs(history, "loss")
    plt.ylim(0, None)
    plt.show()

    # Save model
    model.save("spam_model")
