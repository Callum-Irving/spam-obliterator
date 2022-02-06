from rnn import read_csv_to_tf, shuffle_set, create_model


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

tf.get_logger().setLevel("ERROR")

train, test = read_csv_to_tf("data.csv")
train = shuffle_set(train)
test = shuffle_set(test)


model_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"
pre_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

bert_preprocess_model = hub.KerasLayer(pre_url)
bert_model = hub.KerasLayer(model_url)


def build_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessing_layer = hub.KerasLayer(pre_url, name="preprocessor")
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(model_url, trainable=True, name="BERT_encoder")
    outputs = encoder(encoder_inputs)
    net = outputs["pooled_output"]
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name="classifier")(net)
    return tf.keras.Model(text_input, net)


print("\n\n** BUILDING MODEL **\n")
model = build_model()

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(3e-5),
    metrics=[tf.metrics.BinaryAccuracy()],
)

print("\n\n** TRAINING MODEL **\n")
history = model.fit(train, epochs=5, validation_data=test, validation_steps=5)

print("\n\n** EVALUATING MODEL **\n")
loss, accuracy = model.evaluate(test)

print("\n\n** PLOTTING RESULTS **\n")
import matplotlib.pyplot as plt

history_dict = history.history

acc = history_dict["binary_accuracy"]
val_acc = history_dict["val_binary_accuracy"]
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
# r is for "solid red line"
plt.plot(epochs, loss, "r", label="Training loss")
# b is for "solid blue line"
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
# plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, "r", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")

plt.show()
