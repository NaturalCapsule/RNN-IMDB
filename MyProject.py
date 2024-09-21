import tensorflow.keras.layers as layers
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf

# Load the IMDB dataset with a limit of 10,000 most common words
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)

max_length = 500

# Pad the sequences to ensure uniform length
train_data = pad_sequences(train_data, maxlen=max_length, padding='pre')
test_data = pad_sequences(test_data, maxlen=max_length, padding='pre')


# Define the model
def model():
    inputs = layers.Input(shape=(max_length,))

    # Embedding layer restricted to 10,000 most common words
    x = layers.Embedding(input_dim=10000, output_dim=128, input_length=max_length)(inputs)

    # LSTM layers
    x = layers.LSTM(units=128, return_sequences=True)(x)
    x = layers.MaxPool1D()(x)

    x = layers.LSTM(units=64, return_sequences=True)(x)
    x = layers.MaxPool1D()(x)

    x = layers.LSTM(32, return_sequences=False)(x)

    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    # Output layer for binary classification
    outputs = layers.Dense(1, activation='sigmoid')(x)

    # Create the model
    rnn = tf.keras.models.Model(inputs, outputs)

    return rnn


# Instantiate the model
model = model()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=20, batch_size=64)
