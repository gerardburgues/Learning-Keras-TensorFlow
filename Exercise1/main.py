import tensorflow as tf
from tensorflow import keras


# Loading dataset fashion_mnist
(X_train_full, y_train_full), (X_test,
                               y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Scale input features to 0-1 range
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[:5000] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[:5000]
X_test = X_test / 255.0

# Creating labels names
class_names = ["T-shirt/top", "Trousers", "Pullover", "Dress",
               "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Creating the model using the Sequential API
# Neural Network with 2 hidden layers
# Single stacl of layers connected sequentially
model = keras.Sequential([
    # Flatten --> convert each input image into a 1D array
    keras.layers.Flatten(input_shape=[28, 28]),
    # Dense with 300-100 neurons ReLU activation function
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    # Dense with 10 neurons Softmax activation function
    keras.layers.Dense(10, activation="softmax")]
)

# Getting sumary of the model
print(model.summary())
hidden1 = model.layers[1]
print("weights , biases " , hidden1.get_weights())
