# Import modules and packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Functions
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    
    plt.legend(shadow=True)   # FIXED
    
    plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
    plt.title('Model Results', family='Arial', fontsize=14)
    plt.xlabel('X axis values', family='Arial', fontsize=11)
    plt.ylabel('Y axis values', family='Arial', fontsize=11)
    plt.savefig('model_results.png', dpi=120)

# Correct MAE & MSE functions
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# DATA
print(tf.__version__)

# Create features (reshaped for dense layers)
X = np.arange(-100, 100, 4).reshape(-1, 1)
y = np.arange(-90, 110, 4).reshape(-1, 1)

print(f"X shape: {X.shape}, y shape: {y.shape}")

# Split into train/test
X_train = X[:40]
y_train = y[:40]
X_test = X[40:]
y_test = y[40:]

# Input & output shapes
input_shape = X[0].shape
output_shape = y[0].shape
print(f"Input shape: {input_shape}, Output shape: {output_shape}")

# MODEL

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mae']
)

print(model.summary())

# Train
history = model.fit(X_train, y_train, epochs=100, verbose=1)

# ------------------------------------
# PREDICTIONS + PLOT
# ------------------------------------

y_preds = model.predict(X_test)
plot_predictions(
    train_data=X_train.squeeze(),
    train_labels=y_train.squeeze(),
    test_data=X_test.squeeze(),
    test_labels=y_test.squeeze(),
    predictions=y_preds.squeeze()
)

# METRICS

mae_1 = np.round(mae(y_test, y_preds), 2)
mse_1 = np.round(mse(y_test, y_preds), 2)

print(f"\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.")

# Save metrics
with open('metrics.txt', 'w') as outfile:
    outfile.write(f"Mean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.")

# Save model
model.save('model.h5')
print("✅ Model saved as model.h5 for deployment!")
