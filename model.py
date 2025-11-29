# Import modules and packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Functions and procedures
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(6, 5))
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  plt.scatter(test_data, predictions, c="r", label="Predictions")
  plt.legend(shadow=True)
  plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
  plt.title('Model Results', family='Arial', fontsize=14)
  plt.xlabel('X axis values', family='Arial', fontsize=11)
  plt.ylabel('Y axis values', family='Arial', fontsize=11)
  plt.savefig('model_results.png', dpi=120)

def mae(y_test, y_pred):
  return tf.metrics.mean_absolute_error(y_test, y_pred)

def mse(y_test, y_pred):
  return tf.metrics.mean_squared_error(y_test, y_pred)

# Check Tensorflow version
print(tf.__version__)

# Create features (1D → RESHAPE FOR DENSE LAYERS)
X = np.arange(-100, 100, 4).reshape(-1, 1)  # ✅ (50, 1) shape
y = np.arange(-90, 110, 4).reshape(-1, 1)   # ✅ (50, 1) shape

print(f"X shape: {X.shape}, y shape: {y.shape}")

# Split data into train and test sets
X_train = X[:40] 
y_train = y[:40]
X_test = X[40:] 
y_test = y[40:]

# ✅ FIXED: input_shape now works
input_shape = X[0].shape 
output_shape = y[0].shape
print(f"Input shape: {input_shape}, Output shape: {output_shape}")

# Set random seed
tf.random.set_seed(42)

# Create model (FIXED for 2D input)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(1,)),  # ✅ Better model
    tf.keras.layers.Dense(1)
])

# Compile with better optimizer
model.compile(loss=tf.keras.losses.mse,  # MSE better for regression
              optimizer=tf.keras.optimizers.Adam(),  # Adam > SGD
              metrics=['mae'])

print(model.summary())

# Fit the model
history = model.fit(X_train, y_train, epochs=100, verbose=1)

# Make and plot predictions
y_preds = model.predict(X_test)
plot_predictions(train_data=X_train.squeeze(), train_labels=y_train.squeeze(), 
                test_data=X_test.squeeze(), test_labels=y_test.squeeze(), 
                predictions=y_preds.squeeze())

# Calculate metrics
mae_1 = np.round(float(mae(y_test, y_preds).numpy()), 2)
mse_1 = np.round(float(mse(y_test, y_preds).numpy()), 2)
print(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')

# Write metrics to file
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'Mean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')

# 🚀 SAVE MODEL FOR DEPLOYMENT
model.save('model.h5')
print("✅ Model saved as model.h5 for Render deployment!")
