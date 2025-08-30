import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Generar señal con ruido
# -------------------------
fs = 500
t = np.linspace(0, 1, fs, endpoint=False)
clean_signal = np.sin(2*np.pi*5*t)
noisy_signal = clean_signal + 0.4*np.random.randn(len(t))

# ============================================================
# MODELO 1: Filtro Conv1D en PyTorch
# ============================================================
import torch
import torch.nn as nn

x = torch.tensor(noisy_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

class ConvFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        with torch.no_grad():
            self.conv.weight[:] = torch.ones_like(self.conv.weight)/7.0
    def forward(self, x):
        return self.conv(x)

conv_model = ConvFilter()
filtered_signal = conv_model(x).detach().numpy().flatten()

# ============================================================
# MODELO 2: Autoencoder en TensorFlow
# ============================================================
import tensorflow as tf

X_train = noisy_signal.reshape(1, -1, 1)
Y_train = clean_signal.reshape(1, -1, 1)

inputs = tf.keras.Input(shape=(fs,1))
encoded = tf.keras.layers.Conv1D(8, 5, activation="relu", padding="same")(inputs)
decoded = tf.keras.layers.Conv1D(1, 5, activation="linear", padding="same")(encoded)
autoencoder = tf.keras.Model(inputs, decoded)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(X_train, Y_train, epochs=200, verbose=0)
denoised_autoencoder = autoencoder.predict(X_train).flatten()

# ============================================================
# MODELO 3: LSTM en TensorFlow
# ============================================================
window_size = 20
def create_dataset(noisy, clean, window_size):
    X, Y = [], []
    for i in range(len(noisy) - window_size):
        X.append(noisy[i:i+window_size])
        Y.append(clean[i+window_size])
    return np.array(X), np.array(Y)

X, Y = create_dataset(noisy_signal, clean_signal, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))
Y = Y.reshape((Y.shape[0], 1))

lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='tanh', input_shape=(window_size,1)),
    tf.keras.layers.Dense(1)
])
lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.fit(X, Y, epochs=50, batch_size=16, verbose=0)

predictions = lstm_model.predict(X).flatten()
denoised_lstm = np.concatenate([np.zeros(window_size), predictions])

# ============================================================
# GRAFICAR RESULTADOS
# ============================================================
plt.figure(figsize=(12,10))

# Señal original
plt.subplot(5,1,1)
plt.plot(t, clean_signal, label="Clean", color="orange")
plt.title("Señal limpia (Clean)")

# Señal ruidosa
plt.subplot(5,1,2)
plt.plot(t, noisy_signal, label="Noisy", color="blue")
plt.title("Señal con ruido (Noisy)")

# Conv1D PyTorch
plt.subplot(5,1,3)
plt.plot(t, noisy_signal, alpha=0.5, label="Noisy")
plt.plot(t, filtered_signal, color="green", label="Conv1D Filtered")
plt.legend()
plt.title("Modelo 1: Conv1D (PyTorch)")

# Autoencoder TensorFlow
plt.subplot(5,1,4)
plt.plot(t, noisy_signal, alpha=0.5, label="Noisy")
plt.plot(t, denoised_autoencoder, color="red", label="Autoencoder Denoised")
plt.legend()
plt.title("Modelo 2: Autoencoder (TensorFlow)")

# LSTM TensorFlow
plt.subplot(5,1,5)
plt.plot(t, noisy_signal, alpha=0.5, label="Noisy")
plt.plot(t, denoised_lstm, color="purple", label="LSTM Denoised")
plt.legend()
plt.title("Modelo 3: LSTM (TensorFlow)")

plt.tight_layout()
plt.show()
