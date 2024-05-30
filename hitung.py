import numpy as np
import matplotlib.pyplot as plt

# Data nilai latihan (NL) dan nilai ujian (NT)
TB = np.array([7.1, 4.2, 8.2, 7.5, 3.6, 7.6,])
NL = np.array([1, 2, 2, 5, 6, 6])
NT = np.array([91.0, 65.0, 45.0, 66.0, 61.0, 63.0])

# Fungsi untuk model linear
def model_linear(x, m, b):
  return m * x + b

# Fungsi untuk model eksponensial
def model_eksponensial(x, a, c):
  return a * np.exp(c * x)

# Mencari parameter model linear dengan regresi linier
m, b = np.polyfit(NL, NT, 1)

# Mencari parameter model eksponensial dengan regresi non-linier
from scipy.optimize import curve_fit
a, c = curve_fit(model_eksponensial, NL, NT)[0]

# Menghitung nilai prediksi untuk model linear dan eksponensial
NL_pred_linear = model_linear(NL, m, b)
NL_pred_eksponensial = model_eksponensial(NL, a, c)

# Menghitung galat RMS untuk model linear dan eksponensial
RMS_linear = np.sqrt(np.mean((NT - NL_pred_linear)**2))
RMS_eksponensial = np.sqrt(np.mean((NT - NL_pred_eksponensial)**2))

# Menampilkan hasil
print("Model Linear:")
print(f"Persamaan: y = {m:.2f}x + {b:.2f}")
print(f"Galat RMS: {RMS_linear:.2f}")

print("\nModel Eksponensial:")
print(f"Persamaan: y = {a:.2f}e^{c:.2f}x")
print(f"Galat RMS: {RMS_eksponensial:.2f}")

# Membuat plot
plt.plot(NL, NT, 'o', label='Data')
plt.plot(NL, NL_pred_linear, label='Model Linear')
plt.plot(NL, NL_pred_eksponensial, label='Model Eksponensial')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()
plt.grid(True)
plt.show()
