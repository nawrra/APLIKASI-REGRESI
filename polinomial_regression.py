import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Memuat data dari CSV
data = pd.read_csv('data\student-performance.csv')

# Memilih kolom yang relevan
data = data[['Hours Studied', 'Sample Question Papers Practiced', 'Performance Index']]

# Memisahkan variabel independen dan dependen
X = data[['Sample Question Papers Practiced']]
y = data['Performance Index']

# Memisahkan data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Metode 2: Model Pangkat Sederhana (Polinomial Derajat 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Menghitung galat RMS
rms_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
print(f'RMS Error (Polynomial Model): {rms_poly}')

# Plotting
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred_poly, color='red', label='Predicted')
plt.title('Polynomial Regression')
plt.xlabel('Sample Question Papers Practiced')
plt.ylabel('Performance Index')
plt.legend()
plt.show()

# Menyimpan hasil ke file CSV
poly_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_poly})
poly_results.to_csv('results/poly_results.csv', index=False)
