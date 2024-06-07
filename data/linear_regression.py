import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Memuat data dari CSV
data = pd.read_csv('data/student-performance.csv')

# Memilih kolom yang relevan
data = data[['Hours Studied', 'Sample Question Papers Practiced', 'Performance Index']]

# Memisahkan variabel independen dan dependen
X = data[['Sample Question Papers Practiced']]
y = data['Performance Index']

# Memisahkan data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Metode 1: Model Linear
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Menghitung galat RMS
rms_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
print(f'RMS Error (Linear Model): {rms_linear}')

# Plotting
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_linear, color='red', label='Predicted')
plt.title('Linear Regression')
plt.xlabel('Sample Question Papers Practiced')
plt.ylabel('Performance Index')
plt.legend()
plt.show()

# Menyimpan hasil ke file CSV
linear_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_linear})
linear_results.to_csv('results/linear_results.csv', index=False)
