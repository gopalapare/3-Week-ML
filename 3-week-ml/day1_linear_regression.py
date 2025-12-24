import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. GENERATE DUMMY DATA (House Size vs Price)
# Let's imagine: Price = 50 * Size + 100 + some noise
np.random.seed(42)
sizes = np.random.randint(500, 3500, 100).reshape(-1, 1) # Square feet
prices = 50 * sizes.flatten() + 100 + np.random.normal(0, 5000, 100) # Price in $

# 2. CREATE A DATAFRAME
df = pd.DataFrame({'Size': sizes.flatten(), 'Price': prices})

# 3. SPLIT DATA (Train on 80%, Test on 20%)
X = df[['Size']] 
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. TRAIN THE MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# 5. EVALUATE
predictions = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, predictions):.2f}")
print(f"Intercept (c): {model.intercept_:.2f}")
print(f"Coefficient (m): {model.coef_[0]:.2f}")