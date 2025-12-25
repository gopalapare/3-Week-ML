import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve

# 1. SYNTHETIC DATA GENERATION
# Let's simulate: 'Amount Spent' vs 'Is Fraud'
np.random.seed(42)
amounts = np.random.uniform(10, 1000, 100).reshape(-1, 1)
# If amount > 700, it's likely fraud (1), else legit (0) + some noise
is_fraud = (amounts > 700).astype(int).flatten() 

# 2. CREATE DATAFRAME
df = pd.DataFrame({'Amount': amounts.flatten(), 'IsFraud': is_fraud})

# 3. SPLIT DATA
X = df[['Amount']]
y = df['IsFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAIN MODEL
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. PREDICT
y_pred = model.predict(X_test)

# 6. EVALUATE
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Get probabilities instead of 0/1 labels
y_probs = model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_test, y_probs)
print(f"AUC Score: {auc:.2f}")

# Optional: Plot the curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--') # Diagonal line (Random guessing)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()