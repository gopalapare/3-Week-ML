# 🚀 3-Week Machine Learning Interview Challenge

This repository is a structured 21-day journey to mastering the most important Machine Learning algorithms, from mathematical intuition to production-ready code. 

**Tech Stack:** - **Package Manager:** [uv](https://github.com/astral-sh/uv) (for high-performance dependency management)
- **Environment:** VS Code + Jupyter
- **Libraries:** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn

---

## 📅 Roadmap & Progress

| Day | Algorithm | Focus | Status |
| :--- | :--- | :--- | :--- |
| **01** | **Linear Regression** | Ordinary Least Squares, MSE, R2 Score | ✅ Completed |
| 02 | **Logistic Regression** | Sigmoid, Binary Cross-Entropy, Precision/Recall | ⏳ Next Up |
| 03 | **Decision Trees** | Entropy, Gini Impurity, Overfitting | ⏳ Planned |
| ... | ... | ... | ... |

---

## 🛠️ Daily Log

### Day 1: Linear Regression (Regression)
**Goal:** Predict a continuous value (House Price) based on a single feature (Square Footage).

#### 🧠 Key Takeaways
- **The Math:** Understanding the line of best fit $y = mx + c$.
- **Cost Function:** Using **Mean Squared Error (MSE)** to measure the distance between actual and predicted values.
- **Evaluation:** - **R-Squared ($R^2$):** Tells us the % of variance explained by the model.
    - **Adjusted $R^2$:** Essential for multiple features to avoid "greedy" metrics.

#### 📊 Visualizing the Fit


#### 🚀 How to Run
```bash
uv run day1_linear_regression.py