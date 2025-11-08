
# 📊 Customer Churn Analysis & Prediction

Predicting telecom customer churn using **machine learning** and uncovering actionable business insights.  
Includes data generation, model training, evaluation, and visual analysis.

---

## 🚀 Project Overview
- Analyzed **6,000+ telecom customer records** to identify churn behavior using Python.  
- Built predictive models (**Logistic Regression**, **Random Forest**, **Gradient Boosting**) achieving **~71% ROC-AUC**.  
- Visualized insights through **EDA** (Matplotlib, Seaborn) and optional **Tableau dashboard**.  
- Key churn drivers: **contract type**, **payment method**, **monthly charges**, and **tenure**.

---

## 🧠 Tech Stack
| Category | Tools |
|-----------|-------|
| **Languages** | Python |
| **Libraries** | Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn |
| **Visualization** | Tableau *(optional)* |
| **Environment** | VS Code, Git, Jupyter Notebook |

---

## 📂 Project Structure
```
CustomerChurnAnalysis/
├── data/                 # Generated dataset
│   └── churn_dataset.csv
├── src/                  # Data generation & model training scripts
│   ├── generate_data.py
│   ├── train_churn.py
│   └── predict_churn.py
├── artifacts/            # Model outputs & predictions
│   ├── churn_best_model.pkl
│   ├── churn_predictions.csv
│   └── churn_top_features.csv
├── notebooks/            # Exploratory Data Analysis
│   └── EDA.ipynb
├── requirements.txt
└── README.md
```

---

## 🧩 How to Run

```bash
# 1️⃣ Clone repo
git clone https://github.com/sroowo/CustomerChurnAnalysis.git
cd CustomerChurnAnalysis

# 2️⃣ Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Generate synthetic dataset
python src/generate_data.py --out data/churn_dataset.csv --n 6000 --seed 7

# 5️⃣ Train churn prediction models
python src/train_churn.py --data data/churn_dataset.csv --out artifacts
```

---

## 📈 Results
| Model | ROC-AUC | Accuracy | Key Insight |
|--------|---------|-----------|--------------|
| Logistic Regression | 0.71 | 67 % | Best performer — interpretable and stable |
| Gradient Boosting | 0.70 | 66 % | Captures nonlinear churn patterns |
| Random Forest | 0.69 | 66 % | Robust but slightly overfits |

### 🔍 Top Churn Indicators
- 📅 **Month-to-Month contracts** → highest churn (~51 %)  
- 💳 **Electronic-check payments** → high-risk segment  
- 💸 **Higher monthly charges** → more likely to churn  
- ⏳ **Short tenure (< 12 months)** → higher churn probability

---

## 🖼 Example EDA Visuals
*(Add screenshots later)*  
- Churn Rate by Contract Type  
- Monthly Charges vs Churn (Boxplot)  
- Tenure vs Churn Probability (Scatterplot)  
- Correlation Heatmap  

---

## 📊 Dashboard *(Optional)*
A Tableau dashboard visualizing churn distribution and customer behavior can be built using  
`artifacts/churn_predictions.csv`.

> Coming soon: [Tableau Dashboard Link](#)

---

---

## 🏷 Keywords
`Python` • `Machine Learning` • `Customer Churn` • `EDA` • `Data Visualization` • `Predictive Modeling` • `Tableau`