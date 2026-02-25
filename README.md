# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and convert categorical data into numerical form using one-hot encoding.

2. Separate input and output variables, then scale the data using StandardScaler.

3. Split the dataset into training and testing sets (80% training, 20% testing).

4. Apply Polynomial Regression with Ridge, Lasso, and ElasticNet models using a pipeline and train them.

5. Evaluate the models using Mean Squared Error (MSE) and R² score, then compare the results using graphs.


## Program:
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

data=pd.read_csv('CarPrice_Assignment.csv')
print(data.head())

data = pd.get_dummies(data, drop_first=True)
x=data.drop('price',axis=1)
y=data['price']

scaler = StandardScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y.values.reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

models = {"Ridge": Ridge(alpha=1.0),
          "Lasso": Lasso(alpha=1.0),
          "ElasticNet": ElasticNet (alpha=1.0, l1_ratio=0.5)}

result= {}
for name,model in models.items():
    pipeline = Pipeline([
        ('poly',PolynomialFeatures(degree=2)),
                         ('regressor',model)
    ])

pipeline.fit(x_train, y_train)

mse=mean_squared_error(y_test,predictions)
r2score=r2_score(y_test,predictions)

result[name]={'MSE': mse, 'R2 Score': r2score}

print('Name: A.Jannathul Shaban')
print('Reg. No: 212225220043')
for model_name,metrics in result.items():
    print(f"{model_name} -Mean Squared Error: {metrics['MSE']:.2f},R2_Score: {metrics['R2 Score']:.2f}")

result_df = pd. DataFrame (result). T
result_df.reset_index(inplace=True)
result_df.rename(columns={'index': 'Model'}, inplace=True)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MSE', data=result_df, palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R2 Score', data=result_df, palette='viridis')
plt.title('R² Score')
plt.ylabel('R² Score')
plt.xticks (rotation=45)

plt.tight_layout()
plt.show()

```

## Output:
<img width="599" height="88" alt="image" src="https://github.com/user-attachments/assets/f1ed42e5-98ca-4290-a7de-518cc4b52272" />

<img width="443" height="620" alt="Screenshot 2026-02-25 223632" src="https://github.com/user-attachments/assets/6c7c8522-5355-45a1-8c0f-f6342f191d8d" />

<img width="454" height="614" alt="Screenshot 2026-02-25 223655" src="https://github.com/user-attachments/assets/8b3089aa-b6b8-474c-bf9d-a438be13d27a" />







## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
