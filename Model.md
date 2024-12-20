### Important Imports
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
```

### Separation
```python
data.summary()
data.head(10)

train_data = data[data['id'].isna()]  # Rows with labels
private_data = data[data['label'].isna()]  # Rows with ids

X = train_data.drop(columns=['id', 'label'])
y = train_data['label']
```

### Missing values
```python
X = X.fillna(e.g. some median)

X_private = private_data.drop(columns=['id', 'label'])
X_private = X_private.fillna(X_private.median())
```

### Test Train Split
```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1337)
```

### Possible Training and Accuracy
```python
# Train a Random Forest model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Validate the model
y_pred = clf.predict(X_val)
bac_score = balanced_accuracy_score(y_val, y_pred)
print(cross_val_score(scoring='balanced_accuracy'))

private_predictions = clf.predict(X_private)
```

### Linear Model
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
```

### Tree
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

### Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
```

### Logit (only binary result)
```python
import statsmodels.formula.api as smf

# Mark "rank" as categorical
df_train["rank"] = df_train["rank"].astype("category")

# Define and fit a model
logreg = smf.logit("admit ~ gre + gpa + rank", data=df_train)
result = logreg.fit()
print(result.summary())
```

### Ensemble Methods
```python
from sklearn.ensemble import VotingClassifier

model1 = LogisticRegression(random_state=42)
model2 = RandomForestClassifier(random_state=42)
model3 = SVC(probability=True, random_state=42)

voting_model = VotingClassifier(estimators=[('lr', model1), ('rf', model2), ('svc', model3)], voting='soft')
voting_model.fit(X_train, y_train)
```

### Grid Search for best parameters
```python
# e.g. RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Initialize the model
rf = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200], 
    'max_depth': [10, 20, None],    
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4],  
    'bootstrap': [True, False]      
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1, verbose=2)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Best parameters and model
print("Best Parameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_ # model

```