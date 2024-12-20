
NEVER FORGET TO ADD THE INTERCEPT FOR OLS!
sklearn.linear_model.LogisticRegression adds an intercept automatically

1. Je höher Log-Likelihood, desto besser
2. Je niedriger AIC, desto besser
3. Je niedriger BIC, desto besser

### Prediction vs real
```python
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
for ax, variable_name in zip(axs, ["x1", "x2", "x3"]):
    ax.scatter(data[variable_name], data["y"], label="Ground truth")
    ax.scatter(data[variable_name], predicted_values, label="Model prediction")
    ax.legend()
    ax.set_xlabel(variable_name)

residuals = data["y"] - predicted_values
```

### White Test (Homoskedasticity)
```python
from statsmodels.stats.diagnostic import het_white

statistic, p_value, _, _ = het_white(residuals, X)
print(f"Value of the null-hypothesis that the residuals are homoscedastic: {statistic}")
print(f"p-value of the statistic: {p_value}")
# significant, that it is not a linear model -> no homoscedasticity
```

### Variance Inflation Factor (Variables Independent?)
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

for index, variable_name in enumerate(X.columns):
    if variable_name == "const":
        continue
    print(f"VIF for variable {variable_name} is {vif(X, index)}")

# VIF >> 10 is high -> multicollinear?
```

### Wald Test (Variable Significance, ordinal variable?)
```python
import statsmodels.formula.api as smf

# Specify the restrictions as a list of strings and pass it to "r_matrix"
logreg = smf.logit("admit ~ gre + gpa + rank", data=df_train)
result = logreg.fit()
wald_test_result = result.wald_test(
  	"(rank[T.2] = 0, rank[T.3] = 0, rank[T.4] = 0)", 
  	scalar=True
)
print(f"Test statistic (chi^2_{int(wald_test_result.df_denom)}-distributed): {wald_test_result.statistic}")
print(f"p-value of the statistic: {wald_test_result.pvalue}")
```

### McFadden Ratio (je näher an 1, desto besser; bei nicht linearen Modellen/logit, 0: Intercept only Modell)
```python
print(result.prsquared)

```
