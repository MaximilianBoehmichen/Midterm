```
df.info()
print(df.isna().any())
```

### Renaming
```python
users.rename(columns={
    'uid': 'UserId',
    'p': 'Premium',
    'm1': 'Minutes1',
    'm2': 'Minutes2',
    'm3': 'Minutes3' 
}, inplace=True)
```

### Mapping
```python
users['Premium'] = users['Premium'].map({'0': False, 
                                         '1': True,
                                         'Yes': True,
                                         'No': False},)

df["salutation"] = df["salutation"].replace({
    2 : "Company",
    3 : "Mr.",
    4 : "Mrs."}).astype("category")
```

### Filling missing values
```python
users['Minutes2'] = users['Minutes2'].fillna((users['Minutes1'] + users['Minutes3'])/2)
df = df.fillna(value=means)
```

### Change Attribte type
```python
user_behavior['Genre'] = user_behavior['Genre'].astype('category')
user_behavior['Favorite'] = user_behavior['Favorite'].astype('bool')
```

### Aggregate
```python
user_behavior['Genre'] = user_behavior['Genre'].map({
    'Electronic': 'Electronic',
    'Rock': 'Rock',
    'Hip-Hop': 'Hip-Hop',
    'Pop': 'Pop'
}).fillna('Other').astype('category')
```

### ordinal Attributes
```python
df["size"] = df["size"].astype("str").str.upper()  
df["size"] = pd.Series(pd.Categorical(df["size"], categories=["S", "M", "L", "XL", "XXL", "XXXL"], ordered=True))
```

### Merging
```python
df_customers_extended = pd.merge(df_customers, 
                                 df_orders, 
                                 left_on="ID", 
                                 right_on="Customer", 
                                 how="left",
                                 suffixes=("_customer", "_order"))

bitmask_customers_order = df_customers["ID"].isin(df_orders["Customer"])
df_customers_no_orders = df_customers[~bitmask_customers_order]
```

### One-Hot Encoding
```python
df['Churn'] = df['Churn'].astype('category')

# Identify non-numeric columns
non_numeric_cols = df.select_dtypes(include=['object']).columns

# Apply one-hot encoding to non-numeric columns
df = pd.get_dummies(df, columns=non_numeric_cols, drop_first=True)
```