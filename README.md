## NTC - Google Fit Project

For the 2nd build week on Strive School, the task is to create a machine learning model based on the data from Google Fit.

<br />

### Install and import packages

```python
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

set_config(display='diagram')
```

<br />

### Data Analysis

Load data and create pandas dataframe for that.

```python
dataset = pd.read_csv('./data/*.csv')
print(dataset.head())
```

<br />

### Preprocessing

```python
to_drop = [i for i in dataset.columns if '#std' in i]
dataset.drop(to_drop, axis=1, inplace=True)
dataset.head()

dataset.isna().sum()
```

```python
import plotly.express as px

fig = px.imshow(dataset.corr())
fig.show()
```

<br />

### Pipeline

```python
tree_classifiers = {
  "Decision Tree": DecisionTreeClassifier(),
  "Extra Trees":ExtraTreesClassifier(),
  "Random Forest":RandomForestClassifier(),
  "AdaBoost":AdaBoostClassifier(),
  "Skl GBM":GradientBoostingClassifier(),
  "Skl HistGBM":GradientBoostingClassifier(),
  "XGBoost":XGBClassifier(),
  "LightGBM":LGBMClassifier(),
  "CatBoost":CatBoostClassifier()
}

tree_classifiers = {name: pipeline.make_pipeline(tree_prepro, model) for name, model in tree_classifiers.items()}


tree_classifiers["LightGBM"]
```

<br />

### Split

```python
# split
x_train = dataset_train.drop(['target'], axis=1)
y_train = dataset_train.target
x_test = dataset_test.drop(['target'], axis=1)
y_test = dataset_test.target
```

<br />

### Model

```python
x_train, x_val, y_train, y_val = model_selection.train_test_split(
    x_train, y_train,
    test_size=0.2,
    stratify = y_train,
    random_state=37
)
accuracy: 81%
```

<br />

### Deployment

```python
Streamlit
```

<br />

### Support or Contact

Having trouble with to run your model? Please contact the support and we’ll help you out.
