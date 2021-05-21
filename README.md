## Machine Learning Project

For the 2nd build week on Strive School, the task is to create a machine learning model based on the data from Google Fit.

<br />

### How to run

1. Clone the repository:

```python
$ git clone https://github.com/ntc-google-fit/google_fit_project
$ google_fit_project

```

2. Install dependencies:

```
$ pip install -r requirements.txt
```

3. Start the application:

```
streamlit run app.py
```

If you need extra help with Streamlit, got to the [documentation](https://docs.streamlit.io)

<br />

## The Team

    - [Fabio Fistarol](https://github.com/fistadev)
    - [Deniz Elci](https://github.com/deniz-shelby)
    - [Farrukh Bulbulov](https://github.com/fbulbulov)
    - [Vladimir Gasanov](https://github.com/VladimirGas)

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

### Split

```python
# split by users
train_users = ['U1', 'U3', 'U6', 'U7', 'U10', 'U12']
test_users = ['U2', 'U4', 'U5', 'U8', 'U9', 'U11']

# split
x_train = dataset_train.drop(['target'], axis=1)
y_train = dataset_train.target
x_test = dataset_test.drop(['target'], axis=1)
y_test = dataset_test.target
```

<br />

### Deployment

```python
Streamlit
```

<br />

### Support or Contact

Having trouble with to run your model? Please contact the support and we’ll help you out.

<br />
