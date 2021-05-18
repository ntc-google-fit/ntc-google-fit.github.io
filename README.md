## NTC - Google Fit Project

For the 2nd build week on Strive School, the task is to create a machine learning model based on the data from Google Fit.

<br />

### Install and import packages

```python
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
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
dataset.isna().sum()
```

<br />
<!-- 
### Data Analysis

```python
print(dataset.head()) -->
```

<br />

### Support or Contact

Having trouble with to run your model? Please contact the support and we’ll help you out.
