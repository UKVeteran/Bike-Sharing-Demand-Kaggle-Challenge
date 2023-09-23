# Bike-Sharing-Demand-Kaggle-Challenge

![NicLonsdale](https://github.com/UKVeteran/Bike-Sharing-Demand-Kaggle-Challenge/assets/39216339/de6bed44-ba5e-4462-bfaa-9e90280d64ee)

# RMSE
The root-mean-square error (RMSE) is a measure used to assess how well a predictive model, such as a machine learning algorithm, is performing. It is a way to quantify the average difference between the predicted values and the actual (observed) values. Here's a simple explanation:

1 - Squared Differences: For each data point, you calculate the squared difference between the predicted value and the actual value. This is done to make sure that both overestimations and underestimations contribute to the error, without canceling each other out.
2 - Average: You then take the average (mean) of all these squared differences. This gives you a measure of the typical error the model makes across all data points.
3 - Square Root: Finally, you take the square root of this average to get the RMSE. This step is important because it ensures that the RMSE is in the same units as the original data, making it easier to interpret.

In simpler terms, RMSE tells you how far off, on average, our model's predictions are from the actual values. Smaller RMSE values indicate that the model's predictions are closer to the actual values, while larger RMSE values mean the predictions are further away. It is a way to quantify the goodness of fit of your model to the data, with lower values indicating a better fit.

# Data Viz.

```python

from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')

x = df['humidity']
y = df['windspeed']
z = df['atemp']
c = df['count']

img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
plt.xlabel('humidity', fontsize=22, labelpad=20)
plt.ylabel('windspeed', fontsize=22, labelpad=20)
#plt.zlabel('atemp', fontsize=22, labelpad=25)
ax.view_init(45,60)
fig.colorbar(img)
plt.show()
```

![download](https://github.com/UKVeteran/Bike-Sharing-Demand-Kaggle-Challenge/assets/39216339/3782594c-0c94-48e1-acde-e0946007f24b)


# Stack'em Up - RMSE Score = 0.32

```python

from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
import numpy as np

# Define the base estimators
base_estimators = [
    ('random_forest', RandomForestRegressor(max_depth=9, min_samples_leaf=2, n_estimators=100, random_state=0)),
    ('adaboost', AdaBoostRegressor(base_estimator=RandomForestRegressor(max_depth=9, min_samples_leaf=2),
                                   n_estimators=100, learning_rate=1.0, random_state=0)),
    ('svm', SVR(kernel='linear', C=1.0)),
    ('catboost', CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1))
]

# Create the StackingRegressor with the best parameters
stacked_model = StackingRegressor(estimators=base_estimators, final_estimator=RandomForestRegressor(), cv=5)

# Fit the StackingRegressor
stacked_model.fit(X_train, y_train)

# Predict with the StackingRegressor
y_pred = stacked_model.predict(X_test)

# Calculate the RMSLE
msle = mean_squared_log_error(y_pred, y_test)
rmsle = np.sqrt(msle)

print('RMSLE for the data (StackedRegressor):', rmsle)
```
