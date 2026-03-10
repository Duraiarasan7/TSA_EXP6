# Ex.No: 6               HOLT WINTERS METHOD
### Date: 



### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
# Import necessary modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose


data = pd.read_csv("/kaggle/input/time-series-melbourn/daily-minimum-temperatures-in-me.csv", on_bad_lines='skip')

data.columns = ["Date","Temp"]
data["Date"] = pd.to_datetime(data["Date"])

# Clean invalid values
data["Temp"] = data["Temp"].astype(str).str.replace("?", "", regex=False)
data["Temp"] = pd.to_numeric(data["Temp"], errors="coerce")
data.dropna(inplace=True)

data.set_index("Date", inplace=True)

data.head()


data_monthly = data["Temp"].resample('MS').mean()   # Month start

data_monthly.head()

# Plot the data
data_monthly.plot(title="Monthly Temperature Data")
plt.show()


scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1,1)).flatten(),
    index=data_monthly.index
)

scaled_data.plot(title="Scaled Data Plot")
plt.show()


decomposition = seasonal_decompose(data_monthly, model="additive", period=12)

decomposition.plot()
plt.show()


train_data = scaled_data[:int(len(scaled_data)*0.8)]
test_data = scaled_data[int(len(scaled_data)*0.8):]


model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()

test_predictions_add = model_add.forecast(steps=len(test_data))


ax = train_data.plot(figsize=(10,5))
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)

ax.legend(["train_data","test_predictions_add","test_data"])
ax.set_title("Visual Evaluation")
plt.show()


rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print("RMSE:", rmse)

print("Variance and Mean of dataset:")
print(np.sqrt(scaled_data.var()), scaled_data.mean())


final_model = ExponentialSmoothing(
    data_monthly,
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()

# Predict future values
final_predictions = final_model.forecast(steps=int(len(data_monthly)/4))


ax = data_monthly.plot(figsize=(10,5))
final_predictions.plot(ax=ax)

ax.legend(["data_monthly","final_predictions"])
ax.set_xlabel("Months")
ax.set_ylabel("Temperature")
ax.set_title("Prediction")

plt.show()
```
### OUTPUT:
<img width="519" height="432" alt="image" src="https://github.com/user-attachments/assets/5681784f-371a-4057-b2d4-239f677ff7ee" />
<img width="652" height="431" alt="image" src="https://github.com/user-attachments/assets/87ba70ba-3620-4db0-9f27-03060514acb7" />
<img width="681" height="454" alt="image" src="https://github.com/user-attachments/assets/d40654ae-9314-4264-ae71-874c0ebbae78" />
<img width="849" height="500" alt="image" src="https://github.com/user-attachments/assets/81735cf1-31da-4e20-b7af-0cf48841a4d0" />
<img width="798" height="440" alt="image" src="https://github.com/user-attachments/assets/c63ee529-2e6b-43c2-86d4-37ede7a8a1d2" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
