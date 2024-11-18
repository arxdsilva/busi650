import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('./Superstore_dataset.csv', index_col=0, parse_dates=['Order Date'])

ds = data[['Order Date', 'Sales', 'Quantity', 'Discount', 'Profit']].sort_values(by='Order Date')
ds.head()

ds.set_index('Order Date', inplace=True)
ds['Sales'].plot.scatter(x=ds.index, y=ds['Sales'])
plt.show()

# Describe the overall trend
sales_trend = ds['Sales'].resample('M').sum()
sales_trend.plot()
plt.title('Monthly Sales Trend')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Analyze the discount trend
discount_trend = ds['Discount'].resample('M').mean()
discount_trend.plot()
plt.title('Monthly Discount Trend')
plt.xlabel('Date')
plt.ylabel('Average Discount')
plt.show()

# Identify and describe missing values. 

# Identify and describe missing values. 
missing_values = ds.isnull().sum()
print("Missing Values in Each Column:")
print(missing_values)
# Remove missing values
ds.dropna(inplace=True)

# Apply moving average to smooth out the data
window_size = 30  # A window size of 30 is appropriate for a 10k lines dataset
# Determine an appropriate window size

# A common approach is to use a window size that captures the seasonality in the data.
# For example, if the data has a weekly seasonality, a window size of 7 might be appropriate.
# If the data has monthly seasonality, a window size of 30 might be appropriate.

# Here, we can use a heuristic approach to determine the window size.
# We can start with a smaller window size and gradually increase it to see the effect on the moving average.

# For this example, let's start with a window size of 30 days (approximately one month)
window_size = 30

# Apply moving average on Sales
ds['Sales_MA'] = ds['Sales'].rolling(window=window_size).mean()
ds['Sales_MA'].plot()
plt.title('Sales Moving Average')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Apply moving average on Profit
ds['Profit_MA'] = ds['Profit'].rolling(window=window_size).mean()
ds['Profit_MA'].plot()
plt.title('Profit Moving Average')
plt.xlabel('Date')
plt.ylabel('Profit')
plt.show()

# Identify and describe outliers using histograms.
# Identify and describe outliers using histograms.

# Plot histogram for Sales
plt.hist(ds['Sales'], bins=50, edgecolor='k')
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for Profit
plt.hist(ds['Profit'], bins=50, edgecolor='k')
plt.title('Profit Distribution')
plt.xlabel('Profit')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for Discount
plt.hist(ds['Discount'], bins=50, edgecolor='k')
plt.title('Discount Distribution')
plt.xlabel('Discount')
plt.ylabel('Frequency')
plt.show()

# Perform correlation analysis on the cleaned dataset
correlation_matrix = ds[['Sales', 'Quantity', 'Discount', 'Profit']].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Interpretation of correlation coefficients
# A correlation coefficient close to 1 implies a strong positive relationship,
# close to -1 implies a strong negative relationship, and close to 0 implies no relationship.
