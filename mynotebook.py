

# In[7]:

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/brandondanner/School-Repository/brandon/vehicles_us.csv')

# Perform basic EDA
print(df.head())
print(df.info())
print(df.describe())


fig = px.histogram(df, x='price', 
                   title='type',
                  y='model_year')
fig.show()


# In[8]:


sns.set(style='whitegrid')

   # Create a line plot for average price per model year
avg_price_per_year = df.groupby('model_year')['price'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=avg_price_per_year, x='model_year', y='price')
plt.title('Average Vehicle Price by Model Year')
plt.xlabel('Model Year')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.show()


fig = px.histogram(df, 
                      x='model_year', 
                      color='condition', 
                      barmode='overlay', 
                      title='Histogram of Vehicle Condition by Model Year',
                      labels={'model_year': 'Model Year', 'condition': 'Vehicle Condition'},
                      histnorm='probability density')

fig.update_layout(xaxis_title='Model Year',
                     yaxis_title='Density',
                     barmode='overlay',
                     hovermode='x unified')
fig.show()


import plotly.graph_objects as go

# Get unique manufacturers for the dropdown
unique_models = df['model'].unique()
print(1)
# Create the histogram traces for each manufacturer
traces = []
for m in unique_models:  # Use a different variable name here
    filtered_df = df[df['model'] == m]
    traces.append(go.Histogram(
        x=filtered_df['price'],
        name=m,
        opacity=0.75,
        histnorm='probability density',
        visible=(m == unique_models[0])  # Show the first manufacturer by default
    ))
print(2)
# Create the figure
fig = go.Figure(data=traces)

# Add dropdown buttons
fig.update_layout(
    title='Price Distribution by Manufacturer',
    xaxis_title='Price',
    yaxis_title='Density',
    barmode='overlay',
    updatemenus=[
        {
            'buttons': [
                {
                    'label': m,
                    'method': 'update',
                    'args': [{'visible': [m == model for model in unique_models]}]
                } for m in unique_models
            ],
            'direction': 'down',
            'showactive': True,
        }
    ]
)

# Show the figure
fig.show()

df.columns


df.plot(kind='scatter',
        title='Something',
        alpha=.36,
        figsize=[8, 6],
        xlabel='Odometer',
        ylabel='Price',
        x='odometer',
       y='price')
plt.show()

