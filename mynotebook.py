import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv('/Users/brandondanner/School-Repository/brandon/vehicles_us.csv')


# Line plot for average price per model year (Matplotlib/Seaborn)
avg_price_per_year = df.groupby('model_year')['price'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=avg_price_per_year, x='model_year', y='price')
plt.title('Average Vehicle Price by Model Year')
plt.xlabel('Model Year')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
st.pyplot(plt)  # Correct: st.pyplot for Matplotlib/Seaborn plot
plt.close()  # Close the plot

# Histogram for vehicle condition by model year (Plotly)
fig2 = px.histogram(df, x='model_year', color='condition', barmode='overlay',
                    title='Histogram of Vehicle Condition by Model Year',
                    histnorm='probability density')

st.plotly_chart(fig2)  # Correct: st.plotly_chart for Plotly figure

# Price distribution by manufacturer (Plotly)

# Get unique manufacturers for the dropdown
unique_models = df['model'].unique()

# Create the histogram traces for each manufacturer
traces = []
for m in unique_models:
    filtered_df = df[df['model'] == m]
    traces.append(go.Histogram(
        x=filtered_df['price'],
        name=m,
        opacity=0.75,
        histnorm='probability density',
        visible=(m == unique_models[0])  # Show the first manufacturer by default
    ))

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

# Show the figure using Streamlit
st.plotly_chart(fig)  # Correct: st.plotly_chart for Plotly figure

# Scatter plot for odometer vs price (Matplotlib)
plt.figure(figsize=(8, 6))
df.plot(kind='scatter', x='odometer', y='price', alpha=0.36)
plt.title('Odometer vs Price')
plt.xlabel('Odometer')
plt.ylabel('Price')
plt.xlim(0, 500000)  # Set the x-axis limits (adjust as needed)
plt.ylim(0, 100000)
st.pyplot(plt)  # Correct: st.pyplot for Matplotlib/Seaborn plot
plt.close()
