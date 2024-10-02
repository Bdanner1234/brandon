import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv('./vehicles_us.csv')

st.header("The Average Vehicle Price by Model Year")
# Line plot for average price per model year (Matplotlib/Seaborn)
avg_price_per_year = df.groupby('model_year')['price'].mean().reset_index()
fig = px.line(avg_price_per_year, x='model_year', y='price', labels={'model_year': 'Model Year', 'price': 'Average Price'}, title='Average Vehicle Price by Model Year')
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig)
# Display the plot in Streamlit
#fig, ax = plt.subplots(figsize=(12, 6))
# sns.lineplot(data=avg_price_per_year, x='model_year', y='price', ax=ax)
# ax.set_xlabel('Model Year')
# ax.set_ylabel('Average Price')
# plt.xticks(rotation=45)
# st.pyplot(fig)  # Correct: st.pyplot for Matplotlib/Seaborn plot
# #plt.close()  # Close the plot

st.header("The Vehicle Condition by Model Year")
# Histogram for vehicle condition by model year (Plotly)
fig2 = px.histogram(df, x='model_year', color='condition', barmode='overlay',
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
st.header("Price Distribution by Manufacturer")
# Add dropdown buttons
fig.update_layout(
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
# Create the Plotly Express scatter plot 
st.header("The Odometer and Days Listed vs Price")
fig = px.scatter(df, x='odometer', y='price', opacity=0.36, labels={'odometer': 'Odometer', 'price': 'Price'}) 
fig.update_xaxes(range=[0, 500000]) 
fig.update_yaxes(range=[0, 100000])
# Display the chart in Streamlit 
# st.plotly_chart(fig)
# Create a checkbox for changing x-axis to 'days_listed'
use_days_listed = st.checkbox('Change x-axis to Days Listed')

# Set the x-axis label dynamically based on the checkbox state
x_axis = 'days_listed' if use_days_listed else 'odometer'

# Create the scatter plot with Plotly
fig = px.scatter(df, x=x_axis, y='price', opacity=0.36, labels={x_axis: x_axis.capitalize(), 'price': 'Price'})

# Update x-axis range based on the checkbox state
if x_axis == 'odometer':
    fig.update_xaxes(range=[0, 500000])  # For 'odometer'
else:
    fig.update_xaxes(range=[0, df['days_listed'].max()])  # For 'days_listed'

# Update y-axis range (kept constant)
fig.update_yaxes(range=[0, 100000])

# Display the chart in Streamlit (the chart will dynamically update based on checkbox state)
st.plotly_chart(fig)

# plt.figure(figsize=(8, 6))
# df.plot(kind='scatter', x='odometer', y='price', alpha=0.36)
# plt.xlabel('Odometer')
# plt.ylabel('Price')
# plt.xlim(0, 500000)  # Set the x-axis limits (adjust as needed)
# plt.ylim(0, 100000)
# st.pyplot(plt)  # Correct: st.pyplot for Matplotlib/Seaborn plot
# plt.close()
