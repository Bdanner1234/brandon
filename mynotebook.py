import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv('/Users/brandondanner/School-Repository/brandon/vehicles_us.csv')


# Line plot for average price per model year
avg_price_per_year = df.groupby('model_year')['price'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=avg_price_per_year, x='model_year', y='price')
plt.title('Average Vehicle Price by Model Year')
plt.xlabel('Model Year')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
st.pyplot(plt)  # Use st.pyplot to show the Seaborn plot
plt.close()  # Close the plot

# Histogram for vehicle condition by model year
fig2 = px.histogram(df, x='model_year', color='condition', barmode='overlay',
                    title='Histogram of Vehicle Condition by Model Year',
                    histnorm='probability density')
st.plotly_chart(fig2)

# Price distribution by manufacturer
unique_models = df['model'].unique()
traces = []
for m in unique_models:
    filtered_df = df[df['model'] == m]
    traces.append(go.Histogram(
        x=filtered_df['price'],
        name=m,
        opacity=0.75,
        histnorm='probability density',
        visible=(m == unique_models[0])
    ))
fig3 = go.Figure(data=traces)
fig3.update_layout(title='Price Distribution by Manufacturer',
                   xaxis_title='Price',
                   yaxis_title='Density',
                   barmode='overlay')
st.plotly_chart(fig3)

# Scatter plot for odometer vs price
plt.figure(figsize=(8, 6))
df.plot(kind='scatter', x='odometer', y='price', alpha=0.36)
plt.title('Odometer vs Price')
plt.xlabel('Odometer')
plt.ylabel('Price')
st.pyplot(plt)  # Use st.pyplot to show the scatter plot
plt.close()

# Inform the user
st.write("All figures displayed within the Streamlit app.")
