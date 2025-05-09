import streamlit as st
import pandas as pd
import plotly.express as px

# Cargar el dataset
pisos = pd.read_csv('houses_Madrid.csv')

# Eliminar la columna innecesaria y establecer el índice
pisos.drop(columns=['Unnamed: 0'], inplace=True)
pisos.set_index('id', inplace=True)

# Título de la aplicación
st.title("Explorador de Precios de Vivienda en Madrid")

# Mostrar el DataFrame (las primeras filas)
st.subheader("Primeras filas del dataset:")
st.dataframe(pisos.head())

# Mostrar información básica del dataset
st.subheader("Información del dataset:")
st.write(f"Número de filas y columnas: {pisos.shape}")

# Visualizaciones
st.subheader("Visualizaciones:")

# Histograma de precios
fig_precio = px.histogram(pisos, x='buy_price', title='Distribución de Precios de Vivienda')
st.plotly_chart(fig_precio)

# Dispersión de metros cuadrados vs. precio
fig_metros_precio = px.scatter(pisos, x='sq_mt_built', y='buy_price', title='Metros Cuadrados Construidos vs. Precio')
st.plotly_chart(fig_metros_precio)

# Estadísticas descriptivas
st.subheader("Estadísticas Descriptivas:")
st.dataframe(pisos[['buy_price', 'sq_mt_built', 'n_rooms', 'n_bathrooms']].describe())

# Aquí podríamos añadir filtros y más visualizaciones en el futuro

st.write("¡Esta es una versión inicial!")
