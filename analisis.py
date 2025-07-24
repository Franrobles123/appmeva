

import pandas as pd
import numpy as np

# Crear el DataFrame manualmente
data = {
    "Equipos": ["Equipo 1", "Equipo 2", "Equipo 3", "Equipo 4", "Medrano", "Viamonte", "Cetro"],
    "Febrero": ["0:10", "4:35", "1:30", "0:35", "1:25", "1:00", "2:40"],
    "Marzo": ["0:00", "0:35", "1:20", "0:20", "1:04", "0:45", "2:40"],
    "Abril": ["3:50", "9:25", "1:15", "0:20", "0:00", "0:00", "4:45"],
    "Mayo": ["0:45", "1:55", "0:15", "0:00", "1:00", "3:30", "7:20"],
    "Junio": ["0:00", "1:00", "1:30", "0:15", "1:00", "0:00", "13:00"],
    "Julio": ["0:55", "2:35", "1:15", "0:50", "0:35", "0:00", "10:00"],
    "Agosto": ["2:50", "1:40", "7:00", "3:15", "1:00", "0:40", "1:05"],
    "Septiembre": ["4:35", "0:25", "5:55", "3:05", "1:10", "0:00", "4:10"],
    "Octubre": ["1:15", "2:20", "0:35", "1:40", "2:30", "3:55", "1:15"],
    "Noviembre": ["0:30", "4:00", "3:50", "1:00", "2:00", "0:30", "2:00"],
    "Diciembre": ["1:05", "0:15", "8:35", "1:40", "0:30", "0:00", "0:20"]
}

# Convertir a DataFrame
df = pd.DataFrame(data)


# Función para convertir strings de tiempo "H:MM" a horas decimales
def time_to_decimal(t):
    h, m = map(int, t.split(":"))
    return h + m / 60

# Convertir todas las columnas de tiempo
for col in df.columns[1:]:
    df[col] = df[col].apply(time_to_decimal)

# Calcular promedio mensual por equipo
df['Promedio Mensual'] = df.iloc[:, 1:].mean(axis=1)

# Definir agrupaciones por ubicación
central = ['Equipo 1', 'Equipo 2', 'Equipo 3', 'Equipo 4']
centro = ['Cetro']
viamonte = ['Viamonte']
medrano = ['Medrano']

# Calcular totales por sede
suma_central = df[df['Equipos'].isin(central)].set_index('Equipos').sum(axis=1).sum()
suma_cetro = df[df['Equipos'].isin(centro)].set_index('Equipos').sum(axis=1).values[0]
suma_viamonte = df[df['Equipos'].isin(viamonte)].set_index('Equipos').sum(axis=1).values[0]
suma_medrano = df[df['Equipos'].isin(medrano)].set_index('Equipos').sum(axis=1).values[0]

# Crear serie de comparación
comparacion_ubicaciones = pd.Series({
    'Central': suma_central,
    'Cetro': suma_cetro,
    'Viamonte': suma_viamonte,
    'Medrano': suma_medrano
})

# Mostrar resultados
print(df)
print("\nComparación entre ubicaciones (Total anual en horas):")
print(comparacion_ubicaciones)

import matplotlib.pyplot as plt

# --- 1. Gráfico de barras: Total anual por ubicación ---
plt.figure(figsize=(8, 5))
comparacion_ubicaciones.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Total anual de mantenimiento por sede')
plt.ylabel('Horas totales')
plt.xlabel('Ubicación')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- 2. Gráfico de líneas: Evolución mensual por equipo ---
# Reorganizar datos para graficar
df_lineas = df.drop(columns='Promedio Mensual').set_index('Equipos').T

plt.figure(figsize=(12, 6))
for equipo in df_lineas.columns:
    plt.plot(df_lineas.index, df_lineas[equipo], marker='o', label=equipo)

plt.title('Evolución mensual del mantenimiento por equipo')
plt.ylabel('Horas de mantenimiento')
plt.xlabel('Mes')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

monthly_totals = df.drop(columns='Promedio Mensual').set_index('Equipos').T

plt.figure(figsize=(12, 6))
monthly_totals.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
plt.title("Mantenimiento mensual acumulado por equipo (barras apiladas)")
plt.ylabel("Horas")
plt.xlabel("Mes")
plt.xticks(rotation=45)
plt.legend(title="Equipo", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Reorganizar el DataFrame a formato largo
df_largo = df.drop(columns='Promedio Mensual').melt(id_vars='Equipos', var_name='Mes', value_name='Horas')

# Codificar las categorías (Equipos y Mes) como variables numéricas
df_largo_encoded = df_largo.copy()
df_largo_encoded['Equipo_ID'] = df_largo_encoded['Equipos'].astype('category').cat.codes
df_largo_encoded['Mes_ID'] = df_largo_encoded['Mes'].astype('category').cat.codes

# Dataset para modelo: variables numéricas
X = df_largo_encoded[['Equipo_ID', 'Mes_ID', 'Horas']]

# Aplicar Isolation Forest
model = IsolationForest(contamination=0.1, random_state=42)
df_largo_encoded['Anomalia'] = model.fit_predict(X)

# Marcar las anomalías detectadas
df_anomalias = df_largo_encoded[df_largo_encoded['Anomalia'] == -1]

# Mostrar resultados
print(df_anomalias[['Equipos', 'Mes', 'Horas']])

df_anomalias[['Equipos', 'Mes', 'Horas']]

# Volver a formar el DataFrame de líneas por equipo y mes
df_lineas = df.drop(columns='Promedio Mensual').set_index('Equipos').T

# Crear gráfico de líneas con anomalías destacadas
plt.figure(figsize=(12, 6))

# Graficar todas las líneas
for equipo in df_lineas.columns:
    plt.plot(df_lineas.index, df_lineas[equipo], marker='o', label=equipo)

# Superponer puntos de anomalías
for _, row in df_anomalias.iterrows():
    plt.plot(row['Mes'], row['Horas'], 'ro', markersize=10, label='Anomalía' if 'Anomalía' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title('Evolución mensual del mantenimiento con anomalías destacadas')
plt.ylabel('Horas de mantenimiento')
plt.xlabel('Mes')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# app.py
import streamlit as st
import pandas as pd

# Título
st.title("Registro de mantenimiento")

# Cargar archivo existente
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['Equipo', 'Mes', 'Horas'])

# Formulario para agregar nuevo registro
with st.form("nuevo_registro"):
    equipo = st.selectbox("Equipo", ['Equipo 1', 'Equipo 2', 'Equipo 3', 'Equipo 4', 'Medrano', 'Viamonte', 'Centro'])
    mes = st.selectbox("Mes", ['Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'])
    horas = st.number_input("Horas de mantenimiento", min_value=0.0, step=0.25)

    submitted = st.form_submit_button("Agregar")

    if submitted:
        nuevo = pd.DataFrame({'Equipo': [equipo], 'Mes': [mes], 'Horas': [horas]})
        st.session_state.df = pd.concat([st.session_state.df, nuevo], ignore_index=True)
        st.success("Registro agregado")

# Mostrar tabla actualizada
st.subheader("Registros actuales")
st.dataframe(st.session_state.df)

# Descargar como CSV
csv = st.session_state.df.to_csv(index=False).encode('utf-8')
st.download_button("Descargar CSV", csv, "mantenimiento.csv", "text/csv")