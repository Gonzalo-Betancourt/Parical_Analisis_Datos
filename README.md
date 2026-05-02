# Dashboard OULAD - Trabajo Práctico Integrador 🎓

Este repositorio contiene el código fuente y el análisis de datos para el Trabajo Práctico Integrador de la materia "Análisis de Datos Inicial" (Tecnicatura Universitaria en Programación).

## 🚀 Contenido del Proyecto

1. **Jupyter Notebook (`Template_TPI_Analisis_Datos.ipynb`)**: 
   Contiene todo el proceso de Análisis de Datos, incluyendo la limpieza (ETL), tratamiento de nulos, detección de *outliers* y Feature Engineering, además de gráficos estadísticos con conclusiones relevantes.

2. **Dashboard Interactivo (`app.py`)**: 
   Aplicación web construida con **Streamlit** que permite explorar visualmente los resultados y aplicar filtros dinámicos sobre el dataset limpio.

3. **Datasets (`*.csv`)**: 
   Archivos originales extraídos del Open University Learning Analytics Dataset (OULAD) utilizados para el análisis:
   - `studentInfo.csv`
   - `studentAssessment.csv`
   - `assessments.csv`

## 🛠️ Requisitos de Instalación

```bash
# 1. Clonar el repositorio
git clone <https://github.com/Gonzalo-Betancourt/Parical1_Analisis_Datos>
cd Parical1_Analisis_Datos

# 2. Instalar las dependencias
pip install -r requirements.txt

# 3. Ejecutar el dashboard de Streamlit
streamlit run app.py
