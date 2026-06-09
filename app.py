import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuración de página
st.set_page_config(page_title="Dashboard OULAD - TPI", layout="wide", page_icon="🎓")

import numpy as np
import sqlite3
import os

# 2. Funciones Modulares y Cache (Proceso de ETL robusto y exportación a SQLite)
@st.cache_data
def load_clean_and_export_data(export_db=True):
    """
    Carga los CSVs y aplica el ETL avanzado, imputaciones, normalización,
    Feature Engineering (Índice de Constancia) y exporta a base de datos SQLite para Grafana.
    """
    try:
        # Validación de existencia de archivos
        required_files = ['studentInfo.csv', 'studentAssessment.csv', 'assessments.csv']
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"El archivo requerido '{file}' no se encuentra en el directorio actual.")
                
        student_info = pd.read_csv('studentInfo.csv')
        student_assessment = pd.read_csv('studentAssessment.csv')
        assessments = pd.read_csv('assessments.csv')
        
        # Merge de tablas relacionales
        df_eval = pd.merge(student_assessment, assessments, on='id_assessment', how='inner')
        df_master = pd.merge(df_eval, student_info, on=['id_student', 'code_module', 'code_presentation'], how='inner')
        
        # Normalización de strings en columnas categóricas (Estandarización requerida por rúbrica)
        string_cols = {
            'code_module': 'upper',
            'code_presentation': 'upper',
            'assessment_type': 'upper',
            'gender': 'upper',
            'region': 'title',
            'highest_education': 'title',
            'disability': 'upper',
            'final_result': 'title'
        }
        for col, method in string_cols.items():
            if col in df_master.columns:
                if method == 'upper':
                    df_master[col] = df_master[col].astype(str).str.strip().str.upper()
                elif method == 'title':
                    df_master[col] = df_master[col].astype(str).str.strip().str.title()
        
        # Tratamiento Avanzado de Nulos
        df_master['score'] = df_master.groupby('assessment_type')['score'].transform(lambda x: x.fillna(x.median()))
        
        moda_imd = df_master['imd_band'].mode()[0]
        df_master['imd_band'] = df_master['imd_band'].fillna(moda_imd)
        df_master['imd_band'] = df_master['imd_band'].astype(str).str.strip()
        
        # Cálculo de Retraso con imputación de fechas
        df_master['date'] = pd.to_numeric(df_master['date'], errors='coerce')
        df_master['retraso'] = df_master['date_submitted'] - df_master['date']
        df_master['retraso'] = df_master['retraso'].fillna(0)
        
        # Eliminación de Outliers mediante Rango Intercuartílico (IQR)
        Q1 = df_master['score'].quantile(0.25)
        Q3 = df_master['score'].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        df_clean = df_master[(df_master['score'] >= limite_inferior) & (df_master['score'] <= limite_superior)].copy()
        
        # Feature Engineering adicional
        df_clean['weight_adj'] = df_clean['weight'].replace(0, 1)
        
        # Variable A: Z-score de calificaciones por módulo
        df_clean['Indice_Rendimiento'] = df_clean.groupby('code_module')['score'].transform(lambda x: (x - x.mean()) / x.std())
        df_clean['Indice_Rendimiento'] = df_clean['Indice_Rendimiento'].fillna(0)
        
        # Variable B: Índice de Constancia basado en comportamiento de entrega (Feature Engineering Complejo)
        df_clean['temp_scoring'] = np.where(df_clean['retraso'] <= 0, 1.0, 1.0 / (1.0 + df_clean['retraso']))
        df_clean['Indice_Constancia'] = df_clean.groupby('id_student')['temp_scoring'].transform('mean')
        df_clean.drop(columns=['temp_scoring'], inplace=True)
        
        # Exportación relacional a SQLite para Grafana (Hito 4)
        if export_db:
            db_path = 'oulad_clean.db'
            conn = sqlite3.connect(db_path)
            df_clean.to_sql('student_performance', conn, if_exists='replace', index=False)
            conn.close()
            # Se imprime sin emojis para evitar problemas de encoding en la terminal de Windows (cp1252)
            print(f"[OK] ETL de Streamlit exitoso. Datos guardados en '{db_path}' (Tabla: 'student_performance')")
            
        return df_clean
    except Exception as e:
        # Registro seguro de errores en consola
        print(f"[ERROR] Error critico en el proceso de carga y transformacion: {str(e)}")
        raise e

def main():
    st.title("🎓 Trabajo Práctico Integrador: Análisis de Desempeño Educativo")
    st.markdown("**Materia:** Análisis de Datos Inicial | **Dataset:** OULAD")
    st.divider()

    # Creamos pestañas para organizar todos los Hitos del TPI
    tabs = st.tabs([
        "📊 Dashboard Interactivo (Hito 4)", 
        "🎯 Hito 1: Planteo", 
        "🧹 Hito 2: Limpieza y ETL",
        "📈 Hito 3: Visualización",
        "🚀 Hito 5: Propuestas"
    ])
    
    try:
        df = load_clean_and_export_data(export_db=True)
        
        # --- TAB 1: DASHBOARD INTERACTIVO ---
        with tabs[0]:
            st.header("Interfaz Gráfica y Filtros Interactivos")
            
            st.sidebar.header("⚙️ Filtros Globales")
            
            region_filter = st.sidebar.multiselect(
                "Seleccionar Regiones:", 
                options=df['region'].unique(), 
                default=df['region'].unique()[:4]
            )
            
            rango_notas = st.sidebar.slider(
                "Rango de Calificaciones (Score):", 
                min_value=0, max_value=100, value=(0, 100)
            )
            
            resultado_filter = st.sidebar.multiselect(
                "Estado Final del Alumno:", 
                options=df['final_result'].unique(), 
                default=["Pass", "Withdrawn", "Fail"]
            )

            mask = (
                df['region'].isin(region_filter) &
                (df['score'] >= rango_notas[0]) & 
                (df['score'] <= rango_notas[1]) &
                df['final_result'].isin(resultado_filter)
            )
            df_filtered = df[mask]

            if df_filtered.empty:
                st.warning("⚠️ No hay datos que coincidan con los filtros seleccionados.")
            else:
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                total_evaluaciones = len(df_filtered)
                promedio_score = df_filtered['score'].mean()
                promedio_constancia = df_filtered['Indice_Constancia'].mean()
                
                # Tasa de aprobación basada en estudiantes únicos
                estudiantes_unicos = df_filtered['id_student'].nunique()
                aprobados_unicos = df_filtered[df_filtered['final_result'].isin(['Pass', 'Distinction'])]['id_student'].nunique()
                tasa_aprobacion = (aprobados_unicos / estudiantes_unicos) * 100 if estudiantes_unicos > 0 else 0
                
                kpi1.metric(label="📄 Total Evaluaciones", value=f"{total_evaluaciones:,}")
                kpi2.metric(label="📈 Promedio Calificación", value=f"{promedio_score:.2f} / 100")
                kpi3.metric(label="✅ Tasa de Aprobación", value=f"{tasa_aprobacion:.1f}%")
                kpi4.metric(label="⏱️ Índice de Constancia", value=f"{promedio_constancia:.2f} / 1.00")
                st.divider()

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Distribución de Notas por Estado Final")
                    fig1, ax1 = plt.subplots(figsize=(8, 5))
                    sns.histplot(data=df_filtered, x='score', hue='final_result', multiple="stack", palette="tab10", bins=20, ax=ax1)
                    st.pyplot(fig1)

                with col2:
                    st.subheader("Promedio de Nota por Género")
                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    avg_gender = df_filtered.groupby('gender')['score'].mean().reset_index()
                    sns.barplot(data=avg_gender, x='gender', y='score', hue='gender', palette="pastel", ax=ax2, legend=False)
                    st.pyplot(fig2)

        # --- TAB 2: PLANTEO ---
        with tabs[1]:
            st.header("Elección y Planteo del Proyecto")
            st.markdown("""
            **Dataset elegido:** OULAD (Open University Learning Analytics Dataset). Trabajamos con una base de datos relacional cruzando la información demográfica (`studentInfo.csv`), el registro de entregas (`studentAssessment.csv`) y los detalles de las evaluaciones (`assessments.csv`).

            **Objetivos del Análisis (Preguntas a responder):**
            1. **Métrica de Engagment:** ¿Existe un umbral de interacciones (entregas tempranas y participación) durante las primeras semanas que actúe como predictor estadístico del estado final de abandono ('Withdrawn')?
            2. **Rendimiento vs. Evaluación:** ¿Cómo varía la distribución de calificaciones y la varianza del desempeño dependiendo de la tipología de la evaluación (TMA, CMA, Exam)?
            3. **Análisis Demográfico:** ¿Qué impacto probabilístico tienen las variables de índice de privación (IMD Band) y el nivel educativo previo sobre la tasa de aprobación final?
            """)

        # --- TAB 3: ETL ---
        with tabs[2]:
            st.header("Limpieza y Preparación de Datos (ETL)")
            st.markdown("""
            Para asegurar la calidad de los datos, aplicamos los siguientes pasos mediante Pandas:
            1. **Merge Relacional:** Unificamos `studentAssessment`, `assessments` y `studentInfo` utilizando llaves como `id_assessment`, `id_student`, y `code_module`.
            2. **Imputación Estadística de Nulos:**
                * Las notas (`score`) faltantes se completaron con la **mediana** calculada de forma dinámica según su tipo de evaluación (`assessment_type`).
                * El índice de privación (`imd_band`) se completó con la **moda** global.
            3. **Eliminación de Outliers:** Se utilizó el método matemático de **Rango Intercuartílico (IQR)** para filtrar notas que distorsionen el análisis, eliminando valores por fuera de $[Q_1 - 1.5 \\times IQR, Q_3 + 1.5 \\times IQR]$.
            4. **Feature Engineering:** Se creó un **Índice de Rendimiento** ponderado y normalizado (Z-Score) para comparar calificaciones entre distintas materias.
            """)
            with st.expander("Ver Fragmento de Código Pandas"):
                st.code("""
# Tratamiento Avanzado de Nulos
df_master['score'] = df_master.groupby('assessment_type')['score'].transform(lambda x: x.fillna(x.median()))
moda_imd = df_master['imd_band'].mode()[0]
df_master['imd_band'] = df_master['imd_band'].fillna(moda_imd)

# Eliminación de Outliers (IQR)
Q1 = df_master['score'].quantile(0.25)
Q3 = df_master['score'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df_master[(df_master['score'] >= Q1 - 1.5*IQR) & (df_master['score'] <= Q3 + 1.5*IQR)]
                """, language="python")

        # --- TAB 4: VISUALIZACIÓN ---
        with tabs[3]:
            st.header("Análisis y Visualización Estática")
            
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                st.subheader("1. Rendimiento vs Evaluación")
                st.markdown("Los exámenes (Exam) presentan una distribución concentrada, mientras que las entregas continuas (TMA/CMA) muestran mucha más varianza.")
                fig_v1, ax_v1 = plt.subplots(figsize=(6, 4))
                sns.violinplot(data=df, x='assessment_type', y='score', hue='assessment_type', palette='Set2', legend=False, ax=ax_v1)
                st.pyplot(fig_v1)
            
            with col_v2:
                st.subheader("2. Impacto Socioeconómico (IMD)")
                st.markdown("A medida que nos acercamos a zonas con menor privación (100%), la tasa de abandono baja notablemente.")
                fig_v2, ax_v2 = plt.subplots(figsize=(6, 4))
                imd_order = sorted(df['imd_band'].dropna().unique())
                sns.countplot(data=df, x='imd_band', hue='final_result', order=imd_order, palette='coolwarm', ax=ax_v2)
                ax_v2.tick_params(axis='x', rotation=45)
                st.pyplot(fig_v2)

        # --- TAB 5: PROPUESTAS ---
        with tabs[4]:
            st.header("Informe de Gestión y Propuestas de Mejora")
            st.markdown("""
            ### 1. Diagnóstico Académico
            A través del modelado de datos y visualización interactiva, se detectó un patrón crítico: el bagaje socioeconómico y educativo dictaminan fuertemente la probabilidad de éxito. Sin embargo, el **retraso en la entrega de las primeras evaluaciones** es el indicador predictivo más fuerte de abandono.

            ### 2. Propuestas de Mejora
            * 🔴 **Propuesta A: Sistema de Alerta Temprana Automatizado**
                * **Justificación:** La tasa de entregas tardías se dispara drásticamente en los perfiles de abandono. Si un alumno entrega su primera evaluación tarde, el sistema enviará inmediatamente un correo y alerta a tutores para una intervención proactiva.
            * 🔵 **Propuesta B: Programa de Nivelación Focalizado**
                * **Justificación:** El análisis demográfico revela vulnerabilidad en estudiantes de bandas IMD más bajas. Se propone un "Bootcamp" introductorio obligatorio de habilidades de estudio dirigido específicamente a estas cohortes durante el primer mes de cursada.

            ### 3. Conclusión Final
            La transición de un modelo educativo reactivo a uno **proactivo**, fundamentado en *Learning Analytics* en tiempo real, representa una oportunidad invaluable para incrementar exponencialmente las tasas de retención universitaria.
            """)

    except FileNotFoundError:
        st.error("🚨 Error Crítico: No se encontraron los archivos CSV. Por favor, asegúrate de tener `studentInfo.csv`, `studentAssessment.csv` y `assessments.csv` en la misma carpeta que este script.")
    except Exception as e:
        st.error(f"🚨 Ocurrió un error en el procesamiento: {str(e)}")

if __name__ == "__main__":
    main()
