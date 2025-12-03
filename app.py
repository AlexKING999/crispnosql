import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Predicción de Incumplimiento CRISP-DM", layout="wide", page_icon="💳")

st.title("💳 Ciclo CRISP-DM con Firebase y Streamlit")

# --- 1. CONEXIÓN SEGURA A FIREBASE ---
if not firebase_admin._apps:
    try:
        # Intenta cargar las credenciales desde Streamlit Secrets
        key_dict = dict(st.secrets["firebase"])
        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred)
        st.sidebar.success("Conexión a Firebase OK.")
    except KeyError:
        st.sidebar.error("❌ ERROR: No se encontraron las claves 'firebase' en Streamlit Secrets.")
        st.stop()
    except Exception as e:
        st.sidebar.error(f"❌ Error crítico en Secrets/Inicialización: {e}")
        st.stop()

db = firestore.client()

# --- 2. FUNCIÓN DE CARGA RESILIENTE (CON CACHÉ) ---
@st.cache_data(ttl=600)
def load_data_from_firestore():
    """Carga datos desde Firestore, normaliza columnas y maneja errores."""
    try:
        users_ref = db.collection('credito_clientes') 
        docs = users_ref.stream()
        data = [doc.to_dict() for doc in docs]
        
        if not data:
            st.warning("⚠️ Firebase: No se encontraron documentos en 'credito_clientes'.")
            return pd.DataFrame() 
            
        df = pd.DataFrame(data)
        st.sidebar.info(f"✅ Documentos cargados: {len(df)} registros.")

        # *** CORRECCIÓN CRÍTICA: NORMALIZACIÓN DE COLUMNAS ***
        # Esto asegura que 'EDUCATION' siempre sea 'education' y previene el KeyError
        df.columns = [str(col).lower() for col in df.columns] 

        # Conversión de tipos numérica (necesaria tras cargar desde Firestore)
        for col in df.columns:
            try:
                # La conversión a float antes de int evita errores de pérdida de precisión con NaN
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64', errors='ignore') 
            except:
                pass
                
        # Eliminar columna innecesaria si existe
        if 'unnamed:_0' in df.columns: 
            df = df.drop(columns=['unnamed:_0'])
            
        return df

    except Exception as e:
        st.error(f"❌ Error al cargar datos de Firestore. Revisa las reglas de seguridad: {e}")
        return pd.DataFrame()


# --- EJECUCIÓN PRINCIPAL Y CONTROL DE FLUJO ---
df_raw = load_data_from_firestore()

# Si el DataFrame está vacío, detenemos la ejecución de las fases de análisis
if df_raw.empty:
    st.error("🛑 La aplicación no puede continuar. No se cargaron datos de Firebase. Asegúrate de que el script de ingesta se haya ejecutado exitosamente.")
    st.stop()
    
st.sidebar.subheader("Exploración Inicial")
st.sidebar.dataframe(df_raw.head())


# --- NAVEGACIÓN CRISP-DM (Las 6 fases) ---
tabs = st.tabs(["1. Negocio", "2. Adquisición y Comprensión", "3. Preparación", "4. Modelado", "5. Evaluación", "6. Despliegue"])

# ==========================================
# FASE 1: COMPRENSIÓN DEL NEGOCIO
# ==========================================
with tabs[0]:
    st.header("📈 Fase 1: Comprensión del Negocio")
    st.markdown("""
    El objetivo es predecir la probabilidad de que un cliente **incumpla** el pago de su tarjeta de crédito el próximo mes (`default_payment_next_month`).
    """)

# ==========================================
# FASE 2: ADQUISICIÓN Y COMPRENSIÓN DE DATOS
# ==========================================
with tabs[1]:
    st.header("📊 Fase 2: Adquisición y Comprensión de Datos")
    st.subheader("Datos Cargados")
    st.info(f"Se han cargado {df_raw.shape[0]} registros con {df_raw.shape[1]} columnas desde Firestore.")
    
    st.dataframe(df_raw.head())
    
    st.subheader("Distribución de la Variable Objetivo")
    # Aseguramos que la columna esté normalizada
    target_col = 'default_payment_next_month'
    if target_col in df_raw.columns:
        default_counts = df_raw[target_col].value_counts().reset_index()
        default_counts.columns = ['Incumplimiento (0=No, 1=Sí)', 'Clientes']
        fig = px.pie(default_counts, 
                     names='Incumplimiento (0=No, 1=Sí)', 
                     values='Clientes', 
                     title='Porcentaje de Clientes que Incumplirán el Próximo Mes')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"La columna objetivo '{target_col}' no se encontró en el DataFrame. Revisa la ingesta.")

# ==========================================
# FASE 3: PREPARACIÓN DE DATOS (Fase del KeyError)
# ==========================================
with tabs[2]:
    st.header("🧹 Fase 3: Preparación de Datos")

    df = df_raw.copy()
    
    st.subheader("Estrategia de Transformación")
    st.markdown("""
    1. **Agrupación Categórica:** Los códigos `0`, `5`, `6` en `education` y `0` en `marriage` representan categorías desconocidas o agrupables, y se consolidan para simplificar.
    2. **Definición de Features (Características):** Se seleccionan las columnas clave para el modelado.
    """)

    # 1. Limpieza y Agrupación (Estas líneas ahora son seguras)
    df['education'] = df['education'].replace({0: 4, 5: 4, 6: 4}) 
    df['marriage'] = df['marriage'].replace({0: 3})               
    
    st.success("Limpieza de datos (education y marriage) aplicada.")
    st.dataframe(df[['education', 'marriage']].value_counts().head())

    # 2. Definición de Features
    FEATURES = ['limit_bal', 'age', 'sex', 'education', 'marriage', 
                'pay_0', 'bill_amt1', 'pay_amt1']
    TARGET = 'default_payment_next_month'
    
    # Manejo de columnas faltantes
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        st.error(f"🚨 Faltan columnas clave para el modelado: {missing_features}. Revisa la ingesta original.")
        st.stop()

    X = df[FEATURES]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    st.subheader("Dataset Particionado")
    st.write(f"Tamaño de entrenamiento (X_train): {X_train.shape[0]} filas")
    st.write(f"Tamaño de prueba (X_test): {X_test.shape[0]} filas")

# ==========================================
# FASE 4: MODELADO
# ==========================================
with tabs[3]:
    st.header("⚙️ Fase 4: Modelado")
    
    @st.cache_resource
    def train_model(X_train, y_train):
        st.info("Entrenando Random Forest Classifier...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    if 'X_train' in locals():
        model = train_model(X_train, y_train)
        st.success("Modelo Random Forest entrenado exitosamente.")
        st.write(model)
    else:
         st.warning("Asegúrate de haber completado la Fase 3.")

# ==========================================
# FASE 5: EVALUACIÓN
# ==========================================
with tabs[4]:
    st.header("💯 Fase 5: Evaluación")
    
    if 'model' in locals():
        y_pred = model.predict(X_test)
        
        st.subheader("Métricas de Rendimiento")
        
        accuracy = accuracy_score(y_test, y_pred)
        st.metric(label="Precisión (Accuracy)", value=f"{accuracy:.2f}")
        
        st.subheader("Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        st.code(cm)
        
        st.subheader("Reporte de Clasificación")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df)
    else:
        st.warning("El modelo no ha sido entrenado. Ejecuta la Fase 4.")

# ==========================================
# FASE 6: DESPLIEGUE (Predicción de Usuario)
# ==========================================
with tabs[5]:
    st.header("🚀 Fase 6: Despliegue (Predicción)")
    
    if 'model' in locals():
        st.subheader("Introduce los datos del cliente para la predicción:")
        
        # Formulario de entrada de datos
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                limit_bal = st.number_input("Límite de Crédito (LIMIT_BAL)", min_value=10000, value=100000, step=1000)
                age = st.slider("Edad (AGE)", min_value=21, max_value=79, value=35)
                sex = st.selectbox("Género (SEX)", options=[1, 2], format_func=lambda x: "Hombre" if x == 1 else "Mujer")
            
            with col2:
                education_map = {1: 'Posgrado', 2: 'Universidad', 3: 'Secundaria', 4: 'Otro/Desconocido'}
                education = st.selectbox("Educación", options=list(education_map.keys()), format_func=lambda x: education_map[x])
                
                marriage_map = {1: 'Casado', 2: 'Soltero', 3: 'Otro/Desconocido'}
                marriage = st.selectbox("Estado Civil", options=list(marriage_map.keys()), format_func=lambda x: marriage_map[x])

                pay_0 = st.slider("Estado de Pago (PAY_0) - El mes pasado", min_value=-2, max_value=8, value=0)
            
            with col3:
                bill_amt1 = st.number_input("Monto de Factura (BILL_AMT1) - El mes pasado", value=5000)
                pay_amt1 = st.number_input("Monto Pagado (PAY_AMT1) - El mes pasado", value=2000)
                
            submitted = st.form_submit_button("Predecir Incumplimiento")
            
            if submitted:
                input_data = pd.DataFrame({
                    'limit_bal': [limit_bal], 
                    'age': [age], 
                    'sex': [sex], 
                    'education': [education], 
                    'marriage': [marriage], 
                    'pay_0': [pay_0], 
                    'bill_amt1': [bill_amt1], 
                    'pay_amt1': [pay_amt1]
                })
                
                # Predicción
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1]

                st.subheader("Resultado de la Predicción")
                if prediction == 1:
                    st.error(f"🔴 Se predice **INCUMPLIMIENTO** con una probabilidad del **{probability*100:.2f}%**.")
                else:
                    st.success(f"🟢 Se predice **NO INCUMPLIMIENTO** con una probabilidad del **{(1-probability)*100:.2f}%**.")
    else:
        st.warning("El modelo no está disponible. Asegúrate de completar las Fases 3 y 4.")



