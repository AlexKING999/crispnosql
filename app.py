import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# --- CONFIGURACIÓN E INICIALIZACIÓN ---
st.set_page_config(page_title="Predicción de Incumplimiento CRISP-DM", layout="wide", page_icon="💳")

st.title("💳 Ciclo CRISP-DM con Firebase y Streamlit")

# 1. CONEXIÓN SEGURA A FIREBASE
if not firebase_admin._apps:
    try:
        key_dict = dict(st.secrets["firebase"])
        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred)
        # No mostramos el éxito de la conexión aquí, lo hacemos en la función de carga.
    except Exception as e:
        st.sidebar.error(f"❌ Error crítico en Secrets/Inicialización: {e}")
        st.stop()

db = firestore.client()

# --- FUNCIÓN DE CARGA (CORREGIDA Y RESILIENTE) ---
@st.cache_data(ttl=600)
def load_data_from_firestore():
    try:
        users_ref = db.collection('credito_clientes') 
        docs = users_ref.stream()
        data = [doc.to_dict() for doc in docs]
        
        if not data:
            st.warning("⚠️ Firebase: No se encontraron documentos en 'credito_clientes'. ¿Corrió el script de ingesta?")
            return pd.DataFrame() # Retorna DataFrame vacío
            
        df = pd.DataFrame(data)
        st.sidebar.success(f"✅ Conexión y carga exitosa: {len(df)} registros.")

        # *** CORRECCIÓN CRÍTICA 1: NORMALIZACIÓN Y VERIFICACIÓN ***
        df.columns = [str(col).lower() for col in df.columns] 

        # Verificar que las columnas críticas existan después de la normalización
        required_cols = ['education', 'marriage', 'default_payment_next_month']
        for col in required_cols:
            if col not in df.columns:
                 st.error(f"🚨 Error: La columna requerida '{col}' no existe en la base de datos.")
                 return pd.DataFrame()

        # Conversión de tipos numérica
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
                
        if 'unnamed:_0' in df.columns: 
            df = df.drop(columns=['unnamed:_0'])
            
        return df

    except Exception as e:
        st.error(f"❌ Error al cargar datos de Firestore. Revisa las reglas de seguridad o la conexión: {e}")
        return pd.DataFrame()


# --- NAVEGACIÓN CRISP-DM (Las 6 fases) ---
df_raw = load_data_from_firestore()

tabs = st.tabs(["1. Negocio", "2. Adquisición y Comprensión", "3. Preparación", "4. Modelado", "5. Evaluación", "6. Despliegue"])

# --- Bloque de Control: Evita el KeyError si no hay datos ---
if df_raw.empty and tabs[1] or tabs[2] or tabs[3] or tabs[4] or tabs[5]:
    if not df_raw.empty: # Esta condición solo se cumple si hay un error y df_raw está vacío
        with tabs[1]:
            st.error("🛑 No se pueden mostrar las fases. El DataFrame está vacío.")
            st.stop()

# ==========================================
# FASE 3: PREPARACIÓN DE DATOS (Ahora segura)
# ==========================================
with tabs[2]:
    st.header("🧹 Fase 3: Preparación de Datos")

    df = df_raw.copy()
    
    st.subheader("Estrategia de Transformación")

    # 1. Limpieza y Agrupación (Esta línea ahora es segura gracias a la normalización en load_data)
    df['education'] = df['education'].replace({0: 4, 5: 4, 6: 4}) 
    df['marriage'] = df['marriage'].replace({0: 3})
    
    # 2. Definición de Features
    FEATURES = ['limit_bal', 'age', 'sex', 'education', 'marriage', 
                'pay_0', 'bill_amt1', 'pay_amt1']
    TARGET = 'default_payment_next_month'
    
    # 3. Codificación (One-Hot Encoding)
    df_prepared = pd.get_dummies(df[FEATURES], columns=['sex', 'education', 'marriage'], drop_first=True, dtype=int)
    df_prepared[TARGET] = df[TARGET]

    st.write("Datos listos para el Modelado:")
    st.code(f"Total de Características tras Encoding: {len(df_prepared.columns) - 1}")
    st.dataframe(df_prepared.head())
    
    st.session_state['df_prepared'] = df_prepared
    st.session_state['model_features'] = list(df_prepared.drop(columns=[TARGET]).columns)

# ==========================================
# FASE 4: MODELADO
# ==========================================
with tabs[3]:
    st.header("🤖 Fase 4: Modelado")

    if 'df_prepared' not in st.session_state:
        st.warning("⚠️ Primero ejecuta la Fase 3: Preparación de Datos.")
        st.stop()

    df_model = st.session_state['df_prepared']
    TARGET = 'default_payment_next_month'
    
    X = df_model.drop(columns=[TARGET])
    y = df_model[TARGET]

    # Split Estratificado (importante por el desbalance de clases)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    st.subheader("Algoritmo: Random Forest Classifier")
    
    # Entrenar
    model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, class_weight='balanced')
    with st.spinner("Entrenando modelo con ~21,000 muestras..."):
        model.fit(X_train, y_train)
    
    st.success("Modelo entrenado exitosamente.")
    
    st.session_state['rf_model'] = model
    st.session_state['y_test'] = y_test
    st.session_state['y_pred'] = model.predict(X_test)

# ==========================================
# FASE 5: EVALUACIÓN
# ==========================================
with tabs[4]:
    st.header("📈 Fase 5: Evaluación")
    
    if 'rf_model' not in st.session_state:
        st.warning("⚠️ Primero ejecuta la Fase 4: Modelado.")
        st.stop()

    y_test = st.session_state['y_test']
    y_pred = st.session_state['y_pred']
    
    acc = accuracy_score(y_test, y_pred)
    st.metric("Precisión Global (Accuracy)", f"{acc:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Real: No Incumple (0)', 'Real: Incumple (1)'], 
                             columns=['Pred: No Incumple (0)', 'Pred: Incumple (1)'])
        st.dataframe(cm_df)
        st.caption("Diagonal principal = aciertos. El modelo debe ser bueno prediciendo la fila 'Real: Incumple (1)'.")
    
    with col2:
        st.subheader("Reporte de Clasificación")
        report = classification_report(y_test, y_pred, target_names=['No Incumple (0)', 'Incumple (1)'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        st.caption("Métricas clave: Precisión (Precision) y Exhaustividad (Recall) para la clase 'Incumple (1)'.")

# ==========================================
# FASE 6: DESPLIEGUE
# ==========================================
with tabs[5]:
    st.header("🚀 Fase 6: Despliegue (Aplicación Interactiva)")
    st.markdown("---")

    if 'rf_model' not in st.session_state:
        st.warning("⚠️ El modelo debe estar entrenado (Fase 4) para la predicción en vivo.")
        st.stop()

    model = st.session_state['rf_model']
    features = st.session_state['model_features']
    
    st.subheader("Simulador de Puntuación de Riesgo de Crédito")
    
    # --- Formulario de Entrada ---
    with st.form("prediction_form"):
        # Inputs directos
        col1, col2, col3 = st.columns(3)
        limit_bal = col1.number_input("Límite de Crédito (LIMIT_BAL)", min_value=10000, max_value=1000000, value=100000)
        age = col2.slider("Edad (AGE)", 20, 70, 35)
        pay_0 = col3.selectbox("Estado de Pago Sep. (PAY_0)", range(-2, 9), index=2, help="0: Pago al día; 1: Retraso 1 mes; 2: Retraso 2 meses")
        
        # Inputs categóricos
        col4, col5, col6 = st.columns(3)
        bill_amt1 = col4.number_input("Monto Factura Sep. (BILL_AMT1)", 0, 500000, 20000)
        pay_amt1 = col5.number_input("Monto Pago Anterior Sep. (PAY_AMT1)", 0, 100000, 5000)
        sex = col6.selectbox("Género (SEX)", [1, 2], format_func=lambda x: "Hombre" if x == 1 else "Mujer")
        
        col7, col8 = st.columns(2)
        education = col7.selectbox("Educación (EDUCATION)", [1, 2, 3, 4], format_func=lambda x: {1: 'Posgrado', 2: 'Universidad', 3: 'Secundaria', 4: 'Otro'}.get(x, 'Otro'))
        marriage = col8.selectbox("Estado Civil (MARRIAGE)", [1, 2, 3], format_func=lambda x: {1: 'Casado', 2: 'Soltero', 3: 'Otro'}.get(x, 'Otro'))

        submitted = st.form_submit_button("Predecir Riesgo")

    if submitted:
        # 1. Crear el DataFrame de entrada con las mismas columnas transformadas
        input_data = pd.DataFrame(0, index=[0], columns=features)
        
        # 2. Rellenar las variables directas
        input_data['limit_bal'] = limit_bal
        input_data['age'] = age
        input_data['pay_0'] = pay_0
        input_data['bill_amt1'] = bill_amt1
        input_data['pay_amt1'] = pay_amt1
        
        # 3. Rellenar las variables One-Hot (El código replica la lógica de Fase 3)
        if sex == 2 and 'sex_2' in input_data.columns: input_data['sex_2'] = 1
        if education != 1 and f'education_{education}' in input_data.columns: input_data[f'education_{education}'] = 1
        if marriage != 1 and f'marriage_{marriage}' in input_data.columns: input_data[f'marriage_{marriage}'] = 1

        # 4. Predicción
        proba = model.predict_proba(input_data)[0][1]
        
        st.divider()
        if proba >= 0.2: 
            st.error(f"🚨 ALTO RIESGO DE INCUMPLIMIENTO: {proba:.2%}")
            st.markdown("**Acción Inmediata:** Revisión de crédito y posible contacto proactivo.")
        else:
            st.success(f"✅ RIESGO BAJO. Probabilidad de incumplimiento: {proba:.2%}")

            st.markdown("**Acción Inmediata:** Monitoreo estándar.")


