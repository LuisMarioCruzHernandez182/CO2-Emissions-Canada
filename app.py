from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import logging
import os
import warnings

# Suprimir warnings de sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Rutas de los archivos
MODEL_PATH = 'model_rf_co2.pkl'
SCALER_PATH = 'scaler.pkl'

# Características principales del modelo vehicular
top_5_features = ['Fuel Consumption Comb (mpg)', 'Fuel Consumption City (L/100 km)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'Engine Size(L)']

# Inicializar variables globales
model = None
scaler = None

# Función para cargar modelo y escalador
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Modelo o scaler no encontrados.")
    
    model_loaded = joblib.load(MODEL_PATH)
    scaler_loaded = joblib.load(SCALER_PATH)
    
    return model_loaded, scaler_loaded

# Intentar cargar modelo y escalador al iniciar
try:
    model, scaler = load_model_and_scaler()
    logging.info("✅ Modelo y scaler cargados correctamente.")
    
    # Verificar qué características espera el scaler
    if hasattr(scaler, 'feature_names_in_'):
        logging.info(f"🏷 Características esperadas por el scaler: {list(scaler.feature_names_in_)}")
    if hasattr(scaler, 'n_features_in_'):
        logging.info(f"🔢 Número de características esperadas: {scaler.n_features_in_}")
    if hasattr(scaler, 'mean_'):
        logging.info(f"📊 Media del scaler: {scaler.mean_}")
    if hasattr(scaler, 'scale_'):
        logging.info(f"📏 Escala del scaler: {scaler.scale_}")
        
except Exception as e:
    logging.error(f"❌ Error al cargar el modelo o el escalador: {str(e)}")

@app.route('/')
def home():
    logging.info("🏠 Acceso a página principal")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Logging de la petición recibida
        logging.info("🔍 Petición POST recibida en /predict")
        logging.info(f"📝 Datos del formulario: {dict(request.form)}")
        
        # Obtener datos del formulario vehicular
        fuel_comb_mpg = float(request.form['fuel_comb_mpg'])
        fuel_city_l100km = float(request.form['fuel_city_l100km'])
        cylinders = int(request.form['cylinders'])
        fuel_comb_l100km = float(request.form['fuel_comb_l100km'])
        engine_size = float(request.form['engine_size'])

        logging.info(f"📊 Datos vehiculares procesados:")
        logging.info(f"   - Consumo Combinado (MPG): {fuel_comb_mpg}")
        logging.info(f"   - Consumo Ciudad (L/100km): {fuel_city_l100km}")
        logging.info(f"   - Cilindros: {cylinders}")
        logging.info(f"   - Consumo Combinado (L/100km): {fuel_comb_l100km}")
        logging.info(f"   - Tamaño del Motor (L): {engine_size}")

        # Validar que los valores sean positivos
        if any(val <= 0 for val in [fuel_comb_mpg, fuel_city_l100km, fuel_comb_l100km, engine_size]) or cylinders <= 0:
            logging.error("❌ Valores negativos o cero detectados")
            return render_template('index.html', prediction="Error: Todos los valores deben ser positivos")

        # Crear DataFrame con las columnas EXACTAS que espera el scaler
        # Usando las top_5_features definidas
        input_data = pd.DataFrame([[
            fuel_comb_mpg, fuel_city_l100km, cylinders, fuel_comb_l100km, engine_size
        ]], columns=top_5_features)

        logging.info(f"🔢 DataFrame creado: {input_data.to_dict('records')[0]}")
        logging.info(f"🏷 Columnas del DataFrame: {list(input_data.columns)}")
        logging.info(f"📐 Forma del DataFrame: {input_data.shape}")

        # Verificar que las columnas coincidan
        if hasattr(scaler, 'feature_names_in_'):
            expected_features = list(scaler.feature_names_in_)
            actual_features = list(input_data.columns)
            logging.info(f"🔍 Características esperadas: {expected_features}")
            logging.info(f"🔍 Características actuales: {actual_features}")
            
            if expected_features != actual_features:
                logging.warning("⚠ Las características no coinciden, reordenando...")
                # Intentar reordenar según las características esperadas
                try:
                    input_data = input_data[expected_features]
                except KeyError as ke:
                    logging.error(f"❌ Error: Característica faltante {ke}")
                    return render_template('index.html', prediction=f"Error: Característica del modelo no coincide {ke}")

        # Escalar datos
        logging.info(f"📊 Datos antes del escalado: {input_data.values[0]}")
        input_scaled = scaler.transform(input_data)
        logging.info(f"⚖ Datos después del escalado: {input_scaled[0]}")
        
        # Verificar que el escalado funcionó
        if np.array_equal(input_data.values[0], input_scaled[0]):
            logging.error("❌ ERROR: Los datos no se escalaron correctamente!")
            return render_template('index.html', prediction="Error: Problema con el escalado de datos")

        # Predecir
        prediction = model.predict(input_scaled)[0]
        logging.info(f"🎯 Predicción vehicular realizada: {prediction}")

        # Formatear la predicción
        prediction_formatted = round(prediction, 2)
        logging.info(f"✅ Predicción formateada: {prediction_formatted}")

        return render_template('index.html', prediction=prediction_formatted)

    except KeyError as ke:
        logging.error(f"❌ Campo faltante en el formulario: {str(ke)}")
        logging.error(f"📋 Campos disponibles: {list(request.form.keys())}")
        return render_template('index.html', prediction=f"Error: Campo faltante {str(ke)}")
    except ValueError as ve:
        logging.error(f"❌ Error de valor: {str(ve)}")
        logging.error(f"📋 Valores recibidos: {dict(request.form)}")
        return render_template('index.html', prediction="Error: Valores inválidos. Verifica que todos los campos sean números.")
    except Exception as e:
        logging.error(f"❌ Error en la predicción vehicular: {str(e)}")
        logging.error(f"📋 Tipo de error: {type(e).__name__}")
        return render_template('index.html', prediction=f"Error: {str(e)}")

# Bloquear peticiones no deseadas SILENCIOSAMENTE
@app.before_request
def block_unwanted():
    if request.path.startswith('/logs/'):
        return '', 404

if __name__ == '__main__':
    logging.info("🚀 Iniciando servidor Flask para análisis vehicular...")
    logging.info(f"🔧 Características del modelo: {top_5_features}")
    app.run(debug=True)