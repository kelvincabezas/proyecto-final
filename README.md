# ETL PRODUCE: Sistema de Análisis de Pesca Artesanal en Áncash, Perú 🐟🇵🇪

## Descripción del Proyecto

ETL PRODUCE es una aplicación web innovadora desarrollada para analizar y optimizar los datos de la pesca artesanal en la región de Áncash, Perú. Utilizando técnicas avanzadas de análisis de datos, machine learning y visualización, el proyecto busca proporcionar insights valiosos para pescadores, investigadores y tomadores de decisiones y escalar a nivel nacional.

## 🌟 Características Principales

### 1. Carga de Datos
- Interfaz intuitiva para cargar reportes de desembarque en formato Excel
- Procesamiento automático de datos de diferentes fuentes
- Validación y limpieza de datos

### 2. Dashboard Interactivo
- Visualización dinámica de datos pesqueros
- Gráficos interactivos de:
  - Volumen de captura por especie
  - Ganancias por embarcación
  - Distribución geográfica de capturas
  - Tendencias temporales

### 3. Modelo Predictivo
- Modelo de Random Forest para predecir:
  - Volumen de captura
  - Ganancias potenciales
- Análisis de importancia de características
- Explicaciones generadas por IA sobre las predicciones

### 4. Evaluación de Modelos
- Comparación de múltiples algoritmos de machine learning
- Métricas de rendimiento detalladas
- Visualización de resultados

## 🛠 Tecnologías Utilizadas

- **Lenguaje:** Python
- **Frameworks:** 
  - Streamlit
  - Scikit-learn
  - Pandas
- **Visualización:** 
  - Plotly
  - Folium
- **Machine Learning:** 
  - Random Forest
  - Gradient Boosting
  - Regresión Logística
- **API:** GPT-4

## Instalación y Configuración

### Requisitos Previos
- Python 3.8+
- pip
- Git

### Pasos de Instalación

1. **Clonar el repositorio**
    ```bash
    git clone https://github.com/jersonalvr/etl_produce.git
    cd etl_produce
    ```

2. **Crear un entorno virtual**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3. **Instalar dependencias**
    ```bash
    pip install -r requirements.txt
    ```

4. **Configurar variables de entorno**
    - Crea un archivo `.streamlit/secrets.toml` con las credenciales necesarias.

5. **Ejecutar la aplicación**
    ```bash
    streamlit run app.py
    ```

---

### 📊 Funcionalidades Detalladas

**Carga de Datos**
- Soporta múltiples archivos Excel
- Extracción inteligente de información
- Validación y preprocesamiento de datos

**Dashboard**
- Gráficos interactivos de:
  - Capturas por especie
  - Ganancias
  - Distribución geográfica
  - Tendencias temporales

**Modelo Predictivo**
- Predicción de volumen de captura
- Análisis de factores influyentes
- Explicaciones generadas por IA

**Evaluación de Modelos**
- Comparación de algoritmos
- Métricas de rendimiento
- Visualización de resultados

---

### 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, lee las [pautas de contribución](CONTRIBUTING.md) antes de enviar un pull request.

---

### 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

### 👥 Contacto

**Jerson Ruiz Alva**  
- **Email:** [jersonruizalva@gmail.com](mailto:jersonalvr@gmail.com)  
- **LinkedIn:** [Perfil de LinkedIn](https://www.linkedin.com/in/jersonalvr)

---

### 🌍 Impacto Social

Este proyecto busca:
- Mejorar la toma de decisiones en la pesca artesanal
- Proporcionar herramientas de análisis avanzado
- Contribuir al desarrollo sostenible de la región de Áncash

¡Ayudando a transformar la pesca artesanal con tecnología y datos! 🚢📊

---
