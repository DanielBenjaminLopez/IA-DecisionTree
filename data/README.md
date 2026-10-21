# Gestión de Datos del Proyecto

Esta carpeta contiene los datasets utilizados para las etapas de desarrollo, prueba y entrega final del trabajo práctico.

## 📁 Archivos en este directorio

1. **`provisional_kaggle.csv` (Dataset temporal de pruebas):**
   - **Objetivo:** Permitir la implementación algorítmica, validación de fórmulas de entropía/ganancia, y diseño de la interfaz antes de disponer del dataset definitivo.
   - **Registros:** 150 instancias (ligero y rápido para pruebas interactivas y depuración).
   - **Variables:**
     - `Age` (Numérica continua): Edad del paciente.
     - `Sex` (Categórica): Género (`M`, `F`).
     - `ChestPainType` (Categórica): Tipo de dolor en el pecho (`ATA`, `NAP`, `ASY`, `TA`).
     - `RestingBP` (Numérica continua): Presión arterial en reposo (mm Hg).
     - `Cholesterol` (Numérica continua): Colesterol sérico (mm/dl).
     - `FastingBS` (Binaria): Azúcar en sangre en ayunas > 120 mg/dl (`1`, `0`).
     - `RestingECG` (Categórica): Resultados electrocardiograma (`Normal`, `ST`, `LVH`).
     - `MaxHR` (Numérica continua): Frecuencia cardíaca máxima alcanzada.
     - `ExerciseAngina` (Categórica): Angina inducida por ejercicio (`Y`, `N`).
     - `Oldpeak` (Numérica continua): Depresión del ST inducida por ejercicio.
     - `ST_Slope` (Categórica): Pendiente del segmento ST pico (`Up`, `Flat`, `Down`).
     - `HeartDisease` (**Target / Variable objetivo**): Diagnóstico de cardiopatía (`1` = Presencia, `0` = Ausencia).

2. **`oficial_catedra.csv` (Dataset definitivo):**
   - Archivo que se incorporará en cuanto la cátedra provea el dataset oficial para las simulaciones y el informe definitivo.

## 🔄 Cómo alternar datasets
La interfaz en Google Colab incluye un selector interactivo (`ipywidgets.Dropdown`) que detecta automáticamente los archivos `.csv` presentes en esta carpeta. Para cambiar de dataset:
- Coloca el nuevo archivo en `/data/`.
- Selecciona el archivo deseado en la interfaz del notebook.
- Especifica la columna objetivo (Target) y ejecuta la clasificación.
