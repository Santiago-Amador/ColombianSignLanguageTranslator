Colombian Sign Language Translator (Python)

Descripción:
Este proyecto es un traductor de lenguaje de señas colombiano usando inteligencia artificial. Actualmente, la implementación en Python permite predecir gestos de 
lenguaje de señas a partir de imágenes, utilizando un modelo entrenado con CNN.

Estructura del proyecto

```yml
Python Version: 3.10
Java: 17
Markdown: 3.10
Numpy : 2.2.6  
Keras: 3.12.0 
Matplotlib: 3.10.7
Tensorflow: 2.20
```

## Instalación 
1. Clonar este repositorio:

 ```bash
   $ git clone https://github.com/Santiago-Amador/ColombianSignLanguageTranslator.git
   ```
2. Entrar en el directorio:
   ```bash
   > cd ColombianSignLanguageTranslator
   > cd python
   ```
3 Descargar dependencias python
  ```bash
   > pip install -r requirements.txt
   ```
## Ejecución

- Ingresar imagen a predecir en /python/data/testiamges
- Entrenar el modelo con:
- ```bash
   > python src/predictions/train.py
   ```
  EL modelo entrenado se guarda en: /train/models
- Por ultimo:
  ```bash
   > python src/predictions/predict.py
   ```
## Rutas
Configuración

El proyecto utiliza un archivo config.py con las siguientes configuraciones:

PATHS: rutas relativas a las carpetas de datos, modelos y test.

IMAGE: tamaño y canales de las imágenes para el modelo.

CLASSES: lista de clases extraídas automáticamente de los datos crudos.

TRAINING: parámetros de entrenamiento (batch size, epochs, learning rate, etc.).





