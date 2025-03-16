"""
Sistema de Clasificación de Naranjas con Detección de Defectos
===========================================================

Este programa implementa un sistema híbrido que combina detección de objetos con MobileNetV2
y análisis de defectos mediante modelos de IA para clasificar
naranjas en tiempo real.

Características principales:
- Detección precisa de naranjas usando MobileNetV2
- Análisis de defectos superficiales mediante técnicas avanzadas de visión
- Máscara de segmentación para analizar solo la superficie de los citricos detectados
- Interfaz gráfica con controles ajustables de color
- Generación de informes en CSV
- Captura y almacenamiento de imágenes clasificadas

Autor: [Joan Ruiz Verdú]
Fecha: [08/03/2025]
Versión: 3.0
"""

# cv2 (OpenCV): Biblioteca de visión por computadora para procesamiento de imágenes y video
# Desarrollada originalmente por Intel, ahora mantenida como proyecto de código abierto
import cv2

# numpy: Biblioteca para computación científica con soporte para arrays multidimensionales
# Mantenida por la comunidad de código abierto bajo NumFOCUS
import numpy as np

# os: Módulo de la biblioteca estándar de Python para interactuar con el sistema operativo
# Desarrollado y mantenido por Python Software Foundation
import os

# csv: Módulo de la biblioteca estándar de Python para leer y escribir archivos CSV
# Desarrollado y mantenido por Python Software Foundation
import csv

# datetime: Módulo de la biblioteca estándar de Python para manipular fechas y horas
# Desarrollado y mantenido por Python Software Foundation
from datetime import datetime

# threading: Módulo de la biblioteca estándar de Python para programación con hilos
# Desarrollado y mantenido por Python Software Foundation
import threading

# tkinter: Biblioteca estándar de Python para crear interfaces gráficas de usuario (GUI)
# Desarrollado por Python Software Foundation, basado en Tk de Tcl Core Team
import tkinter as tk

# ttk: Extensión de tkinter que proporciona widgets con tema mejorado
# messagebox: Módulo de tkinter para mostrar cuadros de diálogo
# Ambos desarrollados por Python Software Foundation, basados en Tk
from tkinter import ttk, messagebox, filedialog

# PIL (Pillow): Biblioteca para abrir, manipular y guardar diferentes formatos de imagen
# Originalmente Python Imaging Library, ahora mantenida como Pillow por la comunidad
# ImageTk: Módulo de PIL para integración con tkinter
from PIL import Image, ImageTk

#tensorflow: Biblioteca de código abierto para aprendizaje automático y redes neuronales
#Desarrollada principalmente por Google, proporciona un ecosistema completo para construir, entrenar y desplegar modelos de IA
import tensorflow as tf

# =============================================
# CONFIGURACIÓN DE TENSORFLOW
# =============================================
print("Versión de TensorFlow:", tf.__version__)

class DetectorCitricos:
    """
    Detector de cítricos basado en modelo MobileNetV2 preentrenado y análisis de color.

    Combina técnicas de deep learning y procesamiento de imagen tradicional para detectar
    y clasificar cítricos (naranjas y limones) en imágenes.

    Attributes:
        modelo_base: Modelo MobileNetV2 preentrenado con pesos de ImageNet
        clases_citricos: Diccionario con índices de clases de cítricos en ImageNet
        rangos_color_citricos: Rangos HSV para la detección de diferentes cítricos
        citricos_habilitados: Lista de tipos de cítricos habilitados para detección
    """
    def __init__(self):
        """
        Inicializa el detector de cítricos cargando el modelo MobileNetV2 y configurando
        los rangos de color para la detección.
        """
        # Cargar el modelo MobileNetV2 preentrenado
        self.modelo_base = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=True
        )

        # Clases de ImageNet para diferentes cítricos y frutas
        self.clases_citricos = {
            'naranja': [950, 951], # orange
            'limon': [952, 953],   # lemon (sin acento)
        }

        # Rangos de color HSV para diferentes cítricos - RANGOS AMPLIADOS
        self.rangos_color_citricos = {
            'naranja': (np.array([0, 100, 150]), np.array([30, 255, 255])), # Rango ampliado para naranjas
            'limon': (np.array([15, 70, 150]), np.array([40, 255, 255])),  # Rango ampliado para limones (sin acento)
        }

        # Habilitar ambos tipos pero tratarlos como un solo tipo de pieza
        self.citricos_habilitados = ['naranja', 'limon']

        print("DetectorCitricos cargado correctamente")

    def detectar(self, imagen, rangos_personalizados=None):
        """
        Detecta piezas de cítricos priorizando la clasificación mediante IA.

        Utiliza un enfoque en dos fases:
        1. Clasificación mediante MobileNetV2 para identificar naranjas
        2. Segmentación y análisis de contornos para delimitar las naranjas detectadas

        Esta implementación mejorada es más robusta para detectar naranjas podridas
        o con coloración anormal, ya que prioriza la forma y el reconocimiento por IA
        sobre el color.

        Args:
            imagen: Imagen en formato BGR (OpenCV) donde buscar cítricos
            rangos_personalizados: Diccionario opcional con rangos HSV personalizados para
                          la detección de color, formato {tipo: (inferior, superior)}

        Returns:
            tuple: (detecciones, mascara, contornos)
                  - detecciones: Lista de diccionarios con información de cada detección
                  - mascara: Máscara binaria de los cítricos detectados
                  - contornos: Lista de contornos encontrados
        """
        # PASO 1: Clasificación con MobileNetV2 para determinar si hay naranjas
        imagen_redimensionada = cv2.resize(imagen, (224, 224))
        imagen_rgb = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2RGB)
        array_imagen = np.expand_dims(imagen_rgb, axis=0)
        imagen_preprocesada = tf.keras.applications.mobilenet_v2.preprocess_input(array_imagen)

        predicciones = self.modelo_base.predict(imagen_preprocesada)
        clases_principales = np.argsort(predicciones[0])[-10:]  # Top 10 clases

        # Verificar si MobileNetV2 detecta una naranja (con confianza alta)
        naranja_detectada = False
        confianza_naranja = 0
        tipo_detectado = None

        for tipo_citrico, ids_clase in self.clases_citricos.items():
            for cls in clases_principales:
                if cls in ids_clase and tipo_citrico in self.citricos_habilitados:
                    conf = predicciones[0][cls]
                    if conf > 0.1:  # Umbral de confianza bajo para ser más flexible
                        naranja_detectada = True
                        tipo_detectado = tipo_citrico
                        confianza_naranja = conf
                        break
            if naranja_detectada:
                break

        # PASO 2: Encontrar los contornos de la naranja en la imagen
        # Usando una combinación de técnicas para ser más robusto

        # 2.1: Segmentación HSV tradicional
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        mascara_combinada = np.zeros((imagen.shape[0], imagen.shape[1]), dtype=np.uint8)

        # Usar rangos personalizados si se proporcionan
        rangos_color = rangos_personalizados if rangos_personalizados else self.rangos_color_citricos

        # Añadir rangos adicionales para naranjas podridas
        rangos_naranja = [
            # Naranja normal (usar rangos existentes)
            rangos_color.get('naranja', (np.array([0, 100, 150]), np.array([30, 255, 255]))),
            # Naranja podrida (más oscura/marrón)
            (np.array([0, 30, 30]), np.array([30, 180, 180])),
            # Incluir tonos rojizos para naranjas muy maduras
            (np.array([160, 100, 100]), np.array([180, 255, 255]))
        ]

        for inferior, superior in rangos_naranja:
            mascara = cv2.inRange(hsv, inferior, superior)
            mascara_combinada = cv2.bitwise_or(mascara_combinada, mascara)

        # 2.2: Mejorar con operaciones morfológicas
        kernel = np.ones((5, 5), np.uint8)
        mascara = cv2.morphologyEx(mascara_combinada, cv2.MORPH_OPEN, kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

        # 2.3: Si todavía no hay buena segmentación, usar técnicas adicionales
        if cv2.countNonZero(mascara) < 1000 and naranja_detectada:
            # Intentar segmentación por umbral adaptativo
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            umbral = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
            mascara = cv2.bitwise_or(mascara, umbral)

            # Operaciones morfológicas para limpiar
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

        # PASO 3: Encontrar contornos de las naranjas
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar los contornos por tamaño y forma
        detecciones = []
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area < 1000:  # Filtrar objetos pequeños
                continue

            # Calcular características del contorno
            perimetro = cv2.arcLength(contorno, True)
            circularidad = 4 * np.pi * area / (perimetro * perimetro) if perimetro > 0 else 0

            # Las naranjas son bastante circulares, pero permitimos cierta flexibilidad
            # para naranjas deformes o podridas
            if circularidad < 0.5 and area < 5000 and not naranja_detectada:
                continue

            x, y, w, h = cv2.boundingRect(contorno)

            # Verificar relación de aspecto (las naranjas son bastante circulares)
            relacion_aspecto = float(w) / h if h > 0 else 0
            if relacion_aspecto < 0.6 or relacion_aspecto > 1.5:
                continue

            # Si MobileNetV2 detectó naranja, usamos ese tipo
            # Si no, intentamos determinar por color
            tipo_interno = tipo_detectado
            if not tipo_interno:
                # Analizar el color predominante en la ROI
                roi = imagen[y:y+h, x:x+w]
                if roi.size == 0:
                    continue

                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                max_votos = 0
                for t_citrico in self.citricos_habilitados:
                    inferior, superior = rangos_color.get(t_citrico, self.rangos_color_citricos[t_citrico])
                    mascara_tipo = cv2.inRange(roi_hsv, inferior, superior)
                    votos = cv2.countNonZero(mascara_tipo)
                    if votos > max_votos:
                        max_votos = votos
                        tipo_interno = t_citrico

            deteccion = {
                'bbox': (x, y, x+w, y+h),
                'confianza': confianza_naranja if naranja_detectada else (circularidad * 0.5 + 0.5),
                'area': area,
                'circularidad': circularidad,
                'contorno': contorno,
                'tipo': "pieza",  # Solo mostramos "pieza"
                'tipo_interno': tipo_interno  # Guardamos el tipo interno para análisis
            }
            detecciones.append(deteccion)

        return detecciones, mascara, contornos

class AnalizadorDefectos:
    """
    Analizador de defectos en cítricos basado en análisis de color HSV.

    Detecta diferentes tipos de defectos en la superficie de los cítricos
    mediante segmentación de color y análisis de contornos.

    Attributes:
        tipos_defectos: Diccionario con configuración para diferentes tipos de defectos
        tamanio_minimo_defecto: Tamaño mínimo en píxeles para considerar un defecto válido
        sensibilidad: Factor de sensibilidad para la detección de defectos
    """
    def __init__(self):
        """
        Inicializa el analizador de defectos con rangos de color y parámetros predeterminados.
        """
        # Configuración inicial para detección de defectos
        self.tipos_defectos = {
            'negro': {'nombre': 'Manchas negras', 'inferior': np.array([0, 0, 0]), 'superior': np.array([180, 255, 60])},
            'verde': {'nombre': 'Pudricion verde', 'inferior': np.array([35, 50, 50]), 'superior': np.array([90, 255, 255])},
            'marron': {'nombre': 'Pudricion marron', 'inferior': np.array([10, 50, 20]), 'superior': np.array([20, 255, 120])}
        }
        # Parámetros ajustables
        self.tamanio_minimo_defecto = 30  # Tamaño mínimo de defecto en píxeles
        self.sensibilidad = 1.0     # Factor de sensibilidad para la detección
        print("Analizador de defectos configurado")

    def analizar(self, imagen, mascara_citrico, rangos_defectos=None, tamanio_min=None, sensibilidad=None):
        """
        Analiza defectos en la imagen de un cítrico utilizando la máscara proporcionada.

        Versión mejorada que incorpora análisis de textura y variación de color para
        detectar anomalías, especialmente en naranjas podridas donde toda la superficie
        puede presentar coloración anormal.

        Args:
            imagen: Imagen en formato BGR (OpenCV) del cítrico a analizar
            mascara_citrico: Máscara binaria que delimita la superficie del cítrico
            rangos_defectos: Diccionario opcional con rangos HSV personalizados para la detección
                          de defectos, formato {tipo_defecto: (inferior, superior)}
            tamanio_min: Tamaño mínimo opcional en píxeles para considerar un defecto válido
            sensibilidad: Factor de sensibilidad opcional para ajustar la detección

        Returns:
            tuple: (defectos, area_total_defectos, mascara_combinada)
                  - defectos: Diccionario con tipos de defectos y sus áreas
                  - area_total_defectos: Área total de defectos en píxeles
                  - mascara_combinada: Máscara combinada de todos los defectos
        """
        # Convertir a HSV para análisis de color
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

        # Usar parámetros personalizados si se proporcionan
        if rangos_defectos:
            tipos_defectos_temp = self.tipos_defectos.copy()
            for tipo_defecto, rangos in rangos_defectos.items():
                if tipo_defecto in tipos_defectos_temp:
                    tipos_defectos_temp[tipo_defecto]['inferior'] = rangos[0]
                    tipos_defectos_temp[tipo_defecto]['superior'] = rangos[1]
        else:
            tipos_defectos_temp = self.tipos_defectos.copy()

        # Añadimos detección de "podredumbre general" - áreas con gran desviación de color
        tipos_defectos_temp['podredumbre_general'] = {
            'nombre': 'Podredumbre general',
            'inferior': np.array([0, 0, 0]),
            'superior': np.array([180, 100, 100])  # Colores apagados/oscuros
        }

        # Usar tamaño mínimo personalizado si se proporciona
        tamanio_minimo_defecto = tamanio_min if tamanio_min is not None else self.tamanio_minimo_defecto
        factor_sensibilidad = sensibilidad if sensibilidad is not None else self.sensibilidad

        defectos = {}
        mascaras_defectos = {}
        area_total_defectos = 0

        # Crear una máscara combinada para visualización
        mascara_combinada = np.zeros_like(mascara_citrico)

        # Mejorar contraste en la imagen para mejor detección
        lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        imagen_mejorada = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        hsv_mejorado = cv2.cvtColor(imagen_mejorada, cv2.COLOR_BGR2HSV)

        # Análisis de textura para detectar anomalías
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gris, (5, 5), 0)
        mascara_textura = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
        mascara_textura = cv2.bitwise_and(mascara_textura, mascara_citrico)

        # Para cada tipo de defecto, generar su máscara específica
        for tipo_defecto, config in tipos_defectos_temp.items():
            if tipo_defecto == 'podredumbre_general':
                # Para podredumbre general usamos más la textura
                mascara_defecto = mascara_textura.copy()
            elif tipo_defecto == 'negro':
                # Para manchas negras, usar la imagen mejorada y enfocarse en valores bajos de V
                inferior_ajustado = config['inferior'].copy()
                superior_ajustado = config['superior'].copy()

                # Aplicar factor de sensibilidad (aumentar para más sensibilidad)
                superior_ajustado[2] = min(255, superior_ajustado[2] * factor_sensibilidad)

                mascara_defecto = cv2.inRange(hsv_mejorado, inferior_ajustado, superior_ajustado)
            else:
                # Para otros defectos, usar HSV normal
                mascara_defecto = cv2.inRange(hsv, config['inferior'], config['superior'])

            # Combinar con la máscara de la pieza para limitarlo solo a la fruta
            mascara_defecto = cv2.bitwise_and(mascara_defecto, mascara_citrico)

            # Mejorar la máscara
            tamanio_kernel = 3
            kernel = np.ones((tamanio_kernel, tamanio_kernel), np.uint8)
            mascara_defecto = cv2.morphologyEx(mascara_defecto, cv2.MORPH_OPEN, kernel)
            mascara_defecto = cv2.morphologyEx(mascara_defecto, cv2.MORPH_CLOSE, kernel)

            # Encontrar contornos de este defecto
            contornos_defecto, _ = cv2.findContours(mascara_defecto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            area_defecto = 0
            contornos_validos = []

            # Sumar área de todos los contornos de este defecto
            for contorno in contornos_defecto:
                area = cv2.contourArea(contorno)
                if area > tamanio_minimo_defecto:  # Usar el tamaño mínimo ajustable
                    area_defecto += area
                    contornos_validos.append(contorno)

            # Dibujar los contornos válidos
            if tipo_defecto == 'negro':
                color = (0, 0, 255)  # Rojo para manchas negras
            elif tipo_defecto == 'verde':
                color = (0, 255, 0)  # Verde para pudrición verde
            elif tipo_defecto == 'podredumbre_general':
                color = (255, 0, 255)  # Magenta para podredumbre general
            else:
                color = (255, 0, 0)  # Azul para otros defectos

            cv2.drawContours(imagen, contornos_validos, -1, color, 2)

            if area_defecto > 0:
                defectos[config['nombre']] = area_defecto
                area_total_defectos += area_defecto

            mascaras_defectos[tipo_defecto] = mascara_defecto
            mascara_combinada = cv2.bitwise_or(mascara_combinada, mascara_defecto)

        return defectos, area_total_defectos, mascara_combinada

# Control de color para ajustar los rangos HSV
class ControlColor(ttk.LabelFrame):
    """
    Clase para crear un conjunto de controles deslizantes para ajustar rangos HSV.

    Proporciona una interfaz gráfica que permite ajustar los valores mínimos y máximos
    de H (Hue), S (Saturation) y V (Value) para la detección de colores específicos.

    Attributes:
        h_min, s_min, v_min: Controles deslizantes para los valores mínimos de HSV
        h_max, s_max, v_max: Controles deslizantes para los valores máximos de HSV
    """
    def __init__(self, parent, titulo, valor_inicial_min, valor_inicial_max, callback):
        """
        Inicializa un conjunto de controles para ajustar rangos HSV.

        Args:
            parent: Widget padre en la jerarquía de Tkinter
            titulo: Título para el marco de controles
            valor_inicial_min: Lista o array con los valores iniciales mínimos [H, S, V]
            valor_inicial_max: Lista o array con los valores iniciales máximos [H, S, V]
            callback: Función a invocar cuando cambia algún valor
        """
        super().__init__(parent, text=titulo)

        # Frame para valores mínimos
        frame_min = ttk.LabelFrame(self, text="Mínimos")
        frame_min.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Label(frame_min, text="H:").grid(row=0, column=0)
        self.h_min = ttk.Scale(frame_min, from_=0, to=180, orient=tk.HORIZONTAL)
        self.h_min.set(valor_inicial_min[0])
        self.h_min.grid(row=0, column=1)

        ttk.Label(frame_min, text="S:").grid(row=1, column=0)
        self.s_min = ttk.Scale(frame_min, from_=0, to=255, orient=tk.HORIZONTAL)
        self.s_min.set(valor_inicial_min[1])
        self.s_min.grid(row=1, column=1)

        ttk.Label(frame_min, text="V:").grid(row=2, column=0)
        self.v_min = ttk.Scale(frame_min, from_=0, to=255, orient=tk.HORIZONTAL)
        self.v_min.set(valor_inicial_min[2])
        self.v_min.grid(row=2, column=1)

        # Frame para valores máximos
        frame_max = ttk.LabelFrame(self, text="Máximos")
        frame_max.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Label(frame_max, text="H:").grid(row=0, column=0)
        self.h_max = ttk.Scale(frame_max, from_=0, to=180, orient=tk.HORIZONTAL)
        self.h_max.set(valor_inicial_max[0])
        self.h_max.grid(row=0, column=1)

        ttk.Label(frame_max, text="S:").grid(row=1, column=0)
        self.s_max = ttk.Scale(frame_max, from_=0, to=255, orient=tk.HORIZONTAL)
        self.s_max.set(valor_inicial_max[1])
        self.s_max.grid(row=1, column=1)

        ttk.Label(frame_max, text="V:").grid(row=2, column=0)
        self.v_max = ttk.Scale(frame_max, from_=0, to=255, orient=tk.HORIZONTAL)
        self.v_max.set(valor_inicial_max[2])
        self.v_max.grid(row=2, column=1)

        # Vincular callback a todos los controles
        for control in [self.h_min, self.s_min, self.v_min,
                       self.h_max, self.s_max, self.v_max]:
            control.configure(command=lambda _: callback())

    def obtener_valores(self):
        """
        Devuelve los valores actuales como dos arrays numpy.

        Returns:
            tuple: Dos arrays numpy conteniendo (valores_minimos, valores_maximos)
                  Cada array tiene la forma [H, S, V]
        """
        return (np.array([self.h_min.get(), self.s_min.get(), self.v_min.get()]),
                np.array([self.h_max.get(), self.s_max.get(), self.v_max.get()]))

def crear_directorios():
    """
    Crea la estructura de directorios necesaria para el almacenamiento de datos.

    Crea el directorio para almacenar capturas y si no existe, inicializa el archivo
    CSV de informe con las columnas correspondientes.

    Returns:
        str: Ruta al directorio de capturas
    """
    ruta_guardado = "./capturas/"
    os.makedirs(ruta_guardado, exist_ok=True)

    if not os.path.isfile('informe.csv'):
        with open('informe.csv', 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['Fecha','Imagen','Tipo','Calidad','Defectos','Porcentaje Defectos'])

    return ruta_guardado

def clasificar_fruta(defectos, area_fruta):
    """
    Clasifica la fruta según el porcentaje de defectos detectados.

    Args:
        defectos: Diccionario con tipos de defectos y sus áreas en píxeles
        area_fruta: Área total de la fruta en píxeles

    Returns:
        tuple: (calidad, color, texto_defectos, porcentaje)
               - calidad: Texto describiendo la calidad ("Excelente", "Aceptable", "Rechazada")
               - color: Tupla RGB para representar visualmente la calidad
               - texto_defectos: Descripción textual de los defectos encontrados
               - porcentaje: Porcentaje de la superficie con defectos
    """
    total_defectos = sum(defectos.values()) if defectos else 0
    porcentaje = (total_defectos / area_fruta) * 100 if area_fruta > 0 else 0
    texto_defectos = ", ".join(defectos.keys()) if defectos else "Ninguno"

    if porcentaje < 2:
        return "Excelente", (0, 255, 0), texto_defectos, porcentaje
    elif porcentaje < 5:
        return "Aceptable", (0, 255, 255), texto_defectos, porcentaje
    else:
        return "Rechazada", (0, 0, 255), texto_defectos, porcentaje

class Aplicacion:
    """
    Clase principal que implementa la aplicación de clasificación de cítricos.

    Esta clase gestiona la interfaz gráfica, la captura de video, el procesamiento
    de imágenes, la detección de cítricos y la clasificación según defectos.

    Attributes:
        raiz: Ventana principal de Tkinter
        camara: Objeto de captura de video
        analizando: Indica si el análisis está en curso
        ultimo_fotograma: Último fotograma procesado
        ruta_guardado: Ruta donde se guardan las capturas
        nivel_zoom: Nivel actual de zoom
        centro_zoom: Centro del zoom en coordenadas relativas
        modo_depuracion: Controla si se muestran las máscaras de depuración
        modo_imagen_estatica: Indica si se está procesando una imagen estática
        detector_citricos: Instancia del detector de cítricos
        analizador_defectos: Instancia del analizador de defectos
    """
    def __init__(self, raiz):
        """
        Inicializa la aplicación de clasificación de cítricos.

        Args:
            raiz: Objeto Tk de Tkinter (ventana principal)
        """
        self.raiz = raiz
        self.camara = None
        self.analizando = False
        self.ultimo_fotograma = None
        self.ruta_guardado = crear_directorios()
        self.nivel_zoom = 1.0
        self.centro_zoom = None
        self.modo_depuracion = tk.BooleanVar(value=True)
        self.modo_imagen_estatica = False
        self.imagen_estatica = None

        # Inicializar detectores de IA
        print("Cargando modelos de IA...")
        self.detector_citricos = DetectorCitricos()
        self.analizador_defectos = AnalizadorDefectos()
        print("Modelos cargados correctamente")

        # Configuración de rangos de color y parámetros adicionales
        self.rangos_citricos = {
            'naranja': (np.array([0, 100, 150]), np.array([30, 255, 255])), # Rango ampliado naranja
            'limon': (np.array([15, 70, 150]), np.array([40, 255, 255]))    # Rango ampliado limon (sin acento)
        }

        self.rangos_defectos = {
            'negro': (np.array([0, 0, 0]), np.array([180, 255, 60])),
            'verde': (np.array([35, 50, 50]), np.array([90, 255, 255]))
        }

        # Parámetros ajustables adicionales
        self.tamanio_minimo_defecto = tk.IntVar(value=30)
        self.sensibilidad = tk.DoubleVar(value=1.0)

        self.cargar_logo()
        self.configurar_gui()

    def cargar_logo(self):
        """
        Intenta cargar el logo UCMA desde el archivo y prepararlo para la interfaz.

        Carga el logo en dos versiones:
        - Una versión pequeña como icono de la ventana
        - Una versión mayor para mostrar en la interfaz

        Si ocurre algún error durante la carga, establece ambos como None.
        """
        try:
            # Asumimos que el logo está en el mismo directorio que el script
            ruta_logo = "Logo.png"
            self.logo_original = Image.open(ruta_logo)

            # Crear icono para la ventana
            icono_temp = self.logo_original.copy()
            icono_temp.thumbnail((64, 64), Image.Resampling.LANCZOS)
            self.icono = ImageTk.PhotoImage(icono_temp)

            # Crear versión para mostrar en la interfaz
            display_temp = self.logo_original.copy()
            display_temp.thumbnail((150, 150), Image.Resampling.LANCZOS)
            self.logo_mostrar = ImageTk.PhotoImage(display_temp)
        except Exception as e:
            print(f"Error al cargar el logo: {e}")
            self.icono = None
            self.logo_mostrar = None

    def configurar_gui(self):
        """
        Configura todos los elementos de la interfaz gráfica de usuario.

        Crea y organiza todos los widgets, paneles y controles necesarios:
        - Logo y título
        - Botones de control
        - Panel de ajuste de zoom
        - Visualizador de valores HSV
        - Controles de ajuste de detección de defectos
        - Panel de visualización de video
        - Panel de resultados de texto
        """
        self.raiz.title("Clasificador de Piezas con IA - UCMA")
        self.raiz.geometry("1200x850")

        if self.icono:
            self.raiz.iconphoto(True, self.icono)

        # Panel de logo
        panel_logo = ttk.Frame(self.raiz)
        panel_logo.pack(fill=tk.X, padx=10, pady=5)

        if self.logo_mostrar:
            lbl_logo = ttk.Label(panel_logo, image=self.logo_mostrar)
            lbl_logo.pack(side=tk.LEFT, padx=5, pady=5)

        # Título de la aplicación
        lbl_titulo = ttk.Label(panel_logo,
                              text="Sistema de Clasificación de Citricos",
                              font=("Helvetica", 16, "bold"))
        lbl_titulo.pack(side=tk.LEFT, padx=20, pady=5)

        # Panel principal de control
        panel_control = ttk.Frame(self.raiz)
        panel_control.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Panel de botones básicos
        panel_botones = ttk.LabelFrame(panel_control, text="Controles Básicos")
        panel_botones.pack(side=tk.LEFT, padx=5, pady=5)

        self.boton_iniciar = ttk.Button(panel_botones, text="Iniciar Cámara", command=self.iniciar_analisis)
        self.boton_iniciar.pack(side=tk.LEFT, padx=5)

        self.boton_detener = ttk.Button(panel_botones, text="Detener", state=tk.DISABLED,
                                      command=self.detener_analisis)
        self.boton_detener.pack(side=tk.LEFT, padx=5)

        self.boton_capturar = ttk.Button(panel_botones, text="Capturar", state=tk.DISABLED,
                                     command=self.capturar_imagen)
        self.boton_capturar.pack(side=tk.LEFT, padx=5)

        # Botón para cargar imagen estática
        self.boton_cargar = ttk.Button(panel_botones, text="Cargar Imagen",
                                       command=self.cargar_imagen)
        self.boton_cargar.pack(side=tk.LEFT, padx=5)

        # Panel de zoom
        panel_zoom = ttk.LabelFrame(panel_control, text="Control de Zoom")
        panel_zoom.pack(side=tk.LEFT, padx=5, pady=5)

        self.boton_aumentar_zoom = ttk.Button(panel_zoom, text="Zoom +", command=self.aumentar_zoom)
        self.boton_aumentar_zoom.pack(side=tk.LEFT, padx=5)

        self.boton_disminuir_zoom = ttk.Button(panel_zoom, text="Zoom -", command=self.disminuir_zoom)
        self.boton_disminuir_zoom.pack(side=tk.LEFT, padx=5)

        self.boton_reiniciar_zoom = ttk.Button(panel_zoom, text="Reiniciar Zoom", command=self.reiniciar_zoom)
        self.boton_reiniciar_zoom.pack(side=tk.LEFT, padx=5)

        self.etiqueta_zoom = ttk.Label(panel_zoom, text="Zoom: 1.0x")
        self.etiqueta_zoom.pack(side=tk.LEFT, padx=5)

        # Panel de opciones
        panel_opciones = ttk.LabelFrame(panel_control, text="Opciones")
        panel_opciones.pack(side=tk.RIGHT, padx=5, pady=5)

        # Checkbox para modo de depuración
        self.check_depuracion = ttk.Checkbutton(panel_opciones, text="Mostrar máscaras",
                                           variable=self.modo_depuracion)
        self.check_depuracion.pack(side=tk.LEFT, padx=5)

        # Panel de controles de color
        self.frame_colores = ttk.LabelFrame(self.raiz, text="Ajustes de Detección")
        self.frame_colores.pack(fill=tk.X, padx=10, pady=5)

        # Control para la detección de piezas (naranja/limón combinado)
        self.control_naranja = ControlColor(
            self.frame_colores, "Detección de Piezas",
            self.rangos_citricos['naranja'][0], self.rangos_citricos['naranja'][1],
            self.actualizar_rangos_color
        )
        self.control_naranja.pack(side=tk.LEFT, padx=5, pady=5)

        # Control para la detección de defectos negros
        self.control_negro = ControlColor(
            self.frame_colores, "Manchas Negras",
            self.rangos_defectos['negro'][0], self.rangos_defectos['negro'][1],
            self.actualizar_rangos_color
        )
        self.control_negro.pack(side=tk.LEFT, padx=5, pady=5)

        # Control para la detección de defectos verdes
        self.control_verde = ControlColor(
            self.frame_colores, "Pudricion Verde",
            self.rangos_defectos['verde'][0], self.rangos_defectos['verde'][1],
            self.actualizar_rangos_color
        )
        self.control_verde.pack(side=tk.LEFT, padx=5, pady=5)

        # Añadir controles adicionales para el tamaño de defecto y sensibilidad
        panel_params = ttk.LabelFrame(self.frame_colores, text="Parámetros de Defectos")
        panel_params.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Label(panel_params, text="Tamaño mín. defecto:").grid(row=0, column=0, padx=5, pady=2)
        deslizador_tamanio = ttk.Scale(panel_params, from_=5, to=100, variable=self.tamanio_minimo_defecto,
                                orient=tk.HORIZONTAL, length=150, command=lambda _: self.actualizar_rangos_color())
        deslizador_tamanio.grid(row=0, column=1, padx=5, pady=2)
        etiqueta_tamanio = ttk.Label(panel_params, textvariable=self.tamanio_minimo_defecto)
        etiqueta_tamanio.grid(row=0, column=2, padx=5, pady=2)

        ttk.Label(panel_params, text="Sensibilidad:").grid(row=1, column=0, padx=5, pady=2)
        deslizador_sensibilidad = ttk.Scale(panel_params, from_=0.5, to=2.0, variable=self.sensibilidad,
                               orient=tk.HORIZONTAL, length=150, command=lambda _: self.actualizar_rangos_color())
        deslizador_sensibilidad.grid(row=1, column=1, padx=5, pady=2)
        etiqueta_sensibilidad = ttk.Label(panel_params, textvariable=self.sensibilidad)
        etiqueta_sensibilidad.grid(row=1, column=2, padx=5, pady=2)

        # Etiquetas para mostrar los valores actuales HSV
        panel_valores = ttk.LabelFrame(self.raiz, text="Valores HSV Actuales")
        panel_valores.pack(fill=tk.X, padx=10, pady=5)

        self.lbl_valores_negro = ttk.Label(panel_valores, text="Negro: H:[0,180] S:[0,255] V:[0,60]")
        self.lbl_valores_negro.pack(side=tk.LEFT, padx=10, pady=5)

        self.lbl_valores_verde = ttk.Label(panel_valores, text="Verde: H:[35,90] S:[50,255] V:[50,255]")
        self.lbl_valores_verde.pack(side=tk.LEFT, padx=10, pady=5)

        # Panel de video
        self.etiqueta_video = ttk.Label(self.raiz)
        self.etiqueta_video.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.etiqueta_video.bind('<Button-1>', self.establecer_centro_zoom)

        # Panel de resultados
        self.texto_resultados = tk.Text(self.raiz, height=8, state=tk.DISABLED, font=('Consolas', 10))
        self.texto_resultados.pack(fill=tk.BOTH, padx=10, pady=10)

        # Etiqueta de estado de modelo
        self.lbl_estado_modelo = ttk.Label(self.raiz, text="Modelos IA: MobileNetV2 (cargado)")
        self.lbl_estado_modelo.pack(pady=5)

    def actualizar_rangos_color(self):
        """
        Actualiza los rangos de color desde los controles deslizantes.

        Esta función se llama cuando el usuario ajusta los controles HSV
        y actualiza los rangos utilizados para la detección de cítricos y defectos.
        También actualiza las etiquetas con los valores actuales y reprocesa
        la imagen estática si está en modo imagen estática.
        """
        self.rangos_citricos['naranja'] = self.control_naranja.obtener_valores()
        self.rangos_citricos['limon'] = self.control_naranja.obtener_valores()  # Usar mismo rango para ambos
        self.rangos_defectos['negro'] = self.control_negro.obtener_valores()
        self.rangos_defectos['verde'] = self.control_verde.obtener_valores()

        # Actualizar etiquetas con valores actuales
        negro_min, negro_max = self.rangos_defectos['negro']
        verde_min, verde_max = self.rangos_defectos['verde']

        self.lbl_valores_negro.config(
            text=f"Negro: H:[{int(negro_min[0])},{int(negro_max[0])}] "
                 f"S:[{int(negro_min[1])},{int(negro_max[1])}] "
                 f"V:[{int(negro_min[2])},{int(negro_max[2])}]"
        )

        self.lbl_valores_verde.config(
            text=f"Verde: H:[{int(verde_min[0])},{int(verde_max[0])}] "
                 f"S:[{int(verde_min[1])},{int(verde_max[1])}] "
                 f"V:[{int(verde_min[2])},{int(verde_max[2])}]"
        )

        # Si estamos en modo imagen estática, procesar de nuevo
        if self.modo_imagen_estatica and self.imagen_estatica is not None:
            fotograma_proc, texto = self.procesar_fotograma(self.imagen_estatica)
            self.ultimo_fotograma = fotograma_proc.copy()
            self.actualizar_gui(fotograma_proc, texto)

    def cargar_imagen(self):
        """
        Carga una imagen estática desde el disco para analizarla.

        Abre un diálogo de selección de archivo, carga la imagen seleccionada,
        la procesa y muestra los resultados en la interfaz.
        """
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=(("Imágenes", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*"))
        )

        if ruta:
            try:
                # Detener cámara si está activa
                if self.analizando:
                    self.detener_analisis()

                # Cargar imagen
                self.imagen_estatica = cv2.imread(ruta)
                if self.imagen_estatica is None:
                    raise Exception("No se pudo cargar la imagen")

                # Ajustar tamaño si es necesario
                alto, ancho = self.imagen_estatica.shape[:2]
                if alto > 800 or ancho > 1000:
                    escala = min(800 / alto, 1000 / ancho)
                    nueva_altura = int(alto * escala)
                    nuevo_ancho = int(ancho * escala)
                    self.imagen_estatica = cv2.resize(self.imagen_estatica, (nuevo_ancho, nueva_altura))

                self.modo_imagen_estatica = True

                # Procesar imagen
                fotograma_proc, texto = self.procesar_fotograma(self.imagen_estatica)
                self.ultimo_fotograma = fotograma_proc.copy()
                self.actualizar_gui(fotograma_proc, texto)

                # Habilitar captura
                self.boton_capturar.config(state=tk.NORMAL)

            except Exception as e:
                messagebox.showerror("Error", f"No se pudo procesar la imagen: {e}")

    def iniciar_analisis(self):
        """
        Inicia la captura de video y el análisis de cítricos.

        Configura la cámara, habilita/deshabilita los botones correspondientes
        e inicia un hilo separado para el procesamiento de video.
        """
        if not self.analizando:
            self.analizando = True
            self.modo_imagen_estatica = False
            self.boton_iniciar.config(state=tk.DISABLED)
            self.boton_detener.config(state=tk.NORMAL)
            self.boton_capturar.config(state=tk.NORMAL)

            try:
                self.camara = cv2.VideoCapture(0)
                if not self.camara.isOpened():
                    raise Exception("No se pudo abrir la cámara")

                self.camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camara.set(cv2.CAP_PROP_FPS, 30)

                threading.Thread(target=self.procesar_video, daemon=True).start()
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo iniciar la cámara: {e}")
                self.detener_analisis()

    def detener_analisis(self):
        """
        Detiene la captura de video y el análisis.

        Actualiza los estados de los botones y libera los recursos de la cámara.
        """
        if self.analizando:
            self.analizando = False
            self.boton_iniciar.config(state=tk.NORMAL)
            self.boton_detener.config(state=tk.DISABLED)
            self.boton_capturar.config(state=tk.DISABLED)
            if self.camara:
                self.camara.release()
            self.actualizar_gui(np.zeros((480, 640, 3), np.uint8), "Sistema detenido")

    def procesar_video(self):
        """
        Procesa continuamente los fotogramas de video mientras el análisis está activo.

        Este método se ejecuta en un hilo separado para evitar bloquear la interfaz.
        Captura cada fotograma, aplica el zoom seleccionado, lo procesa con los detectores
        y actualiza la interfaz con los resultados.
        """
        while self.analizando and self.camara.isOpened():
            exito, fotograma = self.camara.read()
            if not exito:
                print("Error al leer el fotograma de la cámara")
                break

            fotograma = self.aplicar_zoom(fotograma)
            fotograma_proc, texto = self.procesar_fotograma(fotograma)

            self.ultimo_fotograma = fotograma_proc.copy()
            self.actualizar_gui(fotograma_proc, texto)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.detener_analisis()

    def procesar_fotograma(self, fotograma):
        """
        Procesa cada fotograma utilizando los detectores de IA.

        Detecta cítricos en el fotograma, analiza defectos en cada uno y genera
        una visualización con resultados.

        Args:
            fotograma: Imagen en formato BGR (OpenCV) a procesar

        Returns:
            tuple: (fotograma_resultado, texto_resultado)
                  - fotograma_resultado: Imagen procesada con anotaciones visuales
                  - texto_resultado: Texto descriptivo de los resultados de análisis
        """
        texto_resultado = ""
        fotograma_resultado = fotograma.copy()

        try:
            # Configurar rangos personalizados para piezas
            rangos_personalizados = {
                'naranja': self.rangos_citricos['naranja'],
                'limon': self.rangos_citricos['limon']
            }

            # Detectar piezas pasando los rangos HSV actualizados
            detecciones, mascara_piezas, contornos = self.detector_citricos.detectar(
                fotograma, rangos_personalizados)

            # Si el modo depuración está activado, mostrar máscaras
            if self.modo_depuracion.get():
                # Mostrar máscara de detección de piezas
                mascara_pequenha = cv2.resize(mascara_piezas, (160, 120))
                mascara_color = cv2.cvtColor(mascara_pequenha, cv2.COLOR_GRAY2BGR)
                fotograma_resultado[10:130, 10:170] = mascara_color
                cv2.putText(fotograma_resultado, "Piezas", (15, 145),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Información de diagnóstico
            num_piezas = len(detecciones)
            cv2.putText(fotograma_resultado, f"Piezas: {num_piezas}",
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            for idx, pieza in enumerate(detecciones):
                # Obtener información de la detección
                x1, y1, x2, y2 = pieza['bbox']

                # Color único para todas las piezas
                color_tipo = (0, 200, 255)  # Color naranja-amarillo para todas las piezas

                # Extraer la región de la pieza
                region_pieza = fotograma[y1:y2, x1:x2]
                if region_pieza.size == 0:
                    continue

                # Crear máscara específica para esta pieza
                mascara_pieza = np.zeros_like(mascara_piezas)
                # Si hay un contorno disponible, usarlo, si no, usar el rectángulo
                if 'contorno' in pieza:
                    cv2.drawContours(mascara_pieza, [pieza['contorno']], 0, 255, -1)
                else:
                    cv2.rectangle(mascara_pieza, (x1, y1), (x2, y2), 255, -1)

                # Recortar la máscara para la región
                mascara_region = mascara_pieza[y1:y2, x1:x2]

                # Analizar defectos pasando todos los parámetros ajustables
                defectos, area_defectos, mascara_defectos = self.analizador_defectos.analizar(
                    region_pieza,
                    mascara_region,
                    self.rangos_defectos,
                    self.tamanio_minimo_defecto.get(),
                    self.sensibilidad.get()
                )

                # Si hay modo depuración, mostrar la máscara de defectos
                if self.modo_depuracion.get() and idx == 0:  # Solo para la primera pieza
                    # Mostrar máscara de defectos
                    defectos_pequenhos = cv2.resize(mascara_defectos, (160, 120))
                    color_defectos = cv2.cvtColor(defectos_pequenhos, cv2.COLOR_GRAY2BGR)
                    color_defectos[defectos_pequenhos > 0] = [0, 0, 255]  # Colorear en rojo
                    fotograma_resultado[10:130, 180:340] = color_defectos
                    cv2.putText(fotograma_resultado, "Defectos", (185, 145),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Clasificar la pieza según defectos
                calidad, color_calidad, texto_defectos, porcentaje = clasificar_fruta(
                    defectos, pieza['area'])

                # Dibujar resultados en la imagen
                cv2.rectangle(fotograma_resultado, (x1, y1), (x2, y2), color_tipo, 2)
                cv2.putText(fotograma_resultado,
                          f"Pieza - {calidad} ({porcentaje:.1f}%)",
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_tipo, 2)

                # Dibujar información sobre defectos
                if defectos:
                    pos_y = y1 - 30
                    for defecto, area in defectos.items():
                        cv2.putText(fotograma_resultado, f"{defecto}: {(area/pieza['area'])*100:.1f}%",
                                   (x1, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        pos_y -= 15

                # Dibujar contorno
                if 'contorno' in pieza:
                    cv2.drawContours(fotograma_resultado, [pieza['contorno']], 0, color_tipo, 2)

                # Actualizar texto de resultados
                texto_resultado += f"Pieza {idx+1} - {calidad}\n"
                texto_resultado += f"Circularidad: {pieza.get('circularidad', 0):.2f}\n"
                texto_resultado += f"Defectos: {texto_defectos}\n"
                texto_resultado += f"Porcentaje: {porcentaje:.1f}%\n\n"

        except Exception as e:
            texto_resultado = f"Error en procesamiento: {str(e)}"
            print(f"Error en procesamiento: {str(e)}")
            import traceback
            traceback.print_exc()

        return fotograma_resultado, texto_resultado or "Sin piezas detectadas"

    def actualizar_gui(self, fotograma, texto):
        """
        Actualiza la interfaz gráfica con el fotograma procesado y el texto de resultados.

        Args:
            fotograma: Imagen procesada para mostrar en la interfaz
            texto: Texto de resultados para mostrar en el panel de texto
        """
        img = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        self.etiqueta_video.imgtk = img
        self.etiqueta_video.configure(image=img)

        self.texto_resultados.config(state=tk.NORMAL)
        self.texto_resultados.delete(1.0, tk.END)
        self.texto_resultados.insert(tk.END, texto)
        self.texto_resultados.config(state=tk.DISABLED)

    def capturar_imagen(self):
        """
        Guarda el último fotograma procesado y registra la información de la clasificación.

        Guarda la imagen en el directorio de capturas y añade una entrada en el archivo CSV
        con la información de fecha, nombre de archivo, tipo de pieza, calidad, defectos y
        porcentaje.
        """
        if self.ultimo_fotograma is not None:
            nombre = f"pieza_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            ruta_completa = os.path.join(self.ruta_guardado, nombre)

            cv2.imwrite(ruta_completa, self.ultimo_fotograma)

            # Extraer información de resultados
            try:
                lineas = self.texto_resultados.get(1.0, tk.END).strip().split('\n')
                if len(lineas) > 1 and '-' in lineas[0]:
                    tipo_pieza = "Pieza"  # Siempre "Pieza" sin distinguir
                    calidad = lineas[0].split('-')[1].strip()
                    defectos = lineas[2].split('Defectos:')[1].strip()
                    porcentaje = lineas[3].split('Porcentaje:')[1].split('%')[0].strip()
                else:
                    tipo_pieza = "Desconocido"
                    calidad = "No analizado"
                    defectos = "Ninguno"
                    porcentaje = "0.0"
            except Exception:
                tipo_pieza = "Error"
                calidad = "Error al analizar"
                defectos = "Desconocido"
                porcentaje = "0.0"

            with open('informe.csv', 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    nombre,
                    tipo_pieza,
                    calidad,
                    defectos,
                    float(porcentaje)
                ])

            messagebox.showinfo("Éxito", f"Imagen guardada en:\n{ruta_completa}")

    def establecer_centro_zoom(self, evento):
        """
        Establece el centro de zoom en el punto donde se hizo clic.

        Args:
            evento: Evento de clic que contiene las coordenadas x, y del cursor
        """
        # Si estamos haciendo zoom, establecer centro
        if self.nivel_zoom > 1.0:
            self.centro_zoom = (evento.x / self.etiqueta_video.winfo_width(),
                                evento.y / self.etiqueta_video.winfo_height())

    def aumentar_zoom(self):
        """
        Incrementa el nivel de zoom en 0.2x hasta un máximo de 5.0x.
        """
        self.nivel_zoom = min(5.0, self.nivel_zoom + 0.2)
        self.etiqueta_zoom.config(text=f"Zoom: {self.nivel_zoom:.1f}x")

    def disminuir_zoom(self):
        """
        Reduce el nivel de zoom en 0.2x hasta un mínimo de 1.0x.
        """
        self.nivel_zoom = max(1.0, self.nivel_zoom - 0.2)
        self.etiqueta_zoom.config(text=f"Zoom: {self.nivel_zoom:.1f}x")

    def reiniciar_zoom(self):
        """
        Restablece el zoom a su valor predeterminado (1.0x) y elimina el centro personalizado.
        """
        self.nivel_zoom = 1.0
        self.centro_zoom = None
        self.etiqueta_zoom.config(text="Zoom: 1.0x")

    def aplicar_zoom(self, imagen):
        """
        Aplica zoom a la imagen según el nivel y centro establecidos.

        Args:
            imagen: Imagen original a la que aplicar zoom

        Returns:
            ndarray: Imagen con zoom aplicado
        """
        if self.nivel_zoom <= 1.0:
            return imagen

        alto, ancho = imagen.shape[:2]
        nuevo_alto = int(alto / self.nivel_zoom)
        nuevo_ancho = int(ancho / self.nivel_zoom)

        if self.centro_zoom is None:
            centro_x = ancho // 2
            centro_y = alto // 2
        else:
            centro_x = int(self.centro_zoom[0] * ancho)
            centro_y = int(self.centro_zoom[1] * alto)

        x1 = max(0, centro_x - nuevo_ancho // 2)
        y1 = max(0, centro_y - nuevo_alto // 2)
        x2 = min(ancho, x1 + nuevo_ancho)
        y2 = min(alto, y1 + nuevo_alto)

        if x2 - x1 < nuevo_ancho:
            x1 = max(0, ancho - nuevo_ancho)
            x2 = ancho
        if y2 - y1 < nuevo_alto:
            y1 = max(0, alto - nuevo_alto)
            y2 = alto

        recortada = imagen[y1:y2, x1:x2]
        return cv2.resize(recortada, (ancho, alto), interpolation=cv2.INTER_LINEAR)

if __name__ == "__main__":
    # Configuración TensorFlow para reducir advertencias
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    try:
        raiz = tk.Tk()
        app = Aplicacion(raiz)
        raiz.protocol("WM_DELETE_WINDOW", lambda: [app.detener_analisis(), raiz.destroy()])
        raiz.mainloop()
    except Exception as e:
        print(f"Error de aplicación: {e}")