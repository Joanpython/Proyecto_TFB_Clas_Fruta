"""
Sistema de Clasificación de Naranjas con Detección de Defectos
===========================================================

Este programa implementa un sistema híbrido que combina detección de objetos con YOLOv10
y análisis de defectos mediante procesamiento de imagen tradicional para clasificar
naranjas en tiempo real.

Características principales:
- Detección precisa de naranjas usando YOLOv10
- Análisis de defectos superficiales mediante procesamiento de imagen
- Máscara de segmentación para analizar solo la superficie de la naranja
- Interfaz gráfica con controles ajustables de color
- Generación de informes en CSV
- Captura y almacenamiento de imágenes clasificadas

Autor: [Joan Ruiz Verdú]
Fecha: [03/03/2025]
Versión: 2
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
from tkinter import ttk, messagebox

# PIL (Pillow): Biblioteca para abrir, manipular y guardar diferentes formatos de imagen
# Originalmente Python Imaging Library, ahora mantenida como Pillow por la comunidad
# ImageTk: Módulo de PIL para integración con tkinter
from PIL import Image, ImageTk

#ultralytics: Biblioteca que implementa modelos YOLO (You Only Look Once) para detección de objetos
#Desarrollada y mantenida por Ultralytics LLC, ofrece implementaciones optimizadas de la familia de modelos YOLO
#YOLO: Clase principal que permite cargar, entrenar e inferir con modelos de la familia YOLO
from ultralytics import YOLO

# io: Módulo de la biblioteca estándar de Python para operaciones de entrada/salida
# Desarrollado y mantenido por Python Software Foundation
import io

# base64: Módulo de la biblioteca estándar de Python para codificar y decodificar datos en base64
# Desarrollado y mantenido por Python Software Foundation
import base64

# =============================================
# CONSTANTES Y CONFIGURACIONES GLOBALES
# =============================================
MODELO_YOLO = YOLO('yolov10s.pt')  # Cargar el modelo YOLOv10 preentrenado
HSV_GLOBAL = None
ULTIMO_CLIC = None

class ControlColor(ttk.LabelFrame):
    """
    Clase para crear un conjunto de controles deslizantes para ajustar rangos HSV.

    Proporciona una interfaz gráfica que permite ajustar los valores mínimos y máximos
    de H (Matiz), S (Saturación) y V (Valor) para la detección de colores específicos.

    Attributes:
        h_min, s_min, v_min: Controles deslizantes para los valores mínimos de HSV
        h_max, s_max, v_max: Controles deslizantes para los valores máximos de HSV
    """
    def __init__(self, padre, titulo, valor_inicial_min, valor_inicial_max, retrollamada):
        """
        Inicializa un conjunto de controles para ajustar rangos HSV.

        Args:
            padre: Widget padre en la jerarquía de Tkinter
            titulo: Título para el marco de controles
            valor_inicial_min: Lista o array con los valores iniciales mínimos [H, S, V]
            valor_inicial_max: Lista o array con los valores iniciales máximos [H, S, V]
            retrollamada: Función a invocar cuando cambia algún valor
        """
        super().__init__(padre, text=titulo)

        # Marco para valores mínimos
        marco_min = ttk.LabelFrame(self, text="Mínimos")
        marco_min.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Label(marco_min, text="H:").grid(row=0, column=0)
        self.h_min = ttk.Scale(marco_min, from_=0, to=180, orient=tk.HORIZONTAL)
        self.h_min.set(valor_inicial_min[0])
        self.h_min.grid(row=0, column=1)

        ttk.Label(marco_min, text="S:").grid(row=1, column=0)
        self.s_min = ttk.Scale(marco_min, from_=0, to=255, orient=tk.HORIZONTAL)
        self.s_min.set(valor_inicial_min[1])
        self.s_min.grid(row=1, column=1)

        ttk.Label(marco_min, text="V:").grid(row=2, column=0)
        self.v_min = ttk.Scale(marco_min, from_=0, to=255, orient=tk.HORIZONTAL)
        self.v_min.set(valor_inicial_min[2])
        self.v_min.grid(row=2, column=1)

        # Marco para valores máximos
        marco_max = ttk.LabelFrame(self, text="Máximos")
        marco_max.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Label(marco_max, text="H:").grid(row=0, column=0)
        self.h_max = ttk.Scale(marco_max, from_=0, to=180, orient=tk.HORIZONTAL)
        self.h_max.set(valor_inicial_max[0])
        self.h_max.grid(row=0, column=1)

        ttk.Label(marco_max, text="S:").grid(row=1, column=0)
        self.s_max = ttk.Scale(marco_max, from_=0, to=255, orient=tk.HORIZONTAL)
        self.s_max.set(valor_inicial_max[1])
        self.s_max.grid(row=1, column=1)

        ttk.Label(marco_max, text="V:").grid(row=2, column=0)
        self.v_max = ttk.Scale(marco_max, from_=0, to=255, orient=tk.HORIZONTAL)
        self.v_max.set(valor_inicial_max[2])
        self.v_max.grid(row=2, column=1)

        # Vincular retrollamada a todos los controles
        for control in [self.h_min, self.s_min, self.v_min,
                       self.h_max, self.s_max, self.v_max]:
            control.configure(command=lambda _: retrollamada())

    def obtener_valores(self):
        """
        Devuelve los valores actuales como dos arrays numpy.

        Returns:
            tuple: Dos arrays numpy conteniendo (valores_minimos, valores_maximos)
                  Cada array tiene la forma [H, S, V]
        """
        return (np.array([self.h_min.get(), self.s_min.get(), self.v_min.get()]),
                np.array([self.h_max.get(), self.s_max.get(), self.v_max.get()]))

# =============================================
# FUNCIONES DE PROCESAMIENTO DE IMAGEN
# =============================================

def corregir_iluminacion(imagen):
    """
    Aplica corrección de iluminación utilizando CLAHE (Ecualización Adaptativa de Histograma con Limitación de Contraste).

    Este método mejora el contraste de la imagen, especialmente útil en condiciones
    de iluminación variable, facilitando la detección de características.

    Args:
        imagen: Imagen en formato BGR (OpenCV)

    Returns:
        ndarray: Imagen con iluminación corregida
    """
    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_corregido = clahe.apply(l)
    lab_corregido = cv2.merge((l_corregido, a, b))
    return cv2.cvtColor(lab_corregido, cv2.COLOR_LAB2BGR)

def configurar_rangos_color():
    """
    Define los rangos HSV iniciales para la detección de defectos en naranjas.

    Returns:
        dict: Diccionario con rangos de color para diferentes tipos de defectos:
              - 'negro': Para detectar manchas negras/oscuras
              - 'verde': Para detectar partes verdes no maduras

              Cada rango se representa como una tupla (min_hsv, max_hsv)
    """
    return {
        'negro': (np.array([0, 0, 0]), np.array([180, 50, 50])),      # Manchas negras
        'verde': (np.array([35, 50, 50]), np.array([85, 255, 255]))   # Partes verdes
    }

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
            csv.writer(f).writerow(['Fecha','Imagen','Calidad','Defectos','Porcentaje Defectos'])

    return ruta_guardado

def clasificar_fruta(defectos, parametros, area_fruta):
    """
    Clasifica la fruta según el porcentaje de defectos detectados.

    Args:
        defectos: Diccionario con tipos de defectos y sus áreas en píxeles
        parametros: Diccionario con parámetros de clasificación
        area_fruta: Área total de la fruta en píxeles

    Returns:
        tuple: (calidad, color, texto_defectos, porcentaje)
               - calidad: Texto describiendo la calidad ("Excelente", "Aceptable", "Rechazada")
               - color: Tupla RGB para representar visualmente la calidad
               - texto_defectos: Descripción textual de los defectos encontrados
               - porcentaje: Porcentaje de la superficie con defectos
    """
    total_defectos = sum(defectos.values())
    porcentaje = (total_defectos / area_fruta) * 100 if area_fruta > 0 else 0
    texto_defectos = ", ".join(defectos.keys()) if defectos else "Ninguno"

    if porcentaje < 2:
        return "Excelente", (0, 255, 0), texto_defectos, porcentaje
    elif porcentaje < parametros['ratio_defecto_max']:
        return "Aceptable", (0, 255, 255), texto_defectos, porcentaje
    else:
        return "Rechazada", (0, 0, 255), texto_defectos, porcentaje

class Aplicacion:
    """
    Clase principal que implementa la aplicación de clasificación de naranjas.

    Esta clase gestiona la interfaz gráfica, la captura de video, el procesamiento
    de imágenes, la detección de naranjas y la clasificación según defectos.

    Attributes:
        raiz: Ventana principal de Tkinter
        camara: Objeto de captura de video
        analizando: Indica si el análisis está en curso
        ultimo_fotograma: Último fotograma procesado
        ruta_guardado: Ruta donde se guardan las capturas
        rangos: Rangos HSV para detección de defectos
        parametros: Parámetros para clasificación
        nivel_zoom: Nivel actual de zoom
        centro_zoom: Centro del zoom en coordenadas relativas
    """
    def __init__(self, raiz):
        """
        Inicializa la aplicación de clasificación de naranjas.

        Args:
            raiz: Objeto Tk de Tkinter (ventana principal)
        """
        self.raiz = raiz
        self.camara = None
        self.analizando = False
        self.ultimo_fotograma = None
        self.ruta_guardado = crear_directorios()
        self.rangos = configurar_rangos_color()
        self.parametros = {
            'area_min_defecto': 50,
            'ratio_defecto_max': 5.0
        }
        self.nivel_zoom = 1.0
        self.centro_zoom = None
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
            self.logo_display = ImageTk.PhotoImage(display_temp)
        except Exception as e:
            print(f"Error al cargar el logo: {e}")
            self.icono = None
            self.logo_display = None

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
        self.raiz.title("Clasificador de Naranjas - UCMA")
        self.raiz.geometry("1200x850")

        # Establecer el icono de la ventana
        if self.icono:
            self.raiz.iconphoto(True, self.icono)

        # Panel de logo
        panel_logo = ttk.Frame(self.raiz)
        panel_logo.pack(fill=tk.X, padx=10, pady=5)

        # Mostrar logo en la interfaz
        if self.logo_display:
            lbl_logo = ttk.Label(panel_logo, image=self.logo_display)
            lbl_logo.pack(side=tk.LEFT, padx=5, pady=5)

        # Título de la aplicación
        lbl_titulo = ttk.Label(panel_logo,
                              text="Sistema de Clasificación de Naranjas con YOLOv10",
                              font=("Helvetica", 16, "bold"))
        lbl_titulo.pack(side=tk.LEFT, padx=20, pady=5)

        # Panel principal de control
        panel_control = ttk.Frame(self.raiz)
        panel_control.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Panel de botones básicos
        panel_botones = ttk.LabelFrame(panel_control, text="Controles Básicos")
        panel_botones.pack(side=tk.LEFT, padx=5, pady=5)

        self.boton_iniciar = ttk.Button(panel_botones, text="Iniciar", command=self.iniciar_analisis)
        self.boton_iniciar.pack(side=tk.LEFT, padx=5)

        self.boton_detener = ttk.Button(panel_botones, text="Detener", state=tk.DISABLED,
                                      command=self.detener_analisis)
        self.boton_detener.pack(side=tk.LEFT, padx=5)

        self.boton_capturar = ttk.Button(panel_botones, text="Capturar", state=tk.DISABLED,
                                     command=self.capturar_imagen)
        self.boton_capturar.pack(side=tk.LEFT, padx=5)

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

        # Panel para mostrar valores HSV
        self.marco_hsv = ttk.LabelFrame(self.raiz, text="Valores HSV")
        self.marco_hsv.pack(fill=tk.X, padx=10, pady=5)

        self.etiqueta_hsv = ttk.Label(self.marco_hsv, text="Haz clic en la imagen para ver valores HSV")
        self.etiqueta_hsv.pack(padx=5, pady=5)

        # Panel de controles de color
        self.marco_colores = ttk.LabelFrame(self.raiz, text="Ajustes de Detección de Defectos")
        self.marco_colores.pack(fill=tk.X, padx=10, pady=5)

        # Controles para cada color de defecto
        self.control_negro = ControlColor(
            self.marco_colores, "Negro (Manchas)",
            self.rangos['negro'][0], self.rangos['negro'][1],
            self.actualizar_rangos_color
        )
        self.control_negro.pack(side=tk.LEFT, padx=5, pady=5)

        self.control_verde = ControlColor(
            self.marco_colores, "Verde (Defectos)",
            self.rangos['verde'][0], self.rangos['verde'][1],
            self.actualizar_rangos_color
        )
        self.control_verde.pack(side=tk.LEFT, padx=5, pady=5)

        # Panel de video
        self.etiqueta_video = ttk.Label(self.raiz)
        self.etiqueta_video.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.etiqueta_video.bind('<Button-1>', self.establecer_centro_zoom)

        # Panel de resultados
        self.texto_resultados = tk.Text(self.raiz, height=8, state=tk.DISABLED, font=('Consolas', 10))
        self.texto_resultados.pack(fill=tk.BOTH, padx=10, pady=10)

    def actualizar_valores_hsv(self, evento):
        """
        Actualiza la etiqueta con los valores HSV del punto donde se hizo clic.

        Args:
            evento: Evento de clic que contiene las coordenadas x, y del cursor
        """
        global HSV_GLOBAL, ULTIMO_CLIC
        ULTIMO_CLIC = (evento.x, evento.y)

        if HSV_GLOBAL is not None:
            # Convertir coordenadas de clic a coordenadas de imagen
            h, w = HSV_GLOBAL.shape[:2]
            x = int(ULTIMO_CLIC[0] * w / self.etiqueta_video.winfo_width())
            y = int(ULTIMO_CLIC[1] * h / self.etiqueta_video.winfo_height())

            if 0 <= x < w and 0 <= y < h:
                h, s, v = HSV_GLOBAL[y, x]
                self.etiqueta_hsv.config(text=f"Valores HSV en el punto: H={h}, S={s}, V={v}")

    def actualizar_rangos_color(self):
        """
        Actualiza los rangos de color desde los controles deslizantes.

        Esta función se llama cuando el usuario ajusta los controles HSV
        y actualiza los rangos utilizados para la detección de defectos.
        """
        self.rangos['negro'] = self.control_negro.obtener_valores()
        self.rangos['verde'] = self.control_verde.obtener_valores()

    def iniciar_analisis(self):
        """
        Inicia la captura de video y el análisis de naranjas.

        Configura la cámara, habilita/deshabilita los botones correspondientes
        e inicia un hilo separado para el procesamiento de video.
        """
        if not self.analizando:
            self.analizando = True
            self.boton_iniciar.config(state=tk.DISABLED)
            self.boton_detener.config(state=tk.NORMAL)
            self.boton_capturar.config(state=tk.NORMAL)
            self.camara = cv2.VideoCapture(0)

            self.camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camara.set(cv2.CAP_PROP_FPS, 30)
            self.camara.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.camara.set(cv2.CAP_PROP_BRIGHTNESS, 128)
            self.camara.set(cv2.CAP_PROP_CONTRAST, 128)

            threading.Thread(target=self.procesar_video, daemon=True).start()

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
        Captura cada fotograma, aplica el zoom seleccionado, lo procesa con YOLOv10
        y actualiza la interfaz con los resultados.
        """
        while self.analizando and self.camara.isOpened():
            exito, fotograma = self.camara.read()
            if not exito:
                break

            fotograma = self.aplicar_zoom(fotograma)
            fotograma_proc, texto = self.procesar_fotograma_yolo(fotograma)

            self.ultimo_fotograma = fotograma_proc.copy()
            self.actualizar_gui(fotograma_proc, texto)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.detener_analisis()

    def crear_mascara_naranja(self, region):
        """
        Crea una máscara que aísla solo la superficie de la naranja.

        Utiliza rangos de color HSV para segmentar la naranja del fondo
        y aplica operaciones morfológicas para mejorar la calidad de la máscara.

        Args:
            region: Región de la imagen donde se encuentra la naranja

        Returns:
            ndarray: Máscara binaria donde los píxeles blancos (255)
                    corresponden a la superficie de la naranja
        """
        # Convertir a HSV para mejor segmentación de color
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Rango de color para naranjas (ajustar según sea necesario)
        naranja_bajo = np.array([5, 100, 100])
        naranja_alto = np.array([25, 255, 255])

        # Crear máscara inicial
        mascara = cv2.inRange(hsv, naranja_bajo, naranja_alto)

        # Operaciones morfológicas para mejorar la máscara
        kernel = np.ones((5, 5), np.uint8)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Opcional: encontrar el contorno más grande y rellenarlo (asumiendo que es la naranja)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contornos:
            contorno_max = max(contornos, key=cv2.contourArea)
            mascara_refinada = np.zeros_like(mascara)
            cv2.drawContours(mascara_refinada, [contorno_max], 0, 255, -1)
            return mascara_refinada

        return mascara

    def detectar_defectos_con_mascara(self, region, mascara_naranja, rangos, parametros):
        """
        Detecta defectos solo dentro de la máscara de naranja.

        Analiza la región de la imagen dentro de la máscara para detectar
        diferentes tipos de defectos basados en rangos HSV específicos.

        Args:
            region: Región de la imagen donde se encuentra la naranja
            mascara_naranja: Máscara binaria que delimita la superficie de la naranja
            rangos: Diccionario con rangos HSV para diferentes tipos de defectos
            parametros: Parámetros para la detección de defectos

        Returns:
            dict: Diccionario con tipos de defectos como claves y área en píxeles como valores
        """
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        defectos = {}

        for tipo in ['negro', 'verde']:
            # Crear máscara para el tipo de defecto
            mascara_defecto = cv2.inRange(hsv, *rangos[tipo])

            # Aplicar la máscara de naranja para detectar defectos solo en la superficie de la naranja
            mascara_defecto = cv2.bitwise_and(mascara_defecto, mascara_naranja)

            # Mejorar la máscara
            mascara_defecto = cv2.morphologyEx(mascara_defecto, cv2.MORPH_OPEN, np.ones((3,3)), iterations=1)
            mascara_defecto = cv2.morphologyEx(mascara_defecto, cv2.MORPH_CLOSE, np.ones((3,3)), iterations=1)

            # Encontrar contornos de defectos
            contornos, _ = cv2.findContours(mascara_defecto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area_total = 0

            for contorno in contornos:
                area = cv2.contourArea(contorno)
                if area > parametros['area_min_defecto']:
                    # Añadir al área total de este tipo de defecto
                    area_total += area

                    # Dibujar contorno del defecto en la región original (opcional)
                    cv2.drawContours(region, [contorno], 0, (0, 0, 255), 1)

            if area_total > 0:
                defectos[tipo] = area_total

        return defectos

    def visualizar_defectos(self, region, defectos, mascara_naranja):
        """
        Crea una visualización de la naranja con defectos destacados.

        Args:
            region: Región de la imagen donde se encuentra la naranja
            defectos: Diccionario con información sobre los defectos detectados
            mascara_naranja: Máscara binaria que delimita la superficie de la naranja

        Returns:
            ndarray: Imagen visualizada con defectos resaltados y fondo oscurecido
        """
        # Crear una copia para visualización
        visualizacion = region.copy()

        # Aplicar un efecto de oscurecimiento al fondo (fuera de la naranja)
        fondo = cv2.bitwise_not(mascara_naranja)
        visualizacion[fondo > 0] = visualizacion[fondo > 0] // 2  # Oscurecer el fondo

        return visualizacion

    def procesar_fotograma_yolo(self, fotograma):
        """
        Procesa cada fotograma utilizando YOLOv10 para detectar naranjas y analizar defectos.

        Aplica corrección de iluminación, detecta objetos con YOLOv10 y, para cada naranja
        detectada, analiza la superficie en busca de defectos y clasifica su calidad.

        Args:
            fotograma: Imagen a procesar capturada desde la cámara

        Returns:
            tuple: (fotograma_resultado, texto_resultado)
                   - fotograma_resultado: Imagen procesada con anotaciones visuales
                   - texto_resultado: Texto descriptivo de los resultados de análisis
        """
        global HSV_GLOBAL

        # Preprocesamiento para mejorar la detección
        fotograma_corregido = corregir_iluminacion(fotograma)
        hsv = cv2.cvtColor(fotograma_corregido, cv2.COLOR_BGR2HSV)
        HSV_GLOBAL = hsv.copy()  # Guardar para consulta de valores HSV

        # Detectar objetos con YOLO
        resultados = MODELO_YOLO(fotograma_corregido)

        texto_resultado = ""
        fotograma_resultado = fotograma.copy()

        # Analizar cada detección de YOLO
        for resultado in resultados:
            for caja in resultado.boxes:
                # Obtener información de la detección
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                id_clase = int(caja.cls[0].item())
                etiqueta = resultado.names[id_clase]
                confianza = float(caja.conf[0].item())

                # Solo procesar si es una naranja ("orange")
                if etiqueta == "orange" and confianza > 0.4:
                    # Extraer la región de la naranja
                    region_naranja = fotograma_corregido[y1:y2, x1:x2]
                    if region_naranja.size == 0:  # Evitar regiones vacías
                        continue

                    # Crear máscara para aislar solo la superficie de la naranja
                    mascara_naranja = self.crear_mascara_naranja(region_naranja)

                    # Detectar defectos solo dentro de la máscara de naranja
                    defectos = self.detectar_defectos_con_mascara(region_naranja, mascara_naranja, self.rangos, self.parametros)

                    # Calcular área real de la naranja (solo píxeles naranja)
                    area_naranja = np.sum(mascara_naranja > 0)

                    # Clasificar la naranja según sus defectos
                    calidad, color, texto_defectos, porcentaje = clasificar_fruta(
                        defectos, self.parametros, area_naranja)

                    # Crear una versión para visualización con defectos destacados
                    region_visualizacion = self.visualizar_defectos(region_naranja, defectos, mascara_naranja)

                    # Actualizar la imagen con resultados
                    cv2.rectangle(fotograma_resultado, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(fotograma_resultado,
                                f"{etiqueta}: {calidad} ({porcentaje:.1f}%)",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Mostrar defectos detectados en la región
                    fotograma_resultado[y1:y2, x1:x2] = region_visualizacion

                    texto_resultado += f"Naranja detectada ({confianza:.2f})\n"
                    texto_resultado += f"Calidad: {calidad}\n"
                    texto_resultado += f"Defectos: {texto_defectos}\n"
                    texto_resultado += f"Porcentaje: {porcentaje:.1f}%\n\n"
                else:
                    # Para otros objetos, solo mostrar etiqueta
                    cv2.rectangle(fotograma_resultado, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(fotograma_resultado, f"{etiqueta} ({confianza:.2f})",
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        return fotograma_resultado, texto_resultado or "Sin naranjas detectadas"

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
        con la información de fecha, nombre de archivo, calidad, defectos y porcentaje.
        """
        if self.ultimo_fotograma is not None:
            nombre = f"naranja_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            ruta_completa = os.path.join(self.ruta_guardado, nombre)

            cv2.imwrite(ruta_completa, self.ultimo_fotograma)

            # Extraer información de resultados
            try:
                lineas = self.texto_resultados.get(1.0, tk.END).strip().split('\n')
                if len(lineas) > 1 and 'Calidad:' in lineas[1]:
                    calidad = lineas[1].split('Calidad:')[1].strip()
                    defectos = lineas[2].split('Defectos:')[1].strip()
                    porcentaje = lineas[3].split('Porcentaje:')[1].split('%')[0].strip()
                else:
                    calidad = "No analizada"
                    defectos = "Ninguno"
                    porcentaje = "0.0"
            except Exception:
                calidad = "Error al analizar"
                defectos = "Desconocido"
                porcentaje = "0.0"

            with open('informe.csv', 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    nombre,
                    calidad,
                    defectos,
                    float(porcentaje)
                ])

            messagebox.showinfo("Éxito", f"Imagen guardada en:\n{ruta_completa}")

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

    def establecer_centro_zoom(self, evento):
        """
        Establece el centro de zoom en el punto donde se hizo clic y actualiza los valores HSV.

        Args:
            evento: Evento de clic que contiene las coordenadas x, y del cursor
        """
        # Si estamos haciendo zoom, establecer centro, si no, mostrar valores HSV
        if self.nivel_zoom > 1.0:
            self.centro_zoom = (evento.x / self.etiqueta_video.winfo_width(),
                                evento.y / self.etiqueta_video.winfo_height())

        # En cualquier caso, mostrar valores HSV
        self.actualizar_valores_hsv(evento)

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
    raiz = tk.Tk()
    app = Aplicacion(raiz)
    raiz.protocol("WM_DELETE_WINDOW", lambda: [app.detener_analisis(), raiz.destroy()])
    raiz.mainloop()