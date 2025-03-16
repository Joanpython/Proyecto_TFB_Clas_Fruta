"""
Sistema de Clasificación de Frutas mediante Visión Artificial
===========================================================

Este programa implementa un sistema de visión artificial para clasificar frutas
basándose en la detección de defectos superficiales mediante procesamiento de imagen
en tiempo real.

Características principales:
- Procesamiento de video en tiempo real
- Detección de frutas y sus defectos
- Interfaz gráfica con controles ajustables
- Generación de informes en CSV
- Captura y almacenamiento de imágenes

Autor: [Joan Ruiz Verdú]
Fecha: [23/02/2025]
Versión: 1.0
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

# io: Módulo de la biblioteca estándar de Python para operaciones de entrada/salida
# Desarrollado y mantenido por Python Software Foundation
import io

# base64: Módulo de la biblioteca estándar de Python para codificar y decodificar datos en base64
# Desarrollado y mantenido por Python Software Foundation
import base64


# =============================================
# CONSTANTES Y CONFIGURACIONES GLOBALES
# =============================================
HSV_GLOBAL = None
ULTIMO_CLIC = None

class ControlColor(ttk.LabelFrame):
    """
    Clase para crear un conjunto de controles deslizantes para ajustar rangos HSV.

    Esta clase genera un panel con sliders para configurar los valores mínimos y máximos
    de los componentes H, S y V de un rango de color.

    Args:
        parent: Widget padre al que se añadirá este control.
        titulo (str): Título para el marco del control.
        valor_inicial_min (list): Valores iniciales [H, S, V] mínimos.
        valor_inicial_max (list): Valores iniciales [H, S, V] máximos.
        callback (callable): Función a llamar cuando cambie cualquier valor.
    """
    def __init__(self, parent, titulo, valor_inicial_min, valor_inicial_max, callback):
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
        Devuelve los valores actuales de los rangos HSV configurados.

        Returns:
            tuple: Dos arrays numpy con los valores mínimos y máximos de HSV.
        """
        return (np.array([self.h_min.get(), self.s_min.get(), self.v_min.get()]),
                np.array([self.h_max.get(), self.s_max.get(), self.v_max.get()]))

# =============================================
# FUNCIONES DE PROCESAMIENTO DE IMAGEN
# =============================================

def corregir_iluminacion(imagen):
    """
    Aplica corrección de iluminación utilizando CLAHE.

    Mejora el contraste local de la imagen convirtiendo a espacio LAB
    y aplicando ecualización de histograma adaptativa con límite de contraste.

    Args:
        imagen (numpy.ndarray): Imagen BGR a procesar.

    Returns:
        numpy.ndarray: Imagen con iluminación corregida.
    """
    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_corregido = clahe.apply(l)
    lab_corregido = cv2.merge((l_corregido, a, b))
    return cv2.cvtColor(lab_corregido, cv2.COLOR_LAB2BGR)

def configurar_rangos_color():
    """
    Define los rangos HSV iniciales para la detección de diferentes colores.

    Returns:
        dict: Diccionario con los rangos HSV predefinidos para cada color.
              Cada valor es una tupla con dos arrays numpy (min, max).
    """
    return {
        'azul': (np.array([35, 100, 50]), np.array([180, 255, 255])),
        'negro': (np.array([0, 0, 0]), np.array([180, 28, 28])),
        'verde': (np.array([35, 50, 50]), np.array([85, 255, 255]))
    }

def crear_directorios():
    """
    Crea la estructura de directorios necesaria para guardar capturas y el archivo CSV.

    Crea el directorio './capturas/' si no existe y el archivo 'informe.csv'
    con encabezados si este no existe.

    Returns:
        str: Ruta al directorio de capturas.
    """
    ruta_guardado = "./capturas/"
    os.makedirs(ruta_guardado, exist_ok=True)

    if not os.path.isfile('informe.csv'):
        with open('informe.csv', 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['Fecha','Imagen','Calidad','Defectos','Porcentaje Defectos'])

    return ruta_guardado

def procesar_fotograma(fotograma, rangos, parametros):
    """
    Procesa cada fotograma para detectar y clasificar frutas.

    Aplica preprocesamiento a la imagen, detecta contornos de frutas y evalúa
    defectos para clasificar la calidad del producto.

    Args:
        fotograma (numpy.ndarray): Imagen BGR a procesar.
        rangos (dict): Diccionario con los rangos HSV para cada color.
        parametros (dict): Parámetros de configuración para la detección.

    Returns:
        tuple: Par (fotograma_resultado, texto_resultado) donde:
            - fotograma_resultado es la imagen con anotaciones
            - texto_resultado es el análisis textual de los resultados

    Global:
        HSV_GLOBAL: Actualiza la variable global con la imagen HSV actual
        ULTIMO_CLIC: Utilizado para el seguimiento de clics de usuario
    """
    global HSV_GLOBAL, ULTIMO_CLIC

    # Preprocesamiento
    fotograma = cv2.GaussianBlur(fotograma, (9,9), 0)
    fotograma_corregido = corregir_iluminacion(fotograma)
    hsv = cv2.cvtColor(fotograma_corregido, cv2.COLOR_BGR2HSV)
    HSV_GLOBAL = hsv.copy()

    # Detección de fondo
    mascara_azul = cv2.inRange(hsv, *rangos['azul'])
    nucleo_grande = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    mascara_azul = cv2.morphologyEx(mascara_azul, cv2.MORPH_CLOSE, nucleo_grande, iterations=3)

    # Detección de objetos
    mascara_objeto = cv2.bitwise_not(mascara_azul)
    mascara_objeto = cv2.morphologyEx(mascara_objeto, cv2.MORPH_OPEN, np.ones((7,7)), iterations=2)

    contornos, _ = cv2.findContours(mascara_objeto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fotograma_resultado = fotograma.copy()
    texto_resultado = ""

    # Análisis de cada contorno detectado
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        x, y, w, h = cv2.boundingRect(contorno)
        relacion_aspecto = w/h
        solidez = area/(w*h) if w*h > 0 else 0

        if (area > parametros['area_min_fruta'] and
            0.7 < relacion_aspecto < 1.3 and
            solidez > 0.7):

            region_interes = fotograma[y:y+h, x:x+w]
            defectos = detectar_defectos(region_interes, rangos, parametros)
            area_fruta = w*h
            calidad, color, texto_defectos, porcentaje = clasificar_fruta(defectos, parametros, area_fruta)

            cv2.rectangle(fotograma_resultado, (x,y), (x+w,y+h), color, 3)
            cv2.putText(fotograma_resultado, f"{calidad} ({porcentaje:.1f}%)",
                       (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            texto_resultado += f"Objeto en ({x},{y}): {calidad}\nDefectos: {texto_defectos}\n\n"

    return fotograma_resultado, texto_resultado or "Sin detecciones válidas"

def detectar_defectos(region_interes, rangos, parametros):
    """
    Detecta defectos en una región de interés.

    Analiza una región de imagen para detectar áreas que corresponden a defectos
    como manchas negras o verdes, filtradas por tamaño y forma.

    Args:
        region_interes (numpy.ndarray): Región de la imagen a analizar.
        rangos (dict): Diccionario con los rangos HSV para cada color.
        parametros (dict): Parámetros de configuración para la detección.

    Returns:
        dict: Diccionario con los tipos de defectos detectados y su área total.
    """
    hsv = cv2.cvtColor(region_interes, cv2.COLOR_BGR2HSV)
    defectos = {}

    for tipo in ['negro', 'verde']:
        mascara = cv2.inRange(hsv, *rangos[tipo])
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((5,5)), iterations=2)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((3,3)), iterations=1)

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            x, y, w, h = cv2.boundingRect(contorno)

            if area > parametros['area_min_defecto'] and (0.3 < w/h < 3.0):
                defectos[tipo] = defectos.get(tipo, 0) + area

    return defectos

def clasificar_fruta(defectos, parametros, area_fruta):
    """
    Clasifica la fruta según el porcentaje de defectos.

    Args:
        defectos (dict): Diccionario con los tipos de defectos y sus áreas.
        parametros (dict): Parámetros de configuración para la clasificación.
        area_fruta (float): Área total de la fruta en píxeles.

    Returns:
        tuple: (calidad, color, texto_defectos, porcentaje) donde:
            - calidad (str): Categoría de calidad ("Excelente", "Aceptable", "Rechazada")
            - color (tuple): Color BGR para visualización
            - texto_defectos (str): Descripción textual de los defectos
            - porcentaje (float): Porcentaje del área con defectos
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
    Clase principal que maneja la interfaz gráfica y la lógica de la aplicación.

    Esta clase implementa una aplicación de clasificación de frutas que:
    - Captura video de la cámara
    - Detecta frutas y sus defectos
    - Clasifica las frutas según la calidad
    - Permite ajustar parámetros de detección
    - Guarda imágenes y resultados

    Args:
        raiz (tk.Tk): Ventana principal de Tkinter.
    """
    def __init__(self, raiz):
        """
        Inicializa la aplicación y configura la interfaz gráfica.

        Args:
            raiz (tk.Tk): Ventana principal de Tkinter.
        """
        self.raiz = raiz
        self.camara = None
        self.analizando = False
        self.ultimo_fotograma = None
        self.ruta_guardado = crear_directorios()
        self.rangos = configurar_rangos_color()
        self.parametros = {
            'area_min_fruta': 1000,
            'area_min_defecto': 50,
            'ratio_defecto_max': 3.0
        }
        self.zoom_level = 1.0
        self.zoom_center = None
        self.cargar_logo()
        self.configurar_gui()

    def cargar_logo(self):
        """
        Carga el logo UCMA desde el archivo.

        Intenta cargar un logo desde el archivo "Logo.png", redimensionarlo
        para diferentes propósitos (icono y visualización) y manejar cualquier error.
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
        Configura la interfaz gráfica de la aplicación.

        Establece la estructura de la ventana principal, crea todos los widgets
        de la interfaz y configura sus propiedades.
        """
        self.raiz.title("Clasificador de Frutas - UCMA")
        self.raiz.geometry("1200x850")  # Aumentado para acomodar el logo

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
                              text="Sistema de Clasificación de Frutas",
                              font=("Helvetica", 16, "bold"))
        lbl_titulo.pack(side=tk.LEFT, padx=20, pady=5)

        # Panel principal de control
        panel_control = ttk.Frame(self.raiz)
        panel_control.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Panel de botones básicos
        panel_botones = ttk.LabelFrame(panel_control, text="Controles Básicos")
        panel_botones.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_iniciar = ttk.Button(panel_botones, text="Iniciar", command=self.iniciar_analisis)
        self.btn_iniciar.pack(side=tk.LEFT, padx=5)

        self.btn_detener = ttk.Button(panel_botones, text="Detener", state=tk.DISABLED,
                                    command=self.detener_analisis)
        self.btn_detener.pack(side=tk.LEFT, padx=5)

        self.btn_capturar = ttk.Button(panel_botones, text="Capturar", state=tk.DISABLED,
                                     command=self.capturar_imagen)
        self.btn_capturar.pack(side=tk.LEFT, padx=5)

        # Panel de zoom
        panel_zoom = ttk.LabelFrame(panel_control, text="Control de Zoom")
        panel_zoom.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_zoom_in = ttk.Button(panel_zoom, text="Zoom +", command=self.zoom_in)
        self.btn_zoom_in.pack(side=tk.LEFT, padx=5)

        self.btn_zoom_out = ttk.Button(panel_zoom, text="Zoom -", command=self.zoom_out)
        self.btn_zoom_out.pack(side=tk.LEFT, padx=5)

        self.btn_reset_zoom = ttk.Button(panel_zoom, text="Reset Zoom", command=self.reset_zoom)
        self.btn_reset_zoom.pack(side=tk.LEFT, padx=5)

        self.lbl_zoom = ttk.Label(panel_zoom, text="Zoom: 1.0x")
        self.lbl_zoom.pack(side=tk.LEFT, padx=5)

        # Panel para mostrar valores HSV
        self.frame_hsv = ttk.LabelFrame(self.raiz, text="Valores HSV")
        self.frame_hsv.pack(fill=tk.X, padx=10, pady=5)

        self.lbl_hsv = ttk.Label(self.frame_hsv, text="Haz clic en la imagen para ver valores HSV")
        self.lbl_hsv.pack(padx=5, pady=5)

        # Panel de controles de color
        self.frame_colores = ttk.LabelFrame(self.raiz, text="Ajustes de Color")
        self.frame_colores.pack(fill=tk.X, padx=10, pady=5)

        # Controles para cada color
        self.control_azul = ControlColor(
            self.frame_colores, "Azul (Fondo)",
            self.rangos['azul'][0], self.rangos['azul'][1],
            self.actualizar_rangos_color
        )
        self.control_azul.pack(side=tk.LEFT, padx=5, pady=5)

        self.control_negro = ControlColor(
            self.frame_colores, "Negro (Defectos)",
            self.rangos['negro'][0], self.rangos['negro'][1],
            self.actualizar_rangos_color
        )
        self.control_negro.pack(side=tk.LEFT, padx=5, pady=5)

        self.control_verde = ControlColor(
            self.frame_colores, "Verde (Defectos)",
            self.rangos['verde'][0], self.rangos['verde'][1],
            self.actualizar_rangos_color
        )
        self.control_verde.pack(side=tk.LEFT, padx=5, pady=5)

        # Panel de video
        self.lbl_video = ttk.Label(self.raiz)
        self.lbl_video.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.lbl_video.bind('<Button-1>', self.set_zoom_center)

        # Panel de resultados
        self.txt_resultados = tk.Text(self.raiz, height=8, state=tk.DISABLED, font=('Consolas', 10))
        self.txt_resultados.pack(fill=tk.BOTH, padx=10, pady=10)

    def actualizar_valores_hsv(self, h, s, v):
        """
        Actualiza la etiqueta con los valores HSV.

        Args:
            h (int): Valor del componente Hue.
            s (int): Valor del componente Saturation.
            v (int): Valor del componente Value.
        """
        self.lbl_hsv.config(text=f"Valores HSV en el punto: H={h}, S={s}, V={v}")

    def actualizar_rangos_color(self):
        """
        Actualiza los rangos de color desde los controles deslizantes.

        Obtiene los valores actuales de los controles deslizantes y
        actualiza el diccionario de rangos de color.
        """
        self.rangos['azul'] = self.control_azul.obtener_valores()
        self.rangos['negro'] = self.control_negro.obtener_valores()
        self.rangos['verde'] = self.control_verde.obtener_valores()

    def mostrar_mascaras(self, hsv):
        """
        Muestra las máscaras de cada color en ventanas separadas.

        Args:
            hsv (numpy.ndarray): Imagen HSV sobre la que se generarán las máscaras.
        """
        for nombre, rango in self.rangos.items():
            mascara = cv2.inRange(hsv, *rango)
            cv2.imshow(f'Máscara {nombre}', mascara)

    def iniciar_analisis(self):
        """
        Inicia la captura de video y el análisis de frutas.

        Configura la cámara, habilita/deshabilita botones apropiados e inicia
        un hilo para procesar el video.
        """
        if not self.analizando:
            self.analizando = True
            self.btn_iniciar.config(state=tk.DISABLED)
            self.btn_detener.config(state=tk.NORMAL)
            self.btn_capturar.config(state=tk.NORMAL)
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

        Libera los recursos de la cámara, actualiza la interfaz y cambia
        el estado de los botones.
        """
        if self.analizando:
            self.analizando = False
            self.btn_iniciar.config(state=tk.NORMAL)
            self.btn_detener.config(state=tk.DISABLED)
            self.btn_capturar.config(state=tk.DISABLED)
            if self.camara:
                self.camara.release()
            self.actualizar_gui(np.zeros((480, 640, 3), np.uint8), "Sistema detenido")

    def procesar_video(self):
        """
        Procesa continuamente los fotogramas de video.

        Este método se ejecuta en un hilo separado, captura fotogramas de la
        cámara, los procesa para detectar frutas y muestra los resultados.
        """
        while self.analizando and self.camara.isOpened():
            exito, fotograma = self.camara.read()
            if not exito:
                break

            fotograma = self.aplicar_zoom(fotograma)
            fotograma_proc, texto = procesar_fotograma(fotograma, self.rangos, self.parametros)

            # Mostrar máscaras
            self.mostrar_mascaras(fotograma)

            self.ultimo_fotograma = fotograma_proc.copy()
            self.actualizar_gui(fotograma_proc, texto)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.detener_analisis()

    def actualizar_gui(self, fotograma, texto):
        """
        Actualiza la interfaz gráfica con un nuevo fotograma y texto.

        Args:
            fotograma (numpy.ndarray): Imagen a mostrar en la interfaz.
            texto (str): Texto de resultados a mostrar.
        """
        img = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        self.lbl_video.imgtk = img
        self.lbl_video.configure(image=img)

        self.txt_resultados.config(state=tk.NORMAL)
        self.txt_resultados.delete(1.0, tk.END)
        self.txt_resultados.insert(tk.END, texto)
        self.txt_resultados.config(state=tk.DISABLED)

    def capturar_imagen(self):
        """
        Captura y guarda la imagen actual, registrando sus datos en el informe.

        Guarda el fotograma actual como archivo de imagen y registra información
        sobre la calidad y defectos en el archivo CSV de informe.
        """
        if self.ultimo_fotograma is not None:
            nombre = f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            ruta_completa = os.path.join(self.ruta_guardado, nombre)

            cv2.imwrite(ruta_completa, self.ultimo_fotograma)

            lineas = self.txt_resultados.get(1.0, tk.END).split('\n')
            calidad = lineas[0].split(': ')[-1] if len(lineas) > 0 else "Desconocida"
            defectos = lineas[1].split(': ')[-1] if len(lineas) > 1 else "Ninguno"
            porcentaje = lineas[0].split('(')[-1].split('%')[0] if '%' in lineas[0] else "0.0"

            with open('informe.csv', 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    nombre,
                    calidad,
                    defectos,
                    float(porcentaje)
                ])

            messagebox.showinfo("Éxito", f"Imagen guardada en:\n{ruta_completa}")

    def zoom_in(self):
        """
        Aumenta el nivel de zoom de la visualización.

        Incrementa el factor de zoom hasta un máximo de 5.0 y actualiza la etiqueta.
        """
        self.zoom_level = min(5.0, self.zoom_level + 0.2)
        self.lbl_zoom.config(text=f"Zoom: {self.zoom_level:.1f}x")

    def zoom_out(self):
        """
        Disminuye el nivel de zoom de la visualización.

        Decrementa el factor de zoom hasta un mínimo de 1.0 y actualiza la etiqueta.
        """
        self.zoom_level = max(1.0, self.zoom_level - 0.2)
        self.lbl_zoom.config(text=f"Zoom: {self.zoom_level:.1f}x")

    def reset_zoom(self):
        """
        Restablece el zoom a los valores predeterminados.

        Establece el nivel de zoom en 1.0 y elimina el centro de zoom.
        """
        self.zoom_level = 1.0
        self.zoom_center = None
        self.lbl_zoom.config(text="Zoom: 1.0x")

    def set_zoom_center(self, event):
        """
        Establece el centro del zoom basado en la posición del clic.

        Args:
            event: Evento de clic del ratón que contiene las coordenadas x,y.
        """
        if self.zoom_level > 1.0:
            self.zoom_center = (event.x / self.lbl_video.winfo_width(),
                              event.y / self.lbl_video.winfo_height())

    def aplicar_zoom(self, imagen):
        """
        Aplica el nivel de zoom actual a una imagen.

        Recorta la imagen alrededor del centro de zoom y la redimensiona
        para mantener las dimensiones originales.

        Args:
            imagen (numpy.ndarray): Imagen original a la que aplicar zoom.

        Returns:
            numpy.ndarray: Imagen con zoom aplicado.
        """
        if self.zoom_level <= 1.0:
            return imagen

        h, w = imagen.shape[:2]
        new_h = int(h / self.zoom_level)
        new_w = int(w / self.zoom_level)

        if self.zoom_center is None:
            center_x = w // 2
            center_y = h // 2
        else:
            center_x = int(self.zoom_center[0] * w)
            center_y = int(self.zoom_center[1] * h)

        x1 = max(0, center_x - new_w // 2)
        y1 = max(0, center_y - new_h // 2)
        x2 = min(w, x1 + new_w)
        y2 = min(h, y1 + new_h)

        if x2 - x1 < new_w:
            x1 = max(0, w - new_w)
            x2 = w
        if y2 - y1 < new_h:
            y1 = max(0, h - new_h)
            y2 = h

        cropped = imagen[y1:y2, x1:x2]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

if __name__ == "__main__":
    raiz = tk.Tk()
    app = Aplicacion(raiz)
    raiz.protocol("WM_DELETE_WINDOW", lambda: [app.detener_analisis(), raiz.destroy()])
    raiz.mainloop()