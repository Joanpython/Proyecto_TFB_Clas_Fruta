# Sistema de Clasificación de Frutas mediante Visión Artificial

Este repositorio contiene el código desarrollado como parte de mi tesis doctoral en [Universitat Carlemany].

## Descripción

Este programa implementa un sistema de visión artificial para clasificar frutas
basándose en la detección de defectos superficiales mediante procesamiento de imagen
en tiempo real.

## Características principales
- Procesamiento de video en tiempo real
- Detección de frutas y sus defectos
- Interfaz gráfica con controles ajustables
- Generación de informes en CSV
- Captura y almacenamiento de imágenes

## Estructura del repositorio

├── hardware/
│   ├── soporte_camara/
│   │   ├── imagenes/
│   │   ├── modelo_3D/
│   │   └── instrucciones.md
├── software/
│   ├── clasificador_naranjas_basico/         # Programa básico (v1)
│   ├── clasificador_naranjas_YOLOv10/        # Programa intermedio con YOLOv10 (v2)
│   ├── clasificador_citricos_mobileNetV2/    # Programa avanzado con MobileNetV2 (v3)
├── docs/
│   └── comparativa.rst   

## Soporte para cámara - Sistema de clasificación de frutas

Este soporte fue diseñado específicamente para mantener la cámara a una distancia y ángulo óptimos para la detección y clasificación de frutas cítricas.

### Especificaciones

- **Altura**: 24 cm desde la superficie de análisis

### Instrucciones de montaje

1. Imprimir las piezas con los archivos STL proporcionados
2. Ensamblar la base y el brazo usando presion, sin usar tornilleria
3. Ajustar la cámara en el soporte superior
4. Conectar la iluminación 

### Cámara
- Cámara web con resolución mínima de 1080p
- [Soporte personalizado para cámara](hardware/soporte_camara/instrucciones.md)
- Iluminación LED difusa (5000K)


## Software

Este repositorio contiene tres implementaciones que representan la evolución del sistema:

1. **Sistema Básico (v1)**: Procesamiento de imagen tradicional para clasificar frutas.

2. **Sistema Intermedio con YOLOv10 (v2)**: Enfoque híbrido que combina detección de objetos mediante YOLOv10 con análisis de defectos.

3. **Sistema Avanzado con MobileNetV2 (v3)**: Implementación más sofisticada que utiliza MobileNetV2 y TensorFlow para conseguir mayor precisión y flexibilidad en la detección y clasificación.

Cada versión mejora el rendimiento y precisión, siendo la v3 la más efectiva y recomendada.
   
### Contenido
- `main.py`: Programa
- `capturas/`: Imágenes capturadas durante las pruebas
- `informe.csv`: Datos de clasificación recogidos

### Requisitos
- Python 3.8+
- TensorFlow 2.8+
- OpenCV 4.5+

## Licencia

Este proyecto está licenciado bajo [MIT License] - ver archivo LICENSE para detalles.

## Cita

Si utilizas este código en tu investigación, por favor cítalo como:

```
Ruiz Verdú, Joan (2025). Sistema de Clasificación de Frutas mediante Visión Artificial. 
Universidad Carlemany. https://www.universitatcarlemany.com
```
