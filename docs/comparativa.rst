==============================================
Comparativa de Sistemas de Clasificación
==============================================

Introducción
------------

Este documento presenta una comparativa detallada entre las tres versiones del sistema de clasificación de frutas desarrolladas para este proyecto. 
Cada versión representa una evolución en términos de técnicas empleadas y capacidades.

Características principales
--------------------------

+-------------------------+-------------------+---------------------+----------------------+
| Característica          | Básico (v1)       | YOLOv10 (v2)        | MobileNetV2 (v3)     |
+=========================+===================+=====================+======================+
| Enfoque técnico         | Procesamiento     | Híbrido: YOLO para  | Red neuronal         |
|                         | tradicional con   | detección + análisis| convolucional con    |
|                         | umbrales HSV      | de imagen           | MobileNetV2          |
+-------------------------+-------------------+---------------------+----------------------+
| Precisión detección     | 85-90%            | 75-85%              | 90-95%               |
+-------------------------+-------------------+---------------------+----------------------+
| Velocidad (FPS)         | 30-60 FPS         | 40 FPS              | 40 FPS               |
+-------------------------+-------------------+---------------------+----------------------+
| Uso de memoria          | Bajo (~200MB)     | Medio (~1GB)        | Medio (~800MB)       |
+-------------------------+-------------------+---------------------+----------------------+
| Requerimientos HW       | CPU estándar      | GPU recomendada     | CPU moderna o GPU    |
+-------------------------+-------------------+---------------------+----------------------+
| Tipos de frutas         | Solo naranjas     | Solo naranjas       | Naranjas y limones   |
+-------------------------+-------------------+---------------------+----------------------+
| Tipos de defectos       | Manchas, golpes,  | Manchas y golpes    | Manchas, golpes,     |
|                         | hongos y otros    |                     | hongos y otros       |
+-------------------------+-------------------+---------------------+----------------------+
| Guardado de datos       | CSV               | CSV                 | CSV                  |
|                         |                   |                     |                      |
+-------------------------+-------------------+---------------------+----------------------+

Detalles técnicos
-----------------

Sistema Básico (v1)
~~~~~~~~~~~~~~~~~~~

* **Tecnología**: OpenCV con procesamiento clásico de imagen
* **Método**: Segmentación por color (HSV), detección de contornos y análisis morfológico
* **Fortalezas**: Simplicidad, bajo consumo de recursos, rápida ejecución
* **Debilidades**: Solo funciona con naranjas, sensible a cambios de iluminación
* **Ajustes**: Requiere calibración manual de umbrales HSV

Sistema Intermedio con YOLOv10 (v2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Tecnología**: YOLO (You Only Look Once) v10 para detección de objetos
* **Método**: Detección de frutas con YOLO + análisis posterior de defectos
* **Fortalezas**: Mayor precisión, menos sensible a iluminación, detecta múltiples frutas
* **Debilidades**: Requiere GPU para buen rendimiento, entrenamiento complejo
* **Ajustes**: Puntuaciones de confianza y non-maximum suppression configurables

Sistema Avanzado con MobileNetV2 (v3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Tecnología**: TensorFlow con arquitectura MobileNetV2 (Tecnología de visión artificial optimizada de Google)
* **Método**: Red neuronal convolucional con transfer learning
* **Fortalezas**: Alta precisión, buena velocidad, clasificación multiclase avanzada
* **Debilidades**: Mayor complejidad de implementación
* **Ajustes**: Modelo optimizado para equilibrar rendimiento y precisión

Conclusiones
-----------

Programa sólo con OpenCV (v1):
El programa sin detección de fruta utiliza solo OpenCV para la detección de cítricos, y la detección de fruta emplea segmentación en espacio de color HSV, 
detección de contornos y análisis geométrico (circularidad y área). En un entorno controlado, la detección de cítricos es eficaz; sin embargo, este método 
no funciona bien con condiciones de iluminación variables, fondos desordenados u otros objetos de color y forma similares a los cítricos, por lo que tiene 
limitaciones. La detección de defectos, como la mancha negra y la podredumbre verde, también está limitada debido al ajuste manual de la gama de colores y 
los parámetros de segmentación, lo que disminuye la robustez del sistema, por lo que no es adecuado para todas las aplicaciones. Aunque el programa con 
OpenCV es adecuado para aplicaciones básicas y controladas, no es generalizable, ya que seguimos pudiendo tener errores en la detención de fruta solo por 
color y no en forma. Lo que conduce al desarrollo de métodos más avanzados, por lo que es necesario seguir investigando.

Programa con detección de frutas utilizando YOLOv10 (v2):
El segundo programa utiliza YOLOv10 para detectar cítricos, y está integrado con OpenCV para la detección de defectos, por lo que se trata de un enfoque 
más avanzado. Los resultados son significativamente mejores que los del primer programa en términos de precisión y robustez, ya que YOLOv10 es capaz de 
detectar cítricos en entornos más complejos, como con variaciones de iluminación y fondos desordenados. La clasificación automática de las frutas 
(es decir, naranjas y limones) reduce la necesidad de realizar ajustes manuales y hace que el programa sea más eficiente, por lo que supone una mejora 
significativa. Sin embargo, la implementación de YOLOv10 requiere un conjunto de datos etiquetados para entrenar el modelo, lo que implica una gran 
preparación de los datos, y además, YOLOv10 está limitado en la detección de defectos específicos, ya que se utiliza principalmente para la detección 
de objetos, no para el análisis detallado de características, por lo que tiene sus limitaciones. La detección de defectos aún tiene que hacerse utilizando 
OpenCV, lo que hace la integración del sistema algo más compleja, por lo que requiere más conocimientos técnicos. En general, este enfoque es muy sólido 
y preciso para detectar cítricos, pero su aplicación y mantenimiento requieren recursos técnicos y computacionales más avanzados, por lo que no es adecuado para todas las aplicaciones.

Programa con OpenCV y MobileNetV2 (v3):
El tercer programa utiliza OpenCV junto con MobileNetV2, que es una red neuronal convolucional ligera, y el programa utiliza OpenCV para el procesamiento 
inicial de la imagen y la segmentación de los cítricos, mientras que MobileNetV2 se utiliza para la identificación y clasificación, por lo que se trata 
de un enfoque híbrido. Los resultados son muy satisfactorios, ya que MobileNetV2 es capaz de detectar los cítricos con una precisión similar a la de YOLOv10 
pero con un menor coste computacional, lo que la hace adecuada para aplicaciones en tiempo real o con recursos limitados, por lo que es una solución más 
eficiente. OpenCV se empareja con el modelo MobileNetV2 para detectar defectos como puntos negros, podredumbre verde y otros problemas de calidad, y este 
método es eficiente y rápido, ofreciendo varias ventajas sobre los otros enfoques, porque es un sistema modular y flexible. Una de las principales ventajas 
de OpenCV+MobileNetV2 es su adaptabilidad a distintos escenarios sin necesidad de una formación exhaustiva y, además, MobileNetV2 se basa en un modelo 
pre-entrenado, que puede ajustarse con un esfuerzo mínimo para adaptarse a escenarios específicos, por lo que es una solución más adaptable. Sin embargo, 
el enfoque OpenCV+MobileNetV2 también tiene algunas limitaciones, porque es necesario ajustar las gamas de colores y los parámetros de segmentación en 
OpenCV para una detección óptima en condiciones variables, y a pesar de ello, el programa OpenCV+MobileNetV2 resulta ser una solución eficaz y versátil 
para aplicaciones prácticas de análisis de cítricos, por lo que constituye un buen resultado.

Conclusiones de resultados:
En conclusión, los tres enfoques ofrecen ventajas y limitaciones variables en función del contexto de aplicación, ya que cada enfoque tiene sus puntos 
fuertes y débiles, y el programa con OpenCV es rentable y accesible pero limitado en escenarios complejos, mientras que el enfoque YOLOv10 proporciona 
una gran precisión y robustez en la detección de los elementos, pero requiere importantes recursos técnicos y computacionales, por lo que existe un 
compromiso. El programa OpenCV+MobileNetV2 equilibra estos aspectos, ofreciendo una solución eficiente, precisa y adaptable para el análisis de cítricos 
y la detección de defectos, por lo que es una buena opción para muchas aplicaciones.
