# Soporte para Cámara - Sistema de Clasificación de Cítricos

Este soporte ha sido diseñado específicamente para mantener la cámara a una distancia y ángulo óptimos para la detección y clasificación de cítricos, con un sistema de unión a presión que facilita el montaje sin necesidad de tornillos.

## Especificaciones Técnicas

- **Altura ajustable**: 25-35 cm desde la superficie de análisis
- **Material**: PLA (ácido poliláctico) -Preferencia de uso Biodegradable-
- **Relleno recomendado**: 25-30% para garantizar la resistencia de las uniones a presión
- **Dimensiones totales**: 30 × 20 × 40 cm (ancho × profundo × alto)
- **Sistema de unión**: Conectores a presión tipo macho-hembra

## Contenido de la Carpeta `modelo_3D`

- `Inferior.stl`: Base estable con espacio para la muestra
- `Pieza.stl`: Brazo de altura
- `Superior.stl`: Soporte específico para webcam estándar
- `Union.stl`: Soporte para unir la piezas a presión
- `Soporte_Camara.stl`: Modelo completo ensamblado (referencia)

## Instrucciones de Impresión 3D

1. **Configuración de impresión crítica para uniones a presión**:
   - Altura de capa: 0.16-0.2 mm (más fino para mejor detalle en los conectores)
   - Temperatura de impresión: 205-215°C (ligeramente más alta para mejor adhesión entre capas)
   - Temperatura de cama: 60°C
   - Velocidad de impresión: 40-50 mm/s (más lento para los conectores)
   - Perímetros: Mínimo 3 para mayor resistencia
   - Tolerancia: Los conectores están diseñados con 0.2mm de tolerancia

2. **Orientación de impresión**:
   - Evitar imprimir los conectores en posición horizontal para mayor resistencia, a no ser que se utilicen soportes.
   - Base en posicion superficial, para garantizar capa inferior y superior robusta.


3. **Tiempo estimado de impresión**:
   - Inferior: 2.5 - 3 hora
   - Pieza: 1.5 - 2 horas
   - Superior: 0.75 horas
   - Union: 1-2 horas

## Instrucciones de Montaje (Sistema a Presión)

1. **Preparación de piezas**:
   - Limpiar cuidadosamente todos los conectores de residuos de impresión
   - Verificar que no haya hilos o rebabas que impidan el encaje

2. **Secuencia de montaje**:
   - Alinear el conector macho de la base con el conector hembra del brazo ajustable
   - Presionar firmemente hasta escuchar un "clic" que indica el encaje completo
   - Repetir el proceso para unir el soporte de cámara al extremo superior del brazo

3. **Ajuste de altura**:
   - El brazo cuenta con un mecanismo de trinquete que permite ajustar la altura
   - Para ajustar, presionar la pestaña de liberación y deslizar hasta la posición deseada

4. **Verificación**:
   - Comprobar que todas las uniones estén completamente encajadas
   - Verificar la estabilidad del conjunto antes de colocar la cámara

## Solución de Problemas de Encaje

- **Si las piezas no encajan**: Lijar suavemente los conectores macho con una lima fina
- **Si las uniones quedan flojas**: Aplicar una pequeña cantidad de cinta adhesiva en el conector macho
- **Para uniones permanentes**: Se puede aplicar una pequeña cantidad de pegamento para PLA en los conectores

## Notas sobre el Diseño a Presión

Este diseño utiliza un sistema de unión a presión cuidadosamente calibrado para proporcionar suficiente resistencia mientras permite el desmontaje cuando sea necesario. Los conectores incluyen pequeños chaflanes para facilitar la alineación inicial y pestañas de retención para asegurar la unión.

Los archivos STL se proporcionan bajo licencia MIT, igual que el resto del proyecto.

https://www.tinkercad.com/things/7zP5vdpdpGA-soportecamara