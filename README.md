# Análisis Armónico del Sistema Biela-Manivela

## Descripción del Proyecto

Este proyecto consiste en el análisis experimental del sistema biela-manivela mediante Python, utilizando datos obtenidos con sensores Vernier. El objetivo principal es analizar el primer y segundo armónico presente en el movimiento del pistón y validar que el sistema no sigue un Movimiento Armónico Simple (MAS) puro debido a la geometría del mecanismo.

## Objetivos

1. **Analizar el movimiento angular y lineal** del sistema biela-manivela
2. **Aplicar ajuste no lineal** con ecuaciones del MAS a los datos experimentales
3. **Realizar análisis espectral** usando la Transformada Discreta de Fourier (DFT)
4. **Identificar y caracterizar** el primer y segundo armónico
5. **Reconstruir la señal** de aceleración usando componentes armónicos
6. **Interpretar físicamente** los resultados en el contexto de motores de combustión interna

## Estructura de Datos Requerida

Los archivos de datos deben estar en formato `.txt` exportados desde Logger Pro con las siguientes columnas:

- **Tiempo (t)**: instante de cada muestra, en s
- **Posición lineal (x)**: desplazamiento del carrito, en m  
- **Velocidad lineal**: en m/s
- **Aceleración lineal**: en m/s²
- **Ángulo (θ)**: rotación de la manivela, en rad
- **Velocidad angular**: en rad/s
- **Aceleración angular**: en rad/s²

### Formato de archivos esperado:
```
biela-manivela-[frecuenciaHz]-[fecha].txt
```
Ejemplo: `biela-manivela-20Hz-09-09-2025.txt`

## Tareas a Realizar

### 1. Análisis del Movimiento Angular y Lineal

**Gráficas requeridas:**
- Ángulo (rad) vs tiempo
- Velocidad angular y aceleración angular vs tiempo
- Posición, velocidad y aceleración lineal vs tiempo

**Análisis:**
- Determinar si se realizó aproximadamente un MCU (Movimiento Circular Uniforme)
- Ajustar las ecuaciones del MAS:
  - `x(t) = A₀ + A cos(ωt + φ)`
  - `v(t) = -ωA sin(ωt + φ)`  
  - `a(t) = -ω²A cos(ωt + φ)`

### 2. Comparación entre Espacio Angular y Lineal

**Análisis requerido:**
- Graficar posición lineal del pistón vs ángulo de la manivela
- Evaluar si x(θ) se comporta como función sinusoidal
- Identificar asimetrías en el movimiento
- Intentar ajuste con `x(θ) = A₀ + A cos(θ + φ)`
- Si no es posible, realizar ajuste por partes

### 3. Análisis Espectral de la Aceleración (DFT)

**Procedimiento:**
1. **Seleccionar intervalo periódico** de la señal de aceleración
2. **Aplicar DFT** para obtener espectro de amplitudes
3. **Identificar los dos primeros armónicos dominantes**:
   - Frecuencias: f₁, f₂
   - Amplitudes: A₁, A₂  
   - Fases: φ₁, φ₂
4. **Reconstruir la señal** aproximada:
   ```
   a_aprox(t) = A₁ cos(2πf₁t + φ₁) + A₂ cos(2πf₂t + φ₂)
   ```
5. **Comparar** señal original vs reconstruida
6. **Analizar** armónicos adicionales y sus posibles causas

### 4. Interpretación Física

**Discusión requerida:**
- Origen físico del primer armónico (componente MAS)
- Origen físico del segundo armónico (corrección geométrica)
- Relación con motores de combustión interna de 4 cilindros
- Implicaciones del segundo armónico en vibraciones del motor

## Herramientas y Librerías

**Librerías Python recomendadas:**
- `numpy` - cálculos numéricos
- `pandas` - manejo de datos
- `matplotlib` - gráficas
- `scipy` - ajuste no lineal, FFT
- `scipy.optimize.curve_fit` - ajuste de funciones

## Fundamento Teórico Clave

### Ecuaciones del Sistema Biela-Manivela

**Posición del pistón:**
```
x(t) = (l - r²/4l) + r cos θ(t) - (r²/4l) cos 2θ(t)
```

**Aceleración (aproximación de segundo armónico):**
```
a(t) = -rω² cos θ(t) + (r²ω²/l) cos 2θ(t)
a(t) = A₁ cos θ(t) + A₂ cos 2θ(t)
```

Donde:
- r = radio de la manivela
- l = longitud de la biela  
- θ(t) = ωt (movimiento circular uniforme)

### Significado Físico de los Armónicos

- **Primer armónico (A₁)**: Componente principal tipo MAS
- **Segundo armónico (A₂)**: Corrección debida a la longitud finita de la biela

## Entregables

1. **Documento con gráficas y cálculos**
2. **Video de 15 minutos** explicando resultados (todos los integrantes)
3. **Enlace a datos experimentales**
4. **Código Python** utilizado para el análisis

## Evaluación

La calificación se basará en:
- **Introducción** (2 pts): objetivo y conclusión principal
- **Métodos** (3 pts): equipo y procedimiento
- **Resultados** (5 pts): gráficas, tratamiento estadístico
- **Discusión** (10 pts): análisis de movimientos, espectral y armónicos

## Referencias Teóricas

- Movimiento Armónico Simple (MAS)
- Transformada Discreta de Fourier (DFT)
- Análisis de sistemas mecánicos biela-manivela
- Aplicaciones en motores de combustión interna

---

**Nota**: Este análisis permite validar experimentalmente que el movimiento del pistón contiene componentes armónicos adicionales relacionadas con la geometría del sistema, más allá del MAS ideal.# lab01
