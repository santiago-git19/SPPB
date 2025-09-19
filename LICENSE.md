# MIT License

## Gait Analysis System - Completo (Client, Server, Automatización)

Este sistema completo de análisis de marcha incluye tres componentes principales:
- **Client**: Sistema de captura con cámaras Orbbec
- **Server**: Servidor de procesamiento con múltiples detectores de pose
- **Código/Automatización**: Algoritmos de análisis y clasificación

El proyecto utiliza componentes de varios proyectos de código abierto. A continuación se detallan las licencias aplicables:

---

## TRT_Pose License

**Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.**

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR
A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Código del Proyecto - Gait Analysis System (Client + Server + Automatización)

**Copyright (c) 2024-2025, Gait Analysis Research Team. All rights reserved.**

Este proyecto está licenciado bajo la Licencia MIT para todo el código desarrollado específicamente para este sistema de análisis de marcha, incluyendo:

- **Cliente de captura** (cámaras Orbbec, grabación sincronizada, interfaz web)
- **Servidor de procesamiento** (coordinador de detectores, ensemble, análisis avanzado)
- **Algoritmos de automatización** (clasificación de acciones, tracking 3D, SPPB)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Dependencias de Terceros

Este proyecto utiliza las siguientes librerías y componentes de terceros, cada una con sus respectivas licencias:

### TRT_Pose
- **Repositorio**: https://github.com/NVIDIA-AI-IOT/trt_pose
- **Licencia**: MIT License (NVIDIA Corporation)
- **Uso**: Detección de poses humanas optimizada para TensorRT (Server y Automatización)

### MMPose
- **Repositorio**: https://github.com/open-mmlab/mmpose
- **Licencia**: Apache License 2.0
- **Uso**: Framework de detección de poses (ViTPose, HRNet, CSP, MSPN) (Server)

### PyTorch
- **Repositorio**: https://github.com/pytorch/pytorch
- **Licencia**: BSD-style license
- **Uso**: Framework de deep learning (Server y Automatización)

### OpenCV
- **Repositorio**: https://github.com/opencv/opencv
- **Licencia**: BSD 3-Clause License
- **Uso**: Procesamiento de imágenes y visión por computadora (todos los componentes)

### NumPy
- **Repositorio**: https://github.com/numpy/numpy
- **Licencia**: BSD License
- **Uso**: Computación científica con arrays (todos los componentes)

### Flask
- **Repositorio**: https://github.com/pallets/flask
- **Licencia**: BSD 3-Clause License
- **Uso**: Framework web para APIs (Client y Server)

### Orbbec SDK (pyorbbecsdk)
- **Repositorio**: https://github.com/orbbec/pyorbbecsdk
- **Licencia**: MIT License
- **Uso**: Control de cámaras Orbbec Gemini 335Le (Client)

---

## Atribuciones Requeridas

Si usas este código en tu proyecto, por favor incluye:

1. **Para TRT_Pose**: Las atribuciones requeridas por NVIDIA según su licencia MIT
2. **Para MMPose**: Las atribuciones requeridas por OpenMMLab según su licencia Apache 2.0
3. **Para Orbbec SDK**: Las atribuciones requeridas según su licencia MIT
4. **Para este proyecto**: Referencia al proyecto completo de análisis de marcha (Client + Server + Automatización)
5. **Para otras dependencias**: Las atribuciones respectivas según sus licencias (PyTorch, OpenCV, Flask, etc.)

---

## Uso Comercial y Académico

Este sistema completo de análisis de marcha puede ser utilizado tanto para fines comerciales como académicos, siempre y cuando se respeten las condiciones de todas las licencias aplicables (MIT, BSD, Apache 2.0) y se mantengan las atribuciones de copyright correspondientes.

### Componentes del Sistema:
- **Client**: Captura sincronizada con múltiples cámaras Orbbec
- **Server**: Procesamiento distribuido con ensemble de detectores
- **Automatización**: Análisis SPPB y clasificación de acciones

---

## Descargo de Responsabilidad

Este software se proporciona "tal como está", sin garantía de ningún tipo. Los autores no se hacen responsables por daños derivados del uso de este software.

Para uso en aplicaciones médicas o de salud, se recomienda una validación adicional y el cumplimiento de las regulaciones aplicables.

---

**Fecha de última actualización**: Septiembre 2025
