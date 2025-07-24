#!/usr/bin/env python3
"""
Prueba del Flujo Paso a Paso de Ajuste de Confianza
===================================================

Script para probar el nuevo flujo donde:
1. _filter_low_confidence_keypoints() filtra keypoints < threshold
2. _adjust_keypoint_confidence() ajusta solo la confianza 
3. _convert_keypoints_format() convierte al formato del modelo
4. _normalize_keypoints() normaliza las coordenadas

Autor: Sistema de IA
Fecha: 2025
"""

import numpy as np
import sys
from pathlib import Path

# Añadir ruta del módulo
sys.path.append(str(Path(__file__).resolve().parent.parent))

def demo_step_by_step_processing():
    """
    Demostración del procesamiento paso a paso de keypoints
    """
    print("🔧 Demostración: Flujo Paso a Paso de Procesamiento")
    print("=" * 60)
    
    # Crear keypoints de ejemplo con diferentes niveles de confianza
    original_keypoints = np.array([
        [100, 200, 0.15],  # Muy baja confianza - será filtrado
        [150, 180, 0.25],  # Baja confianza - será filtrado
        [200, 160, 0.35],  # Confianza baja-media - será ajustado
        [250, 140, 0.5],   # Confianza media - será ajustado
        [300, 120, 0.7],   # Confianza alta - será ajustado
        [350, 100, 0.9],   # Muy alta confianza - será ajustado ligeramente
        [400, 80, 0.4],    # Confianza media-baja - será ajustado
        [450, 60, 0.6],    # Confianza media-alta - será ajustado
        [500, 40, 0.8],    # Confianza alta - será ajustado
        [0, 0, 0.0],       # Keypoint inválido - será filtrado
    ])
    
    confidence_threshold = 0.3
    
    print(f"📊 Keypoints originales ({len(original_keypoints)}):")
    for i, kp in enumerate(original_keypoints):
        status = "❌ Será filtrado" if kp[2] < confidence_threshold else "✅ Será procesado"
        print(f"   [{i:2d}] x={kp[0]:3.0f}, y={kp[1]:3.0f}, conf={kp[2]:.2f} - {status}")
    
    print(f"\n🔄 Procesamiento paso a paso:")
    print("-" * 40)
    
    # PASO 1: Filtrar keypoints de baja confianza
    print("1️⃣ Filtrado de keypoints con baja confianza...")
    filtered_keypoints = simulate_filter_low_confidence(original_keypoints, confidence_threshold)
    
    valid_count = np.sum(filtered_keypoints[:, 2] >= confidence_threshold)
    filtered_count = np.sum(filtered_keypoints[:, 2] == 0.0)
    print(f"   ✅ Keypoints válidos: {valid_count}")
    print(f"   ❌ Keypoints filtrados: {filtered_count}")
    
    # PASO 2: Ajustar confianza de keypoints válidos
    print("\n2️⃣ Ajuste de confianza de keypoints válidos...")
    adjusted_keypoints = simulate_adjust_confidence(filtered_keypoints, confidence_threshold)
    
    valid_mask = adjusted_keypoints[:, 2] >= confidence_threshold
    if np.any(valid_mask):
        original_valid_conf = filtered_keypoints[valid_mask, 2]
        adjusted_valid_conf = adjusted_keypoints[valid_mask, 2]
        avg_increase = np.mean(adjusted_valid_conf - original_valid_conf)
        print(f"   📈 Confianza promedio aumentada en: +{avg_increase:.3f}")
        print(f"   📊 Rango antes: [{np.min(original_valid_conf):.3f}, {np.max(original_valid_conf):.3f}]")
        print(f"   📊 Rango después: [{np.min(adjusted_valid_conf):.3f}, {np.max(adjusted_valid_conf):.3f}]")
    
    # PASO 3: Conversión de formato (simulado como identidad)
    print("\n3️⃣ Conversión al formato del modelo...")
    converted_keypoints = adjusted_keypoints  # Simulación - en real sería _convert_keypoints_format()
    print(f"   🔧 Formato convertido: {converted_keypoints.shape}")
    
    # PASO 4: Normalización de coordenadas
    print("\n4️⃣ Normalización de coordenadas...")
    normalized_keypoints = simulate_normalize_keypoints(converted_keypoints)
    
    valid_mask = normalized_keypoints[:, 2] > 0
    if np.any(valid_mask):
        norm_coords = normalized_keypoints[valid_mask]
        print(f"   📐 Coordenadas normalizadas: {np.sum(valid_mask)} keypoints")
        print(f"   📊 Rango X: [{np.min(norm_coords[:, 0]):.3f}, {np.max(norm_coords[:, 0]):.3f}]")
        print(f"   📊 Rango Y: [{np.min(norm_coords[:, 1]):.3f}, {np.max(norm_coords[:, 1]):.3f}]")
    
    # Mostrar resultado final
    print(f"\n📈 Resultado final:")
    print(f"   📥 Keypoints originales: {len(original_keypoints)}")
    print(f"   📤 Keypoints procesados: {len(normalized_keypoints)}")
    print(f"   ✅ Keypoints válidos finales: {np.sum(normalized_keypoints[:, 2] > 0)}")
    
    # Comparación detallada
    print(f"\n📋 Comparación detallada (solo keypoints válidos):")
    print("   Idx | Original           | Ajustado           | Normalizado")
    print("   " + "-" * 65)
    
    for i, (orig, adj, norm) in enumerate(zip(original_keypoints, adjusted_keypoints, normalized_keypoints)):
        if orig[2] >= confidence_threshold:  # Solo mostrar los que eran válidos originalmente
            print(f"   {i:2d}  | {orig[0]:3.0f},{orig[1]:3.0f},{orig[2]:.2f} | "
                  f"{adj[0]:3.0f},{adj[1]:3.0f},{adj[2]:.3f} | "
                  f"{norm[0]:6.3f},{norm[1]:6.3f},{norm[2]:.3f}")

def simulate_filter_low_confidence(keypoints: np.ndarray, threshold: float) -> np.ndarray:
    """Simula _filter_low_confidence_keypoints"""
    filtered = keypoints.copy()
    low_confidence_mask = keypoints[:, 2] < threshold
    filtered[low_confidence_mask] = [0.0, 0.0, 0.0]
    return filtered

def simulate_adjust_confidence(keypoints: np.ndarray, 
                             threshold: float,
                             boost_factor: float = 2.0,
                             max_confidence: float = 0.95) -> np.ndarray:
    """Simula _adjust_keypoint_confidence"""
    adjusted = keypoints.copy()
    valid_mask = (keypoints[:, 2] >= threshold)
    
    if np.any(valid_mask):
        valid_confidences = keypoints[valid_mask, 2]
        
        # Fórmula de ajuste suave usando función sigmoide
        normalized_conf = (valid_confidences - threshold) / (1.0 - threshold)
        sigmoid_steepness = 3.0
        sigmoid_scale = 1.0 / (1.0 + np.exp(-sigmoid_steepness * (normalized_conf - 0.5)))
        
        confidence_increment = (normalized_conf * boost_factor * sigmoid_scale)
        new_confidences = threshold + confidence_increment * (1.0 - threshold)
        new_confidences = np.minimum(new_confidences, max_confidence)
        
        adjusted[valid_mask, 2] = new_confidences
    
    return adjusted

def simulate_normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Simula _normalize_keypoints"""
    normalized = keypoints.copy()
    valid_mask = (keypoints[:, 0] != 0) | (keypoints[:, 1] != 0)
    
    if np.any(valid_mask):
        valid_keypoints = keypoints[valid_mask]
        if len(valid_keypoints) > 0:
            x_coords = valid_keypoints[:, 0]
            y_coords = valid_keypoints[:, 1]
            
            # Normalización básica
            if np.max(x_coords) > np.min(x_coords):
                x_center = np.mean(x_coords)
                x_scale = np.max(x_coords) - np.min(x_coords)
                normalized[valid_mask, 0] = (x_coords - x_center) / x_scale
            
            if np.max(y_coords) > np.min(y_coords):
                y_center = np.mean(y_coords)
                y_scale = np.max(y_coords) - np.min(y_coords)
                normalized[valid_mask, 1] = (y_coords - y_center) / y_scale
    
    return normalized

def show_function_responsibilities():
    """
    Muestra las responsabilidades de cada función en el flujo
    """
    print("\n🎯 Responsabilidades de Cada Función")
    print("=" * 50)
    
    functions = [
        {
            'name': '_filter_low_confidence_keypoints()',
            'input': 'Array [N, 3] con keypoints originales',
            'process': 'Marca como (0,0,0) keypoints con conf < threshold',
            'output': 'Array [N, 3] con keypoints filtrados',
            'responsibility': 'Solo filtrado por confianza'
        },
        {
            'name': '_adjust_keypoint_confidence()',
            'input': 'Array [N, 3] YA filtrado',
            'process': 'Ajusta confianza de keypoints válidos con sigmoide',
            'output': 'Array [N, 3] con confianzas ajustadas',
            'responsibility': 'Solo ajuste de confianza'
        },
        {
            'name': '_convert_keypoints_format()',
            'input': 'Array [N, 3] con confianzas ajustadas',
            'process': 'Convierte formato (COCO→NVIDIA, OpenPose→NVIDIA, etc.)',
            'output': 'Array [M, 3] en formato del modelo',
            'responsibility': 'Solo conversión de formato'
        },
        {
            'name': '_normalize_keypoints()',
            'input': 'Array [M, 3] en formato del modelo',
            'process': 'Centra y escala coordenadas x,y',
            'output': 'Array [M, 3] normalizado',
            'responsibility': 'Solo normalización de coordenadas'
        }
    ]
    
    for i, func in enumerate(functions, 1):
        print(f"\n{i}️⃣ {func['name']}")
        print(f"   📥 Entrada: {func['input']}")
        print(f"   🔧 Proceso: {func['process']}")
        print(f"   📤 Salida: {func['output']}")
        print(f"   🎯 Responsabilidad: {func['responsibility']}")

if __name__ == "__main__":
    # Ejecutar demostración
    demo_step_by_step_processing()
    show_function_responsibilities()
    
    print("\n✅ Demostración completada")
    print("💡 Cada función tiene una responsabilidad específica:")
    print("   🔍 Filtrado → 🎯 Ajuste → 🔄 Conversión → 📐 Normalización")
    print("🔧 El flujo está integrado en process_keypoints()")
