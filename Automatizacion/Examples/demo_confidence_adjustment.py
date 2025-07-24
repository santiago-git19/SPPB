#!/usr/bin/env python3
"""
Prueba del Ajuste de Confianza en process_keypoints
==================================================

Script para probar la nueva funcionalidad de ajuste de confianza
integrada en el método process_keypoints de TRTPoseClassifier.

Autor: Sistema de IA
Fecha: 2025
"""

import numpy as np
import sys
from pathlib import Path

# Añadir ruta del módulo
sys.path.append(str(Path(__file__).resolve().parent.parent))

def demo_confidence_adjustment():
    """
    Demostración de cómo funciona el ajuste de confianza en process_keypoints
    """
    print("🎯 Demostración: Ajuste de Confianza en process_keypoints")
    print("=" * 60)
    
    # Crear keypoints de ejemplo con diferentes niveles de confianza
    test_keypoints = [
        # Caso 1: Keypoints con confianzas variadas
        np.array([
            [100, 200, 0.15],  # Muy baja confianza - será descartado
            [150, 180, 0.25],  # Baja confianza - será descartado
            [200, 160, 0.35],  # Confianza baja-media - será ajustado
            [250, 140, 0.5],   # Confianza media - será ajustado
            [300, 120, 0.7],   # Confianza alta - será ajustado
            [350, 100, 0.9],   # Muy alta confianza - será ajustado ligeramente
            [400, 80, 0.4],    # Confianza media-baja - será ajustado
            [450, 60, 0.6],    # Confianza media-alta - será ajustado
            [500, 40, 0.8],    # Confianza alta - será ajustado
            [0, 0, 0.0],       # Keypoint inválido - será descartado
        ]),
        
        # Caso 2: Todos los keypoints válidos
        np.array([
            [50, 100, 0.4],
            [100, 150, 0.6],
            [150, 200, 0.8],
            [200, 250, 0.95],
            [250, 300, 0.5],
        ]),
        
        # Caso 3: Mezcla con algunos inválidos
        np.array([
            [75, 125, 0.2],    # Será descartado
            [125, 175, 0.45],  # Será ajustado
            [175, 225, 0.1],   # Será descartado
            [225, 275, 0.85],  # Será ajustado
            [275, 325, 0.3],   # En el límite - será ajustado
        ])
    ]
    
    # Simular el comportamiento del clasificador sin crear instancia completa
    confidence_threshold = 0.3
    
    for i, keypoints in enumerate(test_keypoints, 1):
        print(f"\n📋 Caso {i}: Keypoints de prueba")
        print("-" * 40)
        
        print(f"📊 Keypoints originales ({len(keypoints)}):")
        for j, kp in enumerate(keypoints):
            status = "❌ Descartado" if kp[2] < confidence_threshold else "✅ Válido"
            print(f"   [{j:2d}] x={kp[0]:3.0f}, y={kp[1]:3.0f}, conf={kp[2]:.2f} - {status}")
        
        # Simular el procesamiento completo que hace _adjust_keypoint_confidence
        result = simulate_full_processing(keypoints, confidence_threshold)
        
        print(f"\n📈 Resultado después del ajuste:")
        print(f"   🔧 Keypoints procesados: {len(result)} (formato convertido)")
        
        for j, kp in enumerate(result):
            if kp[2] > 0:  # Solo mostrar keypoints no descartados
                print(f"   [{j:2d}] x={kp[0]:6.3f}, y={kp[1]:6.3f}, conf={kp[2]:.3f}")
            else:
                print(f"   [{j:2d}] x={kp[0]:6.3f}, y={kp[1]:6.3f}, conf={kp[2]:.3f} (descartado)")
        
        # Estadísticas del caso
        original_valid = np.sum(keypoints[:, 2] >= confidence_threshold)
        processed_valid = np.sum(result[:, 2] > 0)
        
        print(f"\n📊 Estadísticas del caso {i}:")
        print(f"   📥 Keypoints originales: {len(keypoints)}")
        print(f"   ✅ Válidos originalmente: {original_valid}")
        print(f"   📤 Keypoints procesados: {len(result)}")
        print(f"   🎯 Válidos tras procesamiento: {processed_valid}")
        
        if original_valid > 0:
            original_range = keypoints[keypoints[:, 2] >= confidence_threshold, 2]
            processed_range = result[result[:, 2] > 0, 2]
            if len(processed_range) > 0:
                print(f"   📈 Confianza original: [{np.min(original_range):.3f}, {np.max(original_range):.3f}]")
                print(f"   📈 Confianza ajustada: [{np.min(processed_range):.3f}, {np.max(processed_range):.3f}]")

def simulate_full_processing(keypoints: np.ndarray, 
                           threshold: float,
                           boost_factor: float = 2.0,
                           max_confidence: float = 0.95) -> np.ndarray:
    """
    Simula el procesamiento completo que hace _adjust_keypoint_confidence
    """
    # Paso 1: Filtrar keypoints con baja confianza
    filtered = keypoints.copy()
    low_confidence_mask = keypoints[:, 2] < threshold
    filtered[low_confidence_mask] = [0.0, 0.0, 0.0]
    
    # Paso 2: Ajustar confianza de keypoints válidos
    adjusted = filtered.copy()
    valid_mask = (filtered[:, 2] >= threshold)
    
    if np.any(valid_mask):
        valid_confidences = filtered[valid_mask, 2]
        
        # Fórmula de ajuste suave usando función sigmoide
        normalized_conf = (valid_confidences - threshold) / (1.0 - threshold)
        sigmoid_steepness = 3.0
        sigmoid_scale = 1.0 / (1.0 + np.exp(-sigmoid_steepness * (normalized_conf - 0.5)))
        
        confidence_increment = (normalized_conf * boost_factor * sigmoid_scale)
        new_confidences = threshold + confidence_increment * (1.0 - threshold)
        new_confidences = np.minimum(new_confidences, max_confidence)
        
        adjusted[valid_mask, 2] = new_confidences
    
    # Paso 3: Simulación de conversión de formato (mantener como está para demo)
    converted = adjusted
    
    # Paso 4: Simulación de normalización básica
    normalized = converted.copy()
    valid_mask = (converted[:, 0] != 0) | (converted[:, 1] != 0)
    
    if np.any(valid_mask):
        valid_keypoints = converted[valid_mask]
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

def show_formula_explanation():
    """
    Explica la fórmula de ajuste de confianza utilizada
    """
    print("\n🔬 Explicación de la Fórmula de Ajuste")
    print("=" * 50)
    
    print("📐 Fórmula utilizada:")
    print("   1. normalized_conf = (original_conf - threshold) / (1.0 - threshold)")
    print("   2. sigmoid_scale = 1 / (1 + exp(-3.0 * (normalized_conf - 0.5)))")
    print("   3. increment = normalized_conf * boost_factor * sigmoid_scale")
    print("   4. new_conf = threshold + increment * (1.0 - threshold)")
    print("   5. final_conf = min(new_conf, max_confidence)")
    
    print("\n🎯 Parámetros:")
    print("   • threshold: 0.3 (mínimo para ser válido)")
    print("   • boost_factor: 2.0 (multiplicador de incremento)")
    print("   • max_confidence: 0.95 (límite superior)")
    print("   • sigmoid_steepness: 3.0 (suavidad de la transición)")
    
    print("\n💡 Características:")
    print("   ✅ Incremento proporcional (no lineal)")
    print("   ✅ Transición suave (función sigmoide)")
    print("   ✅ Límite máximo para evitar sobreajuste")
    print("   ✅ Reutiliza funciones existentes del clasificador")

if __name__ == "__main__":
    # Ejecutar demostración
    demo_confidence_adjustment()
    show_formula_explanation()
    
    print(f"\n✅ Demostración completada")
    print(f"💡 La funcionalidad está integrada en process_keypoints()")
    print(f"🔧 Reutiliza las funciones existentes:")
    print(f"   - _filter_low_confidence_keypoints()")
    print(f"   - _convert_keypoints_format()")
    print(f"   - _normalize_keypoints()")
