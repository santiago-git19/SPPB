from sppb_test import SPPBTest
from utils.config import Config

if __name__ == "__main__":
    # === CONFIGURACIÓN INICIAL ===
    config = Config()
    
    print("🏥 Sistema SPPB con TRT Pose")
    print("=" * 50)
    print("📊 Configuración:")
    print(f"   🔧 Modelo TRT Pose: {config.trt_pose_model}")
    print(f"   🎭 Clasificador: {config.pose_classifier_model}")
    print(f"   📹 Cámara frontal: {config.frontal_camera}")
    print(f"   📹 Cámara lateral: {config.lateral_camera}")
    print(f"   📏 Ratio píxel/metro: {config.pixel_to_meter_ratio}")
    print("=" * 50)
    
    # === INICIALIZACIÓN DEL TEST ===
    try:
        sppb = SPPBTest(config)
        
        # === EJECUCIÓN DEL TEST ===
        # Opción 1: Usar video pregrabado
        # result = sppb.run(video_path="WIN_20250702_12_09_08_Pro.mp4")
        
        # Opción 2: Usar cámaras en vivo
        result = sppb.run(camera_id=config.frontal_camera)
        
        # === MOSTRAR RESULTADOS ===
        print("\n🏆 RESULTADOS DEL TEST SPPB")
        print("=" * 50)
        print(result.to_dict())
        
    except Exception as e:
        print(f"\n❌ Error ejecutando el test SPPB: {e}")
        print("\n💡 Consejos para solucionar problemas:")
        print("   1. Verificar que los modelos estén en las rutas correctas")
        print("   2. Comprobar que las cámaras estén conectadas")
        print("   3. Calibrar el pixel_to_meter_ratio según su configuración")
        print("   4. Revisar los logs para más detalles")
