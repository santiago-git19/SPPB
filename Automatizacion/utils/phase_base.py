import logging
from typing import Optional, Dict, Any
from Automatizacion.utils.pose_detection.trt_pose_proc import TRTPoseProcessor
from Automatizacion.utils.action_classifier import TRTPoseClassifier, create_pose_classifier

logger = logging.getLogger(__name__)

class FullRestartRequested(Exception):
    """Excepción personalizada para señalizar que se requiere reiniciar el test completo."""
    pass

class PhaseBase:
    """
    Clase base para todas las fases del test SPPB.
    Contiene funcionalidad común como la espera de preparación del usuario.
    Utiliza TRT Pose para detección de keypoints y clasificación de poses.
    """
    
    def __init__(self, trt_pose_processor: TRTPoseProcessor, pose_classifier: TRTPoseClassifier, config):
        """
        Inicializa la fase base con procesadores TRT Pose
        
        Args:
            trt_pose_processor: Instancia de TRTPoseProcessor para detección de keypoints
            pose_classifier: Instancia de TRTPoseClassifier para clasificación de poses
            config: Configuración del sistema
        """
        self.trt_pose_processor = trt_pose_processor
        self.pose_classifier = pose_classifier
        self.config = config
        
        # Configurar intervalo de frames
        self.interval = getattr(config, 'fps', 5.0)
        if self.interval:
            self.interval = 1.0 / self.interval
        else:
            self.interval = 0.2
        
        # Estado global del test para reinicio completo
        self._test_state = {
            'phase_results': [],
            'current_phase': None,
            'global_attempt': 1,
            'max_global_attempts': 3
        }
        
        logger.info(f"✅ PhaseBase inicializada con TRT Pose")
        logger.info(f"   🔧 Procesador: {type(self.trt_pose_processor).__name__}")
        logger.info(f"   🎭 Clasificador: {type(self.pose_classifier).__name__}")


    def reset_pose_processors(self):
        """
        Reinicia el estado de los procesadores de pose
        """
        try:
            # Reiniciar secuencia del clasificador
            if hasattr(self.pose_classifier, 'reset_sequence'):
                self.pose_classifier.reset_sequence()
                logger.info("🔄 Secuencia del clasificador reiniciada")
                
            # Reiniciar estado del procesador si tiene método reset
            if hasattr(self.trt_pose_processor, 'reset'):
                self.trt_pose_processor.reset()
                logger.info("🔄 Procesador TRT Pose reiniciado")
                
        except Exception as e:
            logger.warning(f"⚠️ Error reiniciando procesadores: {e}")

    def wait_for_ready(self, message="Presione ENTER cuando esté listo para comenzar..."):
        """
        Espera a que el usuario presione ENTER para indicar que está listo para comenzar la prueba.
        
        Args:
            message (str): Mensaje que se mostrará al usuario
        """
        print("\n" + message)
        input("")  # Espera a que se presione ENTER
        print("¡Comenzando!")

    def print_instructions(self, title, instructions):
        """
        Imprime las instrucciones de una fase de manera consistente.
        
        Args:
            title (str): Título de la fase
            instructions (list): Lista de instrucciones
        """
        print(f"\n=== {title} ===")
        if instructions:
            print("Instrucciones:")
            for instruction in instructions:
                print(f"- {instruction}")
    
    def reset_test(self):
        """
        Reinicia el estado de la prueba para permitir volver a ejecutarla desde el principio.
        Las subclases deben sobrescribir este método para reiniciar su estado específico.
        """
        print("\n⚠️  Reiniciando la prueba...")
        
        # Reiniciar estado básico común
        if hasattr(self, '_last'):
            delattr(self, '_last')
            
        # Reiniciar procesadores TRT Pose
        self.reset_pose_processors()
        
        print("Estado base reiniciado.")
        logger.info("🔄 Estado de la fase reiniciado completamente")

    def restart_full_test(self):
        """
        Reinicia completamente el test SPPB, incluyendo todas las fases y el estado global.
        """
        print("\n🔄 REINICIANDO TEST COMPLETO SPPB...")
        print("⚠️  ATENCIÓN: Se reiniciarán todas las fases del test.")
        print("⚠️  Todos los resultados previos se perderán.")
        
        # Reiniciar estado global
        self._test_state = {
            'phase_results': [],
            'current_phase': None,
            'global_attempt': 1,
            'max_global_attempts': 3
        }
        
        # Reiniciar estado específico de la fase
        self.reset_test()
        
        print("✅ TEST COMPLETO REINICIADO - Comenzando desde el principio...")
        return True

    def wait_for_ready_with_restart(self, message="Presione ENTER cuando esté listo para comenzar..."):
        """
        Espera a que el usuario esté listo, pero también permite reiniciar, salir o saltar la prueba.
        
        Args:
            message (str): Mensaje que se mostrará al usuario
        
        Returns:
            str: 'continue' para continuar, 'restart' para reiniciar, 'exit' para salir, 'skip' para saltar, 'full_restart' para reiniciar test completo
        """
        print("\n" + message)
        print("🚨 Opciones disponibles:")
        print("  - Presione ENTER para CONTINUAR con la prueba")
        print("  - Escriba 'r' + ENTER para REINICIAR la prueba")
        print("  - Escriba 'R' + ENTER para REINICIAR TODO EL TEST SPPB")
        print("  - Escriba 's' + ENTER para SALTAR esta prueba (persona no capacitada)")
        print("  - Escriba 'q' + ENTER para SALIR completamente")
        print("  - Ctrl+C para CANCELAR INMEDIATAMENTE (emergencia)")
        
        try:
            user_input = input(">>> ").strip()
            
            if user_input == 'r':
                print("🔄 REINICIANDO la prueba...")
                return 'restart'
            elif user_input == 'R':
                print("🔄 REINICIANDO TODO EL TEST SPPB...")
                return 'full_restart'
            elif user_input.lower() == 's':
                print("⏭️ SALTANDO esta prueba...")
                return 'skip'
            elif user_input.lower() == 'q':
                print("🚪 SALIENDO del sistema...")
                return 'exit'
            else:
                print("✅ COMENZANDO la prueba...")
                return 'continue'
        except KeyboardInterrupt:
            print("\n🚨 CANCELACIÓN DE EMERGENCIA activada")
            return 'emergency_stop'

    def ask_for_restart(self, error_message=""):
        """
        Pregunta al usuario qué hacer después de un error, con opciones de emergencia.
        
        Args:
            error_message (str): Mensaje de error opcional
        
        Returns:
            str: 'restart', 'skip', 'exit', 'full_restart' según la decisión del usuario
        """
        if error_message:
            print(f"\n❌ ERROR CRÍTICO: {error_message}")
        
        print("\n🚨 OPCIONES DE EMERGENCIA:")
        print("1. REINICIAR la prueba (volver a intentar)")
        print("2. REINICIAR TODO EL TEST SPPB (desde el principio)")
        print("3. SALTAR esta prueba (persona no capacitada)")
        print("4. SALIR completamente del sistema")
        
        while True:
            try:
                choice = input("Seleccione una opción (1/2/3/4): ").strip()
                if choice == '1':
                    print("🔄 REINICIANDO la prueba...")
                    return 'restart'
                elif choice == '2':
                    print("🔄 REINICIANDO TODO EL TEST SPPB...")
                    return 'full_restart'
                elif choice == '3':
                    print("⏭️ SALTANDO esta prueba...")
                    return 'skip'
                elif choice == '4':
                    print("🚪 SALIENDO del sistema...")
                    return 'exit'
                else:
                    print("❌ Opción no válida. Seleccione 1, 2, 3 o 4.")
            except KeyboardInterrupt:
                print("\n🚨 CANCELACIÓN DE EMERGENCIA - Saliendo del sistema...")
                return 'exit'

    def run_with_restart(self, *args, **kwargs):
        """
        Ejecuta la prueba con capacidad de reinicio automático y manejo de emergencias.
        Las subclases deben implementar el método _run_phase.
        
        Returns:
            dict: Resultado de la prueba, None si se canceló, o dict con 'skipped': True si se saltó
        """
        max_attempts = 3
        attempt = 1
        
        while attempt <= max_attempts:
            try:
                print(f"\n🔄 Intento {attempt} de {max_attempts}")
                
                # Ejecutar la fase específica
                result = self._run_phase(*args, **kwargs)
                return self._handle_phase_result(result)
                    
            except KeyboardInterrupt:
                print("\n🚨 INTERRUPCIÓN DE EMERGENCIA detectada")
                action = self.ask_for_restart("Interrupción del usuario (Ctrl+C)")
                
                restart_result = self._handle_restart_action(action)
                if restart_result is not None:
                    return restart_result
                
                attempt += 1
                    
            except Exception as e:
                print(f"\n❌ ERROR TÉCNICO: {str(e)}")
                
                if attempt < max_attempts:
                    action = self.ask_for_restart(f"Error técnico: {str(e)}")
                    
                    restart_result = self._handle_restart_action(action)
                    if restart_result is not None:
                        return restart_result
                    
                    attempt += 1
                else:
                    return self._handle_max_attempts_reached()
        
        return None

    def _handle_phase_result(self, result):
        """
        Maneja el resultado de una fase ejecutada.
        """
        if result is not None:
            if isinstance(result, dict) and result.get('skipped', False):
                print("⏭️ Prueba saltada por decisión del usuario.")
                return result
            else:
                print("✅ Prueba completada exitosamente.")
                return result
        else:
            print("❌ Prueba cancelada por el usuario.")
            return None

    def _handle_restart_action(self, action):
        """
        Maneja las acciones de reinicio del usuario.
        
        Returns:
            dict o None: Resultado si la acción termina la prueba, None si debe continuar
        """
        if action == 'restart':
            self.reset_test()
            return None  # Continuar con el siguiente intento
        elif action == 'full_restart':
            self.restart_full_test()
            raise FullRestartRequested("FULL_RESTART_REQUESTED")  # Señal especial para reiniciar test completo
        elif action == 'skip':
            return {'skipped': True, 'reason': 'emergency_interrupt'}
        else:  # exit
            return None

    def _handle_max_attempts_reached(self):
        """
        Maneja el caso cuando se alcanza el máximo de intentos.
        """
        print("❌ Se alcanzó el máximo de intentos. Terminando prueba.")
        action = self.ask_for_restart("Máximo de intentos alcanzado")
        
        if action == 'full_restart':
            self.restart_full_test()
            raise FullRestartRequested("FULL_RESTART_REQUESTED")
        elif action == 'skip':
            return {'skipped': True, 'reason': 'max_attempts_reached'}
        else:
            return None

    def monitor_emergency_stop(self, message="Presione Ctrl+C para parar la prueba de emergencia"):
        """
        Método para mostrar información sobre cómo parar la prueba en caso de emergencia.
        Debe ser llamado al inicio de bucles largos de procesamiento.
        """
        print(f"🚨 {message}")
        print("   (Esta información se mostrará solo una vez por prueba)")

    def create_skipped_result(self, test_name, reason="user_choice"):
        """
        Crea un resultado estándar para una prueba saltada.
        
        Args:
            test_name (str): Nombre de la prueba
            reason (str): Razón por la cual se saltó la prueba
        
        Returns:
            dict: Resultado estándar para prueba saltada
        """
        return {
            'test': test_name,
            'skipped': True,
            'reason': reason,
            'score': 0,
            'message': f"Prueba {test_name} saltada: {reason}"
        }

    def _run_phase(self, *args, **kwargs):
        """
        Método abstracto que debe ser implementado por las subclases.
        Contiene la lógica específica de cada fase.
        
        Raises:
            NotImplementedError: Si no se implementa en la subclase
        """
        raise NotImplementedError("Las subclases deben implementar el método _run_phase")

    def run_test_with_global_restart(self, *args, **kwargs):
        """
        Ejecuta el test con manejo de reinicio global del test completo SPPB.
        
        Returns:
            dict: Resultado del test, None si se canceló
        """
        while True:
            try:
                return self.run_with_restart(*args, **kwargs)
            except FullRestartRequested:
                print("🔄 Reiniciando test completo SPPB...")
                continue  # Volver a ejecutar el test desde el principio
            except Exception as e:
                print(f"❌ Error inesperado: {str(e)}")
                return None
