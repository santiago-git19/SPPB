class PhaseBase:
    """
    Clase base para todas las fases del test SPPB.
    Contiene funcionalidad común como la espera de preparación del usuario.
    """
    
    def __init__(self, openpose, config):
        self.openpose = openpose
        self.config = config
        self.interval = getattr(config, 'fps', 5.0)
        if self.interval:
            self.interval = 1.0 / self.interval
        else:
            self.interval = 0.2

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
