class SPPBResult:
    def __init__(self, balance, gait, chair):
        self.balance = balance
        self.gait = gait
        self.chair = chair

    def total_score(self): # POR REVISAR SI FUNCIONA BIEN
        """
        Suma las puntuaciones de las tres fases.
        """
        score = 0
        # balance, gait y chair pueden ser dicts o listas de dicts
        # Buscamos la clave 'score' en cada resultado
        if isinstance(self.balance, list):
            for item in self.balance:
                if isinstance(item, dict) and 'score' in item:
                    score += item['score']
        elif isinstance(self.balance, dict) and 'score' in self.balance:
            score += self.balance['score']

        if isinstance(self.gait, dict) and 'score' in self.gait:
            score += self.gait['score']

        if isinstance(self.chair, dict) and 'score' in self.chair:
            score += self.chair['score']

        return score

    def to_dict(self):
        """
        Serializa los resultados a un diccionario.
        """
        return {
            'balance': self.balance,
            'gait': self.gait,
            'chair': self.chair,
            'total_score': self.total_score()
        }
