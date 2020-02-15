class URI:
    _SERVER = 'http://127.0.0.1:5000'

    _METRIC = '/metric'
    _RETRIEVE = '/retrieve'
    _MAX = '/max'
    _MIN = '/min'
    _ALL = '/all'

    def post_metric(self):
        return self._SERVER + self._METRIC

    def retrieve_all(self):
        return self._SERVER + self._METRIC + self._RETRIEVE + self._ALL

    def retrieve_best_min(self):
        return self._SERVER + self._METRIC + self._RETRIEVE + self._MIN

    def retrieve_best_max(self):
        return self._SERVER + self._METRIC + self._RETRIEVE + self._MAX
