class URI:
    _SERVER = 'http://127.0.0.1:5000'

    _METRIC = '/metric'
    _RETRIEVE = '/retrieve'
    _BEST = '/best'
    _ALL = '/all'

    def post_metric(self):
        return self._SERVER + self._METRIC

    def retrieve_all(self):
        return self._SERVER + self._METRIC + self._RETRIEVE + self._ALL

    def retrieve_best(self):
        return self._SERVER + self._METRIC + self._RETRIEVE + self._BEST
