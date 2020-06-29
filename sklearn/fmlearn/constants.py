class URI:
    _SERVER = 'https://fmlearn.herokuapp.com'
    _LOCAL = 'http://127.0.0.1:5000'

    _METRIC = '/metric'
    _RETRIEVE = '/retrieve'
    _MAX = '/max'
    _MIN = '/min'
    _ALL = '/all'
    _PREDICT = '/predict'

    def __init__(self, debug=False):
        if debug:
            self._SERVER = self._LOCAL

    def post_metric(self):
        return self._SERVER + self._METRIC

    def retrieve_all(self):
        return self._SERVER + self._METRIC + self._RETRIEVE + self._ALL

    def retrieve_best_min(self):
        return self._SERVER + self._METRIC + self._RETRIEVE + self._MIN

    def retrieve_best_max(self):
        return self._SERVER + self._METRIC + self._RETRIEVE + self._MAX

    def predict_metric(self):
        return self._SERVER + self._METRIC + self._PREDICT

