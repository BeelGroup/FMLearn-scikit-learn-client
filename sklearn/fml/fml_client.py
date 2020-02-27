import json
import requests as req
from fml.constants import URI
from sklearn import linear_model
from fml.encryption.fml_hash import FMLHash


class FMLClient:
    def __init__(self):
        return

    def _jprint(self, obj):
        # create a formatted string of the Python JSON object
        text = json.dumps(obj, sort_keys=True, indent=4)
        print(text)

    def _post_msg(self, uri, data):
        """
        API call to the federated meta learning server
        """
        res = req.post(uri, json=data)
        print(res.status_code)
        return res.json()

    def publish(self, model, metric_name, metric_value, dataset, params=None):
        """
        Publishes the data collected to the federated meta learning API
        """
        dataset_hash = FMLHash().hashValAndReturnString(dataset)

        algorithm_name = str(model.__class__)

        data = {}
        data['algorithm_name'] = algorithm_name
        data['metric_name'] = metric_name
        data['metric_value'] = metric_value
        data['dataset_hash'] = dataset_hash
        if params != None:
            model_params = []
            for key, value in params.items():
                new_param = {}
                new_param['param_name'] = str(key)
                new_param['param_value'] = str(value)
                model_params.append(new_param)
            data['params'] = model_params
        else:
            data['params'] = ""

        return self._post_msg(URI().post_metric(), data)


    def retrieve_all_metrics(self, dataset):
        """
        Function to retrieve all metric that matches the dataset_hash
        """
        dataset_hash = FMLHash().hashValAndReturnString(dataset)
        
        data = {}
        data['dataset_hash'] = dataset_hash

        return self._post_msg(URI().retrieve_all(), data)

    def retrieve_best_metric(self, dataset, min=True):
        """
        Function to retrieve metric that best matches the dataset_hash
        @min = True fetch the minimum value of the metric
        @min = False fetches the maximum value of the metric
        """
        dataset_hash = FMLHash().hashValAndReturnString(dataset)
        
        data = {}
        data['dataset_hash'] = dataset_hash

        if min:
            return self._post_msg(URI().retrieve_best_min(), data)
        else:
            return self._post_msg(URI().retrieve_best_max(), data)


    def _test_publish(self, model=linear_model.LinearRegression(), metric_name='RMSE', metric_value='0', dataset='asdfasdfasdfd'):
        """
        Test Function to send message to the fml backend server!
        """
        self._jprint(self.publish(model, metric_name, metric_value, dataset))
