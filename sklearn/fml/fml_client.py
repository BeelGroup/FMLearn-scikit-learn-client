import json
import requests as req
from cryptography.fernet import Fernet
from fml.encription.key import FMLKey


class FMLClient:
    def __init__(self):
        return

    def _jprint(self, obj):
        # create a formatted string of the Python JSON object
        text = json.dumps(obj, sort_keys=True, indent=4)
        print(text)

    def _send_msg(self, data):
        """
        API call to the federated meta learning server
        """
        url = 'http://127.0.0.1:5000/metric'
        res = req.post(url, json=data)
        print(res.status_code)
        return res.json()

    def publish(self, algorithm_name, metric_name, metric_value, dataset):
        """
        Publishes the data collected to the federated meta learning API
        """
        key = FMLKey()
        f = Fernet(key.getKey())
        # converts the dataset to a byte object and then encrypts it and converts it to string
        dataset_hash = f.encrypt(dataset.encode()).decode("utf-8")

        data = {}
        data['algorithm_name'] = algorithm_name
        data['metric_name'] = metric_name
        data['metric_value'] = metric_value
        data['dataset_hash'] = dataset_hash
        return self._send_msg(data)

    def _test_publish(self, algorithm_name='Linear Regrission', metric_name='RMSE', metric_value='0', dataset='asdfasdfasdfd'):
        """
        Test Function to send message to the fml backend server!
        """
        self._jprint(self.publish(algorithm_name, metric_name, metric_value, dataset))