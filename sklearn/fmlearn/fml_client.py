import json
import requests as req
import numpy as np

from sklearn import linear_model
from sklearn.utils.multiclass import type_of_target

from autosklearn import constants as ask_const
from autosklearn.data.xy_data_manager import XYDataManager

from fmlearn.constants import URI
from fmlearn.encryption.fml_hash import FMLHash
from fmlearn.metafeatures import MetaFeatures

class FMLClient:
    def __init__(self, debug=False):
        self._task_mapping = {
            'multilabel-indicator': ask_const.MULTILABEL_CLASSIFICATION,
            'multiclass': ask_const.MULTICLASS_CLASSIFICATION,
            'binary': ask_const.BINARY_CLASSIFICATION
        }
        self.dataset_name = None
        self.data_manager = None
        self.target_type = None
        self.meta_features = None
        self.pub_meta_feat = None
        self.uri = URI(debug=debug)
        return

    def _jprint(self, obj):
        """
        create a formatted string of the Python JSON object
        """
        text = json.dumps(obj, sort_keys=True, indent=4)
        print(text)

    def _post_msg(self, uri, data):
        """
        API call to the federated meta learning server
        """
        res = req.post(uri, json=data)
        print(res.status_code)
        return res.json()

    def set_dataset(self, X, y, X_test=None, y_test=None, feat_type=None):
        """
        Stores the obtained dataset parameters in the XYDataManager of auto-sklearn
        and caclulates the metafeatures of the dataset
        """

        utils = MetaFeatures()
        X, y = utils.perform_input_checks(X, y)
        if X_test is not None:
            X_test, y_test = utils.perform_input_checks(X_test, y_test)
            if len(y.shape) != len(y_test.shape):
                raise ValueError('Target value shapes do not match: %s vs %s'
                                 % (y.shape, y_test.shape))

        if feat_type is not None and len(feat_type) != X.shape[1]:
            raise ValueError('Array feat_type does not have same number of '
                             'variables as X has features. %d vs %d.' %
                             (len(feat_type), X.shape[1]))
        if feat_type is not None and not all([isinstance(f, str)
                                              for f in feat_type]):
            raise ValueError('Array feat_type must only contain strings.')
        if feat_type is not None:
            for ft in feat_type:
                if ft.lower() not in ['categorical', 'numerical']:
                    raise ValueError('Only `Categorical` and `Numerical` are '
                                     'valid feature types, you passed `%s`' % ft)

        self.target_type = type_of_target(y)
        task = self._task_mapping.get(self.target_type)
        if task == None:
            task = ask_const.REGRESSION

        self.dataset_name = FMLHash().hashValAndReturnString(str(X))
        self.data_manager = XYDataManager(X, y, X_test, y_test, task, feat_type, self.dataset_name)

        self.meta_features = utils.calculate_metafeatures(self.data_manager, self.dataset_name)

    def publish(self, model, metric_name, metric_value, params=None):
        """
        Publishes the data collected to the federated meta learning API
        """
        if self.data_manager is None:
            raise ValueError('Data Manager not set, set the dataset using \'set_dataset()\' before \'publish()\'')

        algorithm_name = str(model.__class__)

        utils = MetaFeatures()

        data = {}
        data['algorithm_name'] = algorithm_name
        data['metric_name'] = metric_name
        data['metric_value'] = metric_value
        data['dataset_hash'] = self.dataset_name
        data['data_meta_features'] = utils.get_meta_feaures_for_publish(self.meta_features, self.pub_meta_feat)
        data['target_type'] = self.target_type
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

        return self._post_msg(self.uri.post_metric(), data)


    def retrieve_all_metrics(self, dataset):
        """
        Function to retrieve all metric that matches the dataset_hash
        """
        dataset_hash = FMLHash().hashValAndReturnString(dataset)
        
        data = {}
        data['dataset_hash'] = dataset_hash

        return self._post_msg(self.uri.retrieve_all(), data)

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
            return self._post_msg(self.uri.retrieve_best_min(), data)
        else:
            return self._post_msg(self.uri.retrieve_best_max(), data)


    def _test_publish(self, model=linear_model.LinearRegression(), metric_name='RMSE', metric_value='0', dataset='asdfasdfasdfd'):
        """
        Test Function to send message to the fml backend server!
        """
        self._jprint(self.publish(model, metric_name, metric_value, dataset))
