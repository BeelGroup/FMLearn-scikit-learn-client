import json
import requests as req
import numpy as np
import scipy

from sklearn import linear_model
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target

from autosklearn import constants as ask_const
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.smbo import EXCLUDE_META_FEATURES_CLASSIFICATION, EXCLUDE_META_FEATURES_REGRESSION
from autosklearn.metalearning.metafeatures.metafeatures import calculate_all_metafeatures_with_labels

from fml.constants import URI
from fml.encryption.fml_hash import FMLHash

class FMLClient:
    def __init__(self):
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
        return

    def _perform_input_checks(self, X, y):
        """
        Input Validation method for the dataset(X, y)
        """
        X = self._check_X(X)
        if y is not None:
            y = self._check_y(y)
        return X, y

    def _check_X(self, X):
        """
        Input Validation method for dataset's features
        """
        X = check_array(X, accept_sparse="csr", force_all_finite=False)
        if scipy.sparse.issparse(X):
            X.sort_indices()
        return X

    def _check_y(self, y):
        """
        Input Validation method for dataset's target
        """
        y = check_array(y, ensure_2d=False)
        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = np.ravel(y)
        return y

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

    def _get_meta_feaures(self):
        """
        Function which @returns the dataset's meta features as a key, value pair
        """
        meta_feat_map = {}

        if self.meta_features == None:
            return meta_feat_map
    
        metafeature_values = self.meta_features.metafeature_values
        for key, val in metafeature_values.items():
            meta_feat_map[key] = val.value
        
        return meta_feat_map

    def _get_meta_feaures_for_publish(self):
        """
        Function which @returns the dataset's meta features in a format
        which can be used to in the FMLearn application
        """
        meta_feat_list = []

        if self.meta_features == None:
            return meta_feat_list

        if self.pub_meta_feat is not None:
            return self.pub_meta_feat
    
        metafeature_values = self.meta_features.metafeature_values
        for key, val in metafeature_values.items():
            new_feat = {}
            new_feat['feat_name'] = str(key)
            new_feat['feat_value'] = str(val.value)
            meta_feat_list.append(new_feat)
        
        self.pub_meta_feat = meta_feat_list

        return self.pub_meta_feat

    def _calculate_metafeatures(self):
        """
        A function to calculate the dataset's meta features
        internally called Auto-SKLearn's caclulate_all_metafeatures_with_labels()
        and stores the returned DatasetMetaFeatures Object
        """

        categorical = [True if feat_type.lower() in ['categorical'] else False
                   for feat_type in self.data_manager.feat_type]

        EXCLUDE_META_FEATURES = EXCLUDE_META_FEATURES_CLASSIFICATION \
            if self.data_manager.info['task'] in ask_const.CLASSIFICATION_TASKS else EXCLUDE_META_FEATURES_REGRESSION

        if self.data_manager.info['task'] in [ask_const.MULTICLASS_CLASSIFICATION, ask_const.BINARY_CLASSIFICATION,
                            ask_const.MULTILABEL_CLASSIFICATION, ask_const.REGRESSION]:

            result = calculate_all_metafeatures_with_labels(
                self.data_manager.data['X_train'], 
                self.data_manager.data['Y_train'], 
                categorical=categorical,
                dataset_name=self.dataset_name,
                dont_calculate=EXCLUDE_META_FEATURES, )

            for key in list(result.metafeature_values.keys()):
                if result.metafeature_values[key].type_ != 'METAFEATURE':
                    del result.metafeature_values[key]

        else:
            result = None

        self.meta_features = result

    def set_dataset(self, X, y, X_test=None, y_test=None, feat_type=None):
        """
        Stores the obtained dataset parameters in the XYDataManager of auto-sklearn
        and caclulates the metafeatures of the dataset
        """

        X, y = self._perform_input_checks(X, y)
        if X_test is not None:
            X_test, y_test = self._perform_input_checks(X_test, y_test)
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

        self._calculate_metafeatures()

    def publish(self, model, metric_name, metric_value, params=None):
        """
        Publishes the data collected to the federated meta learning API
        """

        algorithm_name = str(model.__class__)

        data = {}
        data['algorithm_name'] = algorithm_name
        data['metric_name'] = metric_name
        data['metric_value'] = metric_value
        data['dataset_hash'] = self.dataset_name
        data['data_meta_features'] = self._get_meta_feaures_for_publish()
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

        print(data)
        #return self._post_msg(URI().post_metric(), data)


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
