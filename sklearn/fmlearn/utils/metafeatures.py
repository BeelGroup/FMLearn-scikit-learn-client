import numpy as np
import scipy

from sklearn.utils import check_array

from autosklearn import constants as ask_const
from autosklearn.smbo import EXCLUDE_META_FEATURES_CLASSIFICATION, EXCLUDE_META_FEATURES_REGRESSION
from autosklearn.metalearning.metafeatures.metafeatures import calculate_all_metafeatures_with_labels

class MetaFeatures:
    def __init__(self):
        return

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

    def perform_input_checks(self, X, y):
        """
        Input Validation method for the dataset(X, y)
        """
        X = self._check_X(X)
        if y is not None:
            y = self._check_y(y)
        return X, y

    def get_meta_feaures(self, meta_features):
        """
        Function which @returns the dataset's meta features as a key, value pair
        """
        meta_feat_map = {}

        if meta_features == None:
            return meta_feat_map
    
        metafeature_values = meta_features.metafeature_values
        for key, val in metafeature_values.items():
            meta_feat_map[key] = val.value
        
        return meta_feat_map

    def get_meta_feaures_for_publish(self, meta_features, pub_meta_feat):
        """
        Function which @returns the dataset's meta features in a format
        which can be used to in the FMLearn application
        """
        meta_feat_list = []

        if meta_features == None:
            return meta_feat_list

        if pub_meta_feat is not None:
            return pub_meta_feat
    
        metafeature_values = meta_features.metafeature_values
        for key, val in metafeature_values.items():
            new_feat = {}
            new_feat['feat_name'] = str(key)
            new_feat['feat_value'] = str(val.value)
            meta_feat_list.append(new_feat)
        
        pub_meta_feat = meta_feat_list

        return pub_meta_feat

    def calculate_metafeatures(self, data_manager, dataset_name):
        """
        A function to calculate the dataset's meta features
        internally called Auto-SKLearn's caclulate_all_metafeatures_with_labels()
        and stores the returned DatasetMetaFeatures Object
        """

        categorical = [True if feat_type.lower() in ['categorical'] else False
                   for feat_type in data_manager.feat_type]

        EXCLUDE_META_FEATURES = EXCLUDE_META_FEATURES_CLASSIFICATION \
            if data_manager.info['task'] in ask_const.CLASSIFICATION_TASKS else EXCLUDE_META_FEATURES_REGRESSION

        if data_manager.info['task'] in [ask_const.MULTICLASS_CLASSIFICATION, ask_const.BINARY_CLASSIFICATION,
                            ask_const.MULTILABEL_CLASSIFICATION, ask_const.REGRESSION]:

            result = calculate_all_metafeatures_with_labels(
                data_manager.data['X_train'], 
                data_manager.data['Y_train'], 
                categorical=categorical,
                dataset_name=dataset_name,
                dont_calculate=EXCLUDE_META_FEATURES, )

            for key in list(result.metafeature_values.keys()):
                if result.metafeature_values[key].type_ != 'METAFEATURE':
                    del result.metafeature_values[key]

        else:
            result = None

        return result
