'''
Name: Mohammed Alqarni
Email: mosalehalqarni@gmail.com
Date: Desember 30, 2024
Nanodegree: ML DevOps Engineer

Project - Predict Customer Churn - of ML DevOps Engineer Nanodegree Udacity

In this module we provide the function that logging and testing
churn_library.py functions
'''

import os
import logging
from pandas import DataFrame, Series
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is
    completed for you to assist with the other test functions
    '''
    try:
        data = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
        return data
    except FileNotFoundError as err:
        logging.error("""Testing import_data: The
        file wasn't found""")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error("""Testing import_data: The file
        doesn't appear to have rows and columns""")
        raise err


def test_eda(perform_eda, data):
    '''
    test perform eda function
    '''
    try:
        perform_eda(data)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as e:
        logging.error("""Testing perform_eda: The eda
        process failed""")
        raise e

    try:
        assert os.path.exists("./images/eda/churn.png")
        assert os.path.exists("./images/eda/customers_ages.png")
        assert os.path.exists("./images/eda/heatmap.png")
        assert os.path.exists("./images/eda/marital_status.png")
        assert os.path.exists("./images/eda/total_trans_ct.png")
    except AssertionError as err:
        logging.error("""Testing perform_eda: The figures
        did not save in their designated paths""")
        raise err


def test_encoder_helper(encoder_helper, data, category_list):
    '''
    test encoder helper
    '''
    try:
        data = encoder_helper(data, category_list)
        logging.info("Testing encoder_helper: SUCCESS")
        return data
    except Exception as e:
        logging.error("""Testing encoder_helper:
        The encoding process failed""")
        raise e

    try:
        assert isinstance(data, DataFrame)
        assert isinstance(category_lst, list)
    except AssertionError as err:
        logging.error("""Testing encoder_helper:
        One of the inputs or both do
        not match the function requirements""")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error("""Testing encoder_helper: The DataFrame doesn't
        appear to have rows and columns""")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, data):
    '''
    test perform_feature_engineering
    '''
    try:
        x_data, x_train, x_test, y_train, y_test = perform_feature_engineering(data)
        logging.info("Testing perform_feature_engineering: SUCCESS")
        return x_data, x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error("""Testing perform_feature_engineering:
        The feature engineering process failed""")
        raise e

    try:
        assert isinstance(x_data, DataFrame)
        assert isinstance(x_train, DataFrame)
        assert isinstance(x_test, DataFrame)
        assert isinstance(y_train, Series)
        assert isinstance(y_test, Series)
    except AssertionError as err:
        logging.error("""Testing perform_feature_engineering:
        One of the outputs does not match the function requirements""")
        raise err


def test_train_models(train_models, x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''
    try:
        train_models(x_train, x_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except Exception as e:
        logging.error("Testing train_models: The train models process failed")
        raise e

    try:
        assert os.path.exists("./models/rfc_model.pkl")
        assert os.path.exists("./models/logistic_model.pkl")
    except AssertionError as err:
        logging.error("""Testing train_models: The models did not
        save in the designated paths""")
        raise err

    try:
        assert os.path.exists("./images/results/lrc_roc_curve.png")
        assert os.path.exists("./images/results/models_accuracy_curve.png")
        assert os.path.exists("./images/results/model_impact.png")
    except AssertionError as err:
        logging.error("""Testing train_models: Results
         curves did not save in the designated paths""")
        raise err


if __name__ == "__main__":
    # Testing import data
    DATA = test_import(cls.import_data)

    #Testing perform eda
    test_eda(cls.perform_eda, DATA)

    #Testing encoder helper
    CATEGORY_LIST = ['Gender', 'Education_Level',
                      'Marital_Status','Income_Category',
                      'Card_Category']
    DATA = test_encoder_helper(cls.encoder_helper, DATA, CATEGORY_LIST)

    #Testing features engineering
    X_DATA, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(cls.perform_feature_engineering, DATA)

    #Testing train model
    test_train_models(cls.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
