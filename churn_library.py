'''
Name: Mohammed Alqarni
Email: mosalehalqarni@gmail.com
Date: Desember 30, 2024
Nanodegree: ML DevOps Engineer

Project - Predict Customer Churn - of ML DevOps Engineer Nanodegree Udacity

In this module we provide the functions that have a codes from churn notebook
'''

# import libraries
import os
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


def import_data(path):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data: pandas dataframe
    '''
    data = pd.read_csv(path)
    print(data.head())

    return data


def perform_eda(data):
    '''
    perform eda on data and save figures to images folder
    input:
            data: pandas dataframe

    output:
            None
    '''
    # Create a new feature calls Churn
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Create a churn histogram
    churn_figure = plt.figure(figsize=(20, 10))
    data['Churn'].hist()
    churn_figure.savefig('./images/eda/churn.png')

    # Create a customer age histogram
    age_figure = plt.figure(figsize=(20, 10))
    data['Customer_Age'].hist()
    age_figure.savefig('./images/eda/customers_ages.png')

    # Create a marital status count plot
    marital_figure = plt.figure(figsize=(20, 10))
    data.Marital_Status.value_counts('normalize').plot(kind='bar')
    marital_figure.savefig('./images/eda/marital_status.png')

    # Create a total trans count histogram
    trans_ct_figure = plt.figure(figsize=(20, 10))
    sns.histplot(data['Total_Trans_Ct'], stat='density', kde=True)
    trans_ct_figure.savefig('./images/eda/total_trans_ct.png')

    # Create a heatmap to visulaize all relations between features
    heatmap_figure = plt.figure(figsize=(20, 10))
    sns.heatmap(data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    heatmap_figure.savefig('./images/eda/heatmap.png')


def encoder_helper(data, category_list):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data: pandas dataframe
            category_lst: list of columns
            that contain categorical features

    output:
            data: pandas dataframe with new columns for
    '''
    new_category_list = []
    for category in category_list:
        new_category = f"{category}_Churn"
        new_category_list.append(new_category)

    for category in category_list:
        lst = []
        groups = data.groupby(category).mean()['Churn']

    for value in data[category]:
        lst.append(groups.loc[value])

    for new_category in new_category_list:
        data[new_category] = lst

    return data


def perform_feature_engineering(data):
    '''
    input:
              data: pandas dataframe
    output:
              x_data: X data
              x_train: X training data
              x_test: X testing data
              x_train: y training data
              y_test: y testing data
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    x_data = data[keep_cols]
    y_data = data['Churn']

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42)

    return x_data, x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Random Forest Rebort
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/rfc_rebort.png')

    # Logistic Regression Rebort
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/lrc_rebort.png')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Our models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # Train models
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc,
                          param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Load models
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    # LogisticRegression accuracy plot
    lrc_plot = plot_roc_curve(lr_model, x_test, y_test)
    lrc_plot.figure_.savefig('./images/results/lrc_roc_curve.png')

    # Models accuracy plot
    plt.figure(figsize=(15, 8))
    axes = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, x_test, y_test, ax=axes, alpha=0.8)
    lrc_plot.plot(ax=axes, alpha=0.8)
    rfc_disp.figure_.savefig('./images/results/models_accuracy_curve.png')

    # Average impact on model output magnitude
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
    plt.savefig('./images/results/model_impact.png')


if __name__ == "__main__":

    # 1st function
    PATH = r"./data/bank_data.csv"
    DATA = import_data(PATH)

    # 2nd function
    perform_eda(DATA)

    # 3rd function
    CATEGORY_LIST = ['Gender', 'Education_Level', 'Marital_Status',
                    'Income_Category', 'Card_Category']
    DATA = encoder_helper(DATA, CATEGORY_LIST)

    # 4th function
    X_DATA, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(DATA)

    # 5th function
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)

    # 6th function
    RFC_MODEL = joblib.load('./models/rfc_model.pkl')
    OUTPUT_PTH = './images/results/feature_importances.png'
    feature_importance_plot(RFC_MODEL, X_DATA, OUTPUT_PTH)

    # 7th function
    Y_TRAIN_PREDS_RF = RFC_MODEL.predict(X_TRAIN)
    Y_TEST_PRESD_RF = RFC_MODEL.predict(X_TEST)

    LR_MODEL = joblib.load('./models/logistic_model.pkl')
    Y_TRAIN_PREDS_LR = LR_MODEL.predict(X_TRAIN)
    Y_TEST_PREDS_LR = LR_MODEL.predict(X_TEST)
    classification_report_image(Y_TRAIN,
                                Y_TEST,
                                Y_TRAIN_PREDS_LR,
                                Y_TRAIN_PREDS_RF,
                                Y_TEST_PREDS_LR,
                                Y_TEST_PRESD_RF)
