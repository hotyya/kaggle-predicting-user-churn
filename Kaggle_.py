import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_predict

def compute_meta_features(model_, train_data, train_target, test_):
    train_answers = cross_val_predict(model_, train_data, train_target, cv = 5, 
                                      method = 'predict_proba')[:, 1]
    model_.fit(train_data, train_target)
    return train_answers, model_.predict_proba(test_)[:, 1]

def remove_spaces(value):
    try:
        value = float(value)
        return value
    except ValueError:
        return 0.

def data_analyze(data_, num_data, cat_data, target_col):
    gone_ = data_[data_['Churn'] == 1]
    stay_ = data_[data_['Churn'] == 0]

    n_cols = 3
    fig, axes = plt.subplots(3, 1, figsize = (18 , 10))
    fig.suptitle('Гистограмма для численных признаков', fontsize = 15)

    for i, n_col in enumerate(num_data):
        axes[i].hist(x = gone_[n_col], bins = 18, density = True, alpha = 0.5, 
                     label = 'ушел', edgecolor = 'black', color = 'purple')
        axes[i].hist(x = stay_[n_col], bins = 18, density = True, alpha = 0.5, 
                     label = 'остался', edgecolor = 'black', color = 'white')
        axes[i].set_title(n_col, size = 10)
        axes[i].legend()
    plt.show()

    fig, axes = plt.subplots(4, 4, figsize = (25, 25))
    ax_ = axes.ravel()
    fig.suptitle('Категориальные признаки', fontsize = 15)
    
    for i, cat_ in enumerate(cat_data):
        ax_[i].pie(x = data_[cat_].value_counts(), labels = data_[cat_].value_counts().index,
                  colors = ['pink', 'white', 'navajowhite', 'lightsteelblue'], labeldistance = 0.5,
                  shadow = True)
        ax_[i].set_title(cat_, fontsize = 10)
    plt.show()

    plt.pie(data_['Churn'].value_counts(), labels = ['остался', 'ушел'], autopct = '%.0f',
            colors = ['pink', 'white'], shadow = True)
    plt.title('Процентное соотношение целевой переменной')
    plt.show()
    return 0

def corr_(data_, target_col):
    for col_ in range(30):
        temp_data = pd.concat([target_col, data_[data_.columns[[col_]]]], axis = 1)
        sns.heatmap(temp_data.corr(), annot = True, cmap = 'coolwarm',
                    vmin = -1, vmax = 1, annot_kws = {"size" : 16})
        plt.show()
    return 0

def data_preprocessing(data_, test_data, flag):
    print(data_.info())
    print(np.sum(data_.isna()))
    
    data_['TotalSpent'] = data_['TotalSpent'].apply(remove_spaces)
    test_data['TotalSpent'] = test_data['TotalSpent'].apply(remove_spaces)

    num_data = data_[data_.columns[[0, 1, 2]]]
    cat_data = data_[data_.columns[[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]]
    num_test_data = test_data[test_data.columns[[0, 1, 2]]]
    cat_test_data = test_data[test_data.columns[[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]]
    target_col = data_['Churn']
    
    print(data_['TotalSpent'].value_counts())

    #data_analyze(data_, num_data, cat_data, target_col)


    dum_data = pd.get_dummies(cat_data, drop_first = True)
    dum_test_data = pd.get_dummies(cat_test_data, drop_first = True)
    data_new = pd.concat([num_data, dum_data], axis = 1)
    test_data_new = pd.concat([num_test_data, dum_test_data], axis = 1)

    #corr_(data_new, target_col)

    if (flag):
        data_new = data_new.drop(columns = ['HasMultiplePhoneNumbers_Yes',
                                       'HasDeviceProtection_Yes', 'HasMovieSubscription_Yes', 
                                       'HasPhoneService_Yes', 
                                       'HasMultiplePhoneNumbers_No phone service', 
                                       'HasOnlineBackup_Yes', 'HasOnlineTV_Yes', 
                                       'PaymentMethod_Mailed check', 'Sex_Male'])
        test_data_new = test_data_new.drop(columns = ['HasMultiplePhoneNumbers_Yes',
                                       'HasDeviceProtection_Yes', 'HasMovieSubscription_Yes', 
                                       'HasPhoneService_Yes', 
                                       'HasMultiplePhoneNumbers_No phone service', 
                                       'HasOnlineBackup_Yes', 'HasOnlineTV_Yes', 
                                       'PaymentMethod_Mailed check', 'Sex_Male'])

    if (not flag):
        data_new['Feat_1'] = data_new['TotalSpent'] * data_new['ClientPeriod']
        data_new['Feat_2'] = data_new['TotalSpent'] * data_new['MonthlySpending']
        data_new['Feat_3'] = data_new['ClientPeriod'] * data_new['ClientPeriod']
        data_new['Feat_4'] = data_new['TotalSpent'] * data_new['TotalSpent']

        test_data_new['Feat_1'] = test_data_new['TotalSpent'] * test_data_new['ClientPeriod']
        test_data_new['Feat_2'] = test_data_new['TotalSpent'] * test_data_new['MonthlySpending']
        test_data_new['Feat_3'] = test_data_new['ClientPeriod'] * test_data_new['ClientPeriod']
        test_data_new['Feat_4'] = test_data_new['TotalSpent'] * test_data_new['TotalSpent']

    return data_new, target_col, test_data_new

def my_logistic_regression(data_, target_col, test_):
    train_data, test_data, train_target, test_target = train_test_split(data_, target_col,
                                                                        test_size = 0.2, 
                                                                        random_state = 42)
    scaled_train = StandardScaler()
    scaled_test = StandardScaler()
    scaled_test_data_ = StandardScaler()
    scaled_train_data = scaled_train.fit_transform(train_data)
    scaled_test_data = scaled_test.fit_transform(test_data)
    scaled_test_ = scaled_test_data_.fit_transform(test_)
    
    model_ = LogisticRegression()

    param_grid = {'penalty': ['l1', 'l2', 'elasticnet'],
                  'C': np.linspace(0., 2., 10)}

    search_param = GridSearchCV(model_, param_grid, scoring = 'roc_auc', refit = True, n_jobs = -1)
    search_param.fit(scaled_train_data, train_target)
    print(search_param.best_params_)

    log_reg = LogisticRegression(C = 2., solver = 'saga', penalty = 'l2')
    log_reg.fit(scaled_train_data, train_target)
    y_pred_ = log_reg.predict_proba(scaled_test_data)[:, 1]

    print("-----------------------------------")
    print(roc_auc_score(test_target, y_pred_))
    print(search_param.best_score_)

    y_pred = log_reg.predict_proba(scaled_test_)[:, 1]
    y_pred1 = pd.DataFrame(y_pred)
    id = pd.DataFrame(np.array(np.arange(0, len(y_pred))))
    y_prediction = pd.concat([id, y_pred1], axis = 1)
    y_prediction.columns = ['Id', 'Churn']

    return y_prediction

def knn(data_, target_col, test_):
    train_data, test_data, train_target, test_target = train_test_split(data_, target_col,
                                                                        test_size = 0.2, 
                                                                        random_state = 42)
    scaled_train = StandardScaler()
    scaled_test = StandardScaler()
    scaled_test_data_ = StandardScaler()
    scaled_train_data = scaled_train.fit_transform(train_data)
    scaled_test_data = scaled_test.fit_transform(test_data)
    scaled_test_ = scaled_test_data_.fit_transform(test_)

    model_ = KNeighborsClassifier()

    param_grid = {'n_neighbors': np.arange(1, 1000, 50), 
                  'weights': ['uniform', 'distance'], 
                  'p': [1, 2]
                  }

    search_param = GridSearchCV(model_, param_grid, cv = 5, scoring = 'roc_auc',
                               n_jobs = -1)
    search_param.fit(scaled_train_data, train_target)
    print(search_param.best_params_)

    knn_model = KNeighborsClassifier(n_neighbors = 401, p = 1, weights = 'uniform')
    knn_model.fit(scaled_train_data, train_target)
    y_pred_ = knn_model.predict_proba(scaled_test_data)[:, 1]

    print("------------------------")
    print(roc_auc_score(test_target, y_pred_))
    print(search_param.best_score_)

    y_pred = knn_model.predict_proba(scaled_test_)[:, 1]
    y_pred1 = pd.DataFrame(y_pred)
    id = pd.DataFrame(np.array(np.arange(0, len(y_pred))))
    y_prediction = pd.concat([id, y_pred1], axis = 1)
    y_prediction.columns = ['Id', 'Churn']
    return y_prediction

def forest(data_, target_col, test_):
    train_data, test_data, train_target, test_target = train_test_split(data_, target_col,
                                                                        test_size = 0.2, 
                                                                        random_state = 42)
    scaled_train = StandardScaler()
    scaled_test = StandardScaler()
    scaled_test_data_ = StandardScaler()
    scaled_train_data = train_data
    scaled_test_data = test_data
    scaled_test_ = test_

    model_ = RandomForestClassifier()

    param_grid = {'max_depth': np.arange(1, 10)
                  }

    search_param = GridSearchCV(model_, param_grid, scoring = 'roc_auc', n_jobs = -1)
    search_param.fit(scaled_train_data, train_target)
    print(search_param.best_params_)

    ran_for = RandomForestClassifier(n_estimators = 700, max_depth = 7)
    ran_for.fit(scaled_train_data, train_target)
    y_pred_ = ran_for.predict_proba(scaled_test_data)[:, 1]

    print("---------------------")
    print(roc_auc_score(test_target, y_pred_))
    print(search_param.best_score_)

    y_pred = ran_for.predict_proba(scaled_test_)[:, 1]
    y_pred1 = pd.DataFrame(y_pred)
    id = pd.DataFrame(np.array(np.arange(0, len(y_pred))))
    y_prediction = pd.concat([id, y_pred1], axis = 1)
    y_prediction.columns = ['Id', 'Churn']
    return y_prediction

def boost(data_, target_col, cat_cols, test_):
    model_ = CatBoostClassifier(logging_level = 'Silent')
    data_ = data_.drop(columns = ['Churn'])

    cat_cols = [
    'Sex',
    'IsSeniorCitizen',
    'HasPartner',
    'HasChild',
    'HasPhoneService',
    'HasMultiplePhoneNumbers',
    'HasInternetService',
    'HasOnlineSecurityService',
    'HasOnlineBackup',
    'HasDeviceProtection',
    'HasTechSupportAccess',
    'HasOnlineTV',
    'HasMovieSubscription',
    'HasContractPhone',
    'IsBillingPaperless',
    'PaymentMethod'
    ]

    train_data, test_data, train_target, test_target = train_test_split(data_, target_col,
                                                                        test_size = 0.2,
                                                                        random_state = 42)

    #model_.fit(train_data, train_target, cat_features = cat_cols)
    #y_pred = model_.predict_proba(test_data)
    #print(roc_auc_score(test_target, y_pred[:, 1]))

    param_grid = {'iterations': np.arange(100, 1001, 100),
                  'depth': [2, 11, 2]}

    #search_param = GridSearchCV(model_, param_grid, scoring = 'roc_auc', cv = 5, n_jobs = -1)
    #search_param.fit(train_data, train_target, cat_features = cat_cols)
    #print(search_param.best_params_)

    boost_ = CatBoostClassifier(logging_level = 'Silent', depth = 2, iterations = 1000)
    boost_.fit(train_data, train_target, cat_features = cat_cols)
    y_pred = boost_.predict_proba(test_data)[:, 1]
    print(roc_auc_score(test_target, y_pred))

    y_pred = boost_.predict_proba(test_)[:, 1]

    print(y_pred)

    y_pred1 = pd.DataFrame(y_pred)
    print(y_pred1)
    id = pd.DataFrame(np.array(np.arange(0, len(y_pred))))
    print(id)
    y_prediction = pd.concat([id, y_pred1], axis = 1)
    y_prediction.columns = ['Id', 'Churn']

    print(y_prediction)

    return y_prediction

def stack(data_, test_):
    meta_features_train = np.zeros((data_.shape[0], 0))
    meta_features_test = np.zeros((test_.shape[0], 0))

    model_ = LogisticRegression(C = 1., solver = 'saga', penalty = 'elasticnet', l1_ratio = 0.2)
    data_new, target_col, test_data_new = data_preprocessing(data_, test_, False)
    scaled_train = StandardScaler()
    scaled_test = StandardScaler()
    scaled_train_data = scaled_train.fit_transform(data_new)
    scaled_test_data = scaled_test.fit_transform(test_data_new)
    train_data, test_data = compute_meta_features(model_, scaled_train_data, target_col, 
                                                  scaled_test_data)
    meta_features_train = np.append(meta_features_train, train_data.reshape((train_data.size, 1)),
                                   axis = 1)
    meta_features_test = np.append(meta_features_test, test_data.reshape((test_data.size, 1)),
                                   axis = 1)

    model_ = KNeighborsClassifier(n_neighbors = 58, p = 1, weights = 'uniform')
    data_new, target_col, test_data_new = data_preprocessing(data_, test_, True)
    scaled_train = StandardScaler()
    scaled_test = StandardScaler()
    scaled_train_data = scaled_train.fit_transform(data_new)
    scaled_test_data = scaled_test.fit_transform(test_data_new)
    train_data, test_data = compute_meta_features(model_, scaled_train_data, target_col, 
                                                  scaled_test_data)
    meta_features_train = np.append(meta_features_train, train_data.reshape((train_data.size, 1)),
                                   axis = 1)
    meta_features_test = np.append(meta_features_test, test_data.reshape((test_data.size, 1)),
                                   axis = 1)

    model_ = RandomForestClassifier(n_estimators = 150, max_depth = 6)
    data_new, target_col, test_data_new = data_preprocessing(data_, test_, True)
    train_data, test_data = compute_meta_features(model_, data_new, target_col, 
                                                  test_data_new)
    meta_features_train = np.append(meta_features_train, train_data.reshape((train_data.size, 1)),
                                   axis = 1)
    meta_features_test = np.append(meta_features_test, test_data.reshape((test_data.size, 1)),
                                   axis = 1)

    model_ = RandomForestClassifier(n_estimators = 250, max_depth = 7)
    data_new, target_col, test_data_new = data_preprocessing(data_, test_, True)
    train_data, test_data = compute_meta_features(model_, data_new, target_col, 
                                                  test_data_new)
    meta_features_train = np.append(meta_features_train, train_data.reshape((train_data.size, 1)),
                                   axis = 1)
    meta_features_test = np.append(meta_features_test, test_data.reshape((test_data.size, 1)),
                                   axis = 1)

    model_ = LogisticRegression(C = 120., solver = 'saga', penalty = 'elasticnet', l1_ratio = 0.2)
    data_new, target_col, test_data_new = data_preprocessing(data_, test_, False)
    scaled_train = StandardScaler()
    scaled_test = StandardScaler()
    scaled_train_data = scaled_train.fit_transform(data_new)
    scaled_test_data = scaled_test.fit_transform(test_data_new)
    train_data, test_data = compute_meta_features(model_, scaled_train_data, target_col, 
                                                  scaled_test_data)
    meta_features_train = np.append(meta_features_train, train_data.reshape((train_data.size, 1)),
                                   axis = 1)
    meta_features_test = np.append(meta_features_test, test_data.reshape((test_data.size, 1)),
                                   axis = 1)

    #model_ = CatBoostClassifier(logging_level = 'Silent', depth = 1, iterations = 1000)
    #data_new, target_col, test_data_new = data_preprocessing(data_, test_, True)
    #cat_cols = [
    #'Sex',
    #'IsSeniorCitizen',
    #'HasPartner',
    #'HasChild',
    #'HasPhoneService',
    #'HasMultiplePhoneNumbers',
    #'HasInternetService',
    #'HasOnlineSecurityService',
    #'HasOnlineBackup',
    #'HasDeviceProtection',
    #'HasTechSupportAccess',
    #'HasOnlineTV',
    #'HasMovieSubscription',
    #'HasContractPhone',
    #'IsBillingPaperless',
    #'PaymentMethod'
    #]
    #data = data_.drop(columns = ['Churn'])
    #model_.fit(data, target_col, cat_features = cat_cols)
    #train_data = model_.predict_proba(data)[:, 1]
    #test_data = model_.predict_proba(test_)[:, 1]
    #meta_features_train = np.append(meta_features_train, train_data.reshape((train_data.size, 1)),
    #                               axis = 1)
    #meta_features_test = np.append(meta_features_test, test_data.reshape((test_data.size, 1)),
    #                               axis = 1)

    #model_ = CatBoostClassifier(logging_level = 'Silent', depth = 2, iterations = 700)
    #data_new, target_col, test_data_new = data_preprocessing(data_, test_, True)
    #model_.fit(data, target_col, cat_features = cat_cols)
    #train_data = model_.predict_proba(data)[:, 1]
    #test_data = model_.predict_proba(test_)[:, 1]
    #meta_features_train = np.append(meta_features_train, train_data.reshape((train_data.size, 1)),
    #                               axis = 1)
    #meta_features_test = np.append(meta_features_test, test_data.reshape((test_data.size, 1)),
    #                               axis = 1)

    meta_features_train = np.append(meta_features_train, (meta_features_train[:, 2]*meta_features_train[:, 4]).reshape((meta_features_train[:, 2]*meta_features_train[:, 4]).size, 1),
                                    axis = 1)
    meta_features_test = np.append(meta_features_test, (meta_features_test[:, 2]*meta_features_test[:, 4]).reshape((meta_features_test[:, 2]*meta_features_test[:, 4]).size, 1),
                                    axis = 1)

    meta_features_train = np.append(meta_features_train, (meta_features_train[:, 5]*meta_features_train[:, 3]).reshape((meta_features_train[:, 5]*meta_features_train[:, 3]).size, 1),
                                    axis = 1)
    meta_features_test = np.append(meta_features_test, (meta_features_test[:, 5]*meta_features_test[:, 3]).reshape((meta_features_test[:, 5]*meta_features_test[:, 3]).size, 1),
                                    axis = 1)

    meta_features_train = np.append(meta_features_train, (meta_features_train[:, 0]*meta_features_train[:, 1]).reshape((meta_features_train[:, 0]*meta_features_train[:, 1]).size, 1),
                                    axis = 1)
    meta_features_test = np.append(meta_features_test, (meta_features_test[:, 0]*meta_features_test[:, 1]).reshape((meta_features_test[:, 0]*meta_features_test[:, 1]).size, 1),
                                    axis = 1)

    y_prediction = forest(meta_features_train, target_col, meta_features_test)
    #y_prediction = my_logistic_regression(meta_features_train, target_col, meta_features_test)
    #y_prediction = knn(meta_features_train, target_col, meta_features_test)
    return y_prediction

data_ = pd.read_csv('train.csv')
data_['TotalSpent'] = data_['TotalSpent'].apply(remove_spaces)
data_.drop(columns = ['Churn'])
test_ = pd.read_csv('test.csv')
test_['TotalSpent'] = test_['TotalSpent'].apply(remove_spaces)
#data_new, target_col, test_data_new = data_preprocessing(data_, test_, True)
#my_logistic_regression(data_new, target_col)

y_pred = stack(data_, test_)
#y_pred = boost(data_, target_col, cat_data, test_)
y_pred.to_csv('submisson_.csv', index = False)






