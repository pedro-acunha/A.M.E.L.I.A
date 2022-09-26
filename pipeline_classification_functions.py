import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis

# Metrics
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# Model selection
from sklearn.model_selection import cross_val_score,StratifiedKFold

#Plotting
import matplotlib.pyplot as plt

def KNNeighbours_clf(type_task, n_neighbors,threshold, X_train, X_test, y_train, y_test):
    """KNN Classifier for Binary Classification

    Args:
        type_task (string): Type of Classification task
        n_neighbors (int): Number of neighbors for distance-based classification
        threshold (float): Float between 0 and 1, to define threshold percentage
        X_train (DataFrame): DataFrame with training features
        X_test (DataFrame): DataFrame with testing features
        y_train (DataFrame): DataFrame with training target
        y_test (DataFrame): DataFrame with testing target

    Returns:
        DataFrame: DataFrame with prediction class for the test data
    """
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)

    knn_clf.fit(X_train, y_train)

    knn_pred_train = knn_clf.predict(X_train)

    if type_task == 'Supervised':
        knn_pred = knn_clf.predict(X_test)
        print('------------------------------------------------------------------------------')
        print('K-Nearest Neighbors Model\n ----------------------------------\n')
        print("KNN accuracy, train set : {:.4f}".format(accuracy_score(y_train, knn_pred_train)))
        print("KNN accuracy, test set : {:.4f}".format(accuracy_score(y_test, knn_pred)))
        print("KNN score, test set: Precision=",precision_recall_fscore_support(y_test, knn_pred, average='macro')[0],
              ", Recall= ",precision_recall_fscore_support(y_test, knn_pred, average='macro')[1],
              ", FScore= ",precision_recall_fscore_support(y_test, knn_pred, average='macro')[2],
              ", AUC= ",roc_auc_score(y_test, knn_clf.predict_proba(X_test)[:, 1]),
              ",\n Classification Report: \n", classification_report(y_test, knn_pred, digits=5))

        # Cross-validation
        knn_cv = cross_val_score(knn_clf, X_train, y_train, cv=5)
        print('\n KNN model cross-validation : {:.4f} +/- {:.4f}'.format(np.mean(knn_cv), np.std(knn_cv, ddof=1)))

        print('Confusion Matrix KNN\n')
        print(confusion_matrix(y_test, knn_pred))

        return knn_pred

    else:
        knn_pred_prob = knn_clf.predict_proba(X_test)[:,1]
        knn_pred_prob = pd.DataFrame(knn_pred_prob, columns=['qso_prob'])
        knn_pred_thre = np.where(knn_pred_prob > threshold, 1, 0)


        print('------------------------------------------------------------------------------')
        print('K-Nearest Neighbors Model with prob > ' + str(threshold) + '\n ----------------------------------\n')
        print("KNN accuracy, train set : {:.4f}".format(accuracy_score(y_train, knn_pred_train)))
        print("KNN accuracy, test set : {:.4f}".format(accuracy_score(y_test, knn_pred_thre)))
        print("KNN score, test set: Precision=",precision_recall_fscore_support(y_test, knn_pred_thre, average='macro')[0],
            ", Recall= ",precision_recall_fscore_support(y_test, knn_pred_thre, average='macro')[1],
            ", FScore= ",precision_recall_fscore_support(y_test, knn_pred_thre, average='macro')[2],
            ",\n Classification Report: \n", classification_report(y_test, knn_pred_thre, digits=5))

        print('Confusion Matrix KNN\n')
        print(confusion_matrix(y_test, knn_pred_thre))

        return knn_pred_thre

def RandomForest_clf(type_task, n_estimators, max_depth, threshold, X_train, X_test, y_train, y_test,plot_feat):
    """Random Forest Classifier for Binary Classification

    Args:
        type_task (string): Type of Classification task
        n_estimators (int): Number of estimators for RF
        max_depth (int): Number for max_depth
        threshold (float): Float between 0 and 1, to define threshold percentage
        X_train (DataFrame): DataFrame with training features
        X_test (DataFrame): DataFrame with testing features
        y_train (DataFrame): DataFrame with training target
        y_test (DataFrame): DataFrame with testing target

    Returns:
        DataFrame: DataFrame with prediction class for the test data

    """
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs =-1, random_state=24)
    rf_clf.fit(X_train, y_train)
    rf_pred_train = rf_clf.predict(X_train)

    if plot_feat == 'y':
        features = list(X_train.columns)
        features_rf = rf_clf.feature_importances_
        rf_feat = pd.DataFrame({'feature_importance': features_rf,
                      'feature_names': features}).sort_values(by=['feature_importance'],ascending=False)

        fig, ax = plt.subplots()
        ax.barh(np.arange(len(rf_feat['feature_names'])),rf_feat['feature_importance'],
                align='center',height=0.3)
        ax.set_yticks(np.arange(len(rf_feat['feature_names'])))
        ax.set_yticklabels(rf_feat['feature_names'])
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Feature importance')
        ax.set_title('Feature importance for Random Forest')

        rects = ax.patches

        # For each bar: Place a label
        for rect in rects:
            # Get X and Y placement of label from rect.
            x_value = rect.get_width()
            y_value = rect.get_y() + rect.get_height() / 2

            # Number of points between bar and label. Change to your liking.
            space = 5
            # Vertical alignment for positive values
            ha = 'left'

            # If value of bar is negative: Place label left of bar
            if x_value < 0:
                # Invert space to place label to the left
                space *= -1
                # Horizontally align label at right
                ha = 'right'

            # Use X value as label and format number with one decimal place
            label = "{:.3f}".format(x_value)

            # Create annotation
            plt.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(space, 0),          # Horizontally shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                va='center',                # Vertically center label
                ha=ha)                      # Horizontally align label differently for
                                            # positive and negative values.
    else:
        pass

    if type_task == 'Supervised':
        rf_pred = rf_clf.predict(X_test)

        print('------------------------------------------------------------------------------')
        print('Random Forest Model\n ----------------------------------\n')
        print("RF accuracy, train set : {:.4f}".format(accuracy_score(y_train, rf_pred_train)))
        print("RF accuracy, test set : {:.4f}".format(accuracy_score(y_test, rf_pred)))
        print("RF score, test set: Precision=",precision_recall_fscore_support(y_test, rf_pred, average='macro')[0],
              ", Recall= ",precision_recall_fscore_support(y_test, rf_pred, average='macro')[1],
              ", FScore= ",precision_recall_fscore_support(y_test, rf_pred, average='macro')[2],
              ", AUC= ",roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:, 1]),
              ",\n Classification Report: \n", classification_report(y_test, rf_pred, digits=5))

        # Cross-validation
        rf_cv = cross_val_score(rf_clf, X_train, y_train, cv=5)
        print('\n RF model cross-validation : {:.4f} +/- {:.4f}'.format(np.mean(rf_cv), np.std(rf_cv, ddof=1)))

        print('Confusion Matrix RF\n')
        print(confusion_matrix(y_test, rf_pred))
        return rf_pred

    else:
        rf_pred_prob = rf_clf.predict_proba(X_test)[:,1]
        rf_pred_prob = pd.DataFrame(rf_pred_prob, columns=['qso_prob'])
        rf_pred_thre = np.where(rf_pred_prob > threshold, 1, 0)

        print('------------------------------------------------------------------------------')
        print('Random Forest Model with prob > ' + str(threshold) + '\n ----------------------------------\n')
        print("RF accuracy, train set : {:.4f}".format(accuracy_score(y_train, rf_pred_train)))
        print("RF accuracy, test set : {:.4f}".format(accuracy_score(y_test, rf_pred_thre)))
        print("RF score, test set: Precision=",precision_recall_fscore_support(y_test, rf_pred_thre, average='macro')[0],
              ", Recall= ",precision_recall_fscore_support(y_test, rf_pred_thre, average='macro')[1],
              ", FScore= ",precision_recall_fscore_support(y_test, rf_pred_thre, average='macro')[2],
              ",\n Classification Report: \n", classification_report(y_test, rf_pred_thre, digits=5))

        print('Confusion Matrix RF\n')
        print(confusion_matrix(y_test, rf_pred_thre))

        return rf_pred_thre

def XGBoost_clf(type_task, n_estimators, threshold, X_train, X_test, y_train, y_test,plot_feat):
    """XGBoost Classifier for Binary Classification

    Args:
        type_task (string): Type of Classification task
        n_estimators (int): Number of estimators to use
        threshold (float): Float between 0 and 1, to define threshold percentage
        X_train (DataFrame): DataFrame with training features
        X_test (DataFrame): DataFrame with testing features
        y_train (DataFrame): DataFrame with training target
        y_test (DataFrame): DataFrame with testing target

    Returns:
        DataFrame: DataFrame with prediction class for the test data

    """
    xgb_clf = xgb.XGBClassifier(n_estimators=n_estimators,n_jobs=-1,tree_method='exact',random_state=24)
    xgb_clf.fit(X_train, y_train)
    xgb_pred_train = xgb_clf.predict(X_train)

    if plot_feat == 'y':
        from xgboost import plot_importance
        plot_importance(xgb_clf,title='Feature importance for XGBoost')
    else:
        pass

    if type_task == 'Supervised':
        xgb_pred = xgb_clf.predict(X_test)

        print('------------------------------------------------------------------------------')
        print('XGBoost Model\n ----------------------------------\n')
        print("XGBoost accuracy, train set : {:.4f}".format(accuracy_score(y_train, xgb_pred_train)))
        print("XGBoost accuracy, test set : {:.4f}".format(accuracy_score(y_test, xgb_pred)))
        print("XGBoost score, test set: Precision=",precision_recall_fscore_support(y_test, xgb_pred, average='macro')[0],
              ", Recall= ",precision_recall_fscore_support(y_test, xgb_pred, average='macro')[1],
              ", FScore= ",precision_recall_fscore_support(y_test, xgb_pred, average='macro')[2],
              ", AUC= ",roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1]),
              ",\n Classification Report: \n", classification_report(y_test, xgb_pred, digits=5))

        # Cross-validation
        xgb_cv = cross_val_score(xgb_clf, X_train, y_train, cv=10)
        print('\n XGBoost model cross-validation : {:.4f} +/- {:.4f}'.format(np.mean(xgb_cv), np.std(xgb_cv, ddof=1)))

        print('Confusion Matrix XGBoost\n')
        print(confusion_matrix(y_test, xgb_pred))

        return xgb_pred

    else:
        xgb_pred_prob = xgb_clf.predict_proba(X_test)[:,1]
        xgb_pred_prob = pd.DataFrame(xgb_pred_prob, columns=['qso_prob'])
        xgb_pred_thre = np.where(xgb_pred_prob > threshold, 1, 0)

        print('------------------------------------------------------------------------------')
        print('XGBoost Model with prob > ' + str(threshold) + '\n ----------------------------------\n')
        print("XGBoost accuracy, train set : {:.4f}".format(accuracy_score(y_train, xgb_pred_train)))
        print("XGBoost accuracy, test set : {:.4f}".format(accuracy_score(y_test, xgb_pred_thre)))
        print("XGBoost score, test set: Precision=",precision_recall_fscore_support(y_test, xgb_pred_thre, average='macro')[0],
              ", Recall= ",precision_recall_fscore_support(y_test, xgb_pred_thre, average='macro')[1],
              ", FScore= ",precision_recall_fscore_support(y_test, xgb_pred_thre, average='macro')[2],
              ",\n Classification Report: \n", classification_report(y_test, xgb_pred_thre, digits=5))

        print('Confusion Matrix XGBoost\n')
        print(confusion_matrix(y_test, xgb_pred_thre))

        return xgb_pred_thre

def CatBoost_clf(type_task, n_estimators, threshold, X_train, X_test, y_train, y_test,plot_feat):
    """CatBoost Classifier for Binary Classification

    Args:
        type_task (string): Type of Classification task
        n_estimators (int): Number of estimators to use
        threshold (float): Float between 0 and 1, to define threshold percentage
        X_train (DataFrame): DataFrame with training features
        X_test (DataFrame): DataFrame with testing features
        y_train (DataFrame): DataFrame with training target
        y_test (DataFrame): DataFrame with testing target

    Returns:
        DataFrame: DataFrame with prediction class for the test data
    """
    cb_clf = CatBoostClassifier(logging_level='Silent',thread_count=-1, n_estimators=n_estimators)
    cb_clf.fit(X_train, y_train)

    cb_pred_train = cb_clf.predict(X_train)

    if plot_feat == 'y':
        features = list(X_train.columns)
        features_cb = cb_clf.get_feature_importance(Pool(X_train, y_train))

        cb_feat = pd.DataFrame({'feature_importance': features_cb,
                      'feature_names': features}).sort_values(by=['feature_importance'],ascending=False)

        fig, ax = plt.subplots()
        ax.barh(np.arange(len(cb_feat['feature_names'])),cb_feat['feature_importance'],
                align='center',height=0.3)
        ax.set_yticks(np.arange(len(cb_feat['feature_names'])))
        ax.set_yticklabels(cb_feat['feature_names'])
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Feature importance')
        ax.set_title('Feature importance for CatBoost')

        rects = ax.patches

        # For each bar: Place a label
        for rect in rects:
            # Get X and Y placement of label from rect.
            x_value = rect.get_width()
            y_value = rect.get_y() + rect.get_height() / 2

            # Number of points between bar and label. Change to your liking.
            space = 5
            # Vertical alignment for positive values
            ha = 'left'

            # If value of bar is negative: Place label left of bar
            if x_value < 0:
                # Invert space to place label to the left
                space *= -1
                # Horizontally align label at right
                ha = 'right'

            # Use X value as label and format number with one decimal place
            label = "{:.0f}".format(x_value)

            # Create annotation
            plt.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(space, 0),          # Horizontally shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                va='center',                # Vertically center label
                ha=ha)                      # Horizontally align label differently for
                                            # positive and negative values.
    else:
        pass

    if type_task == 'Supervised':
        cb_pred = cb_clf.predict(X_test)

        print('------------------------------------------------------------------------------')
        print('CatBoost Model\n ----------------------------------\n')
        print("CatBoost accuracy, train set : {:.4f}".format(accuracy_score(y_train, cb_pred_train)))
        print("CatBoost accuracy, test set : {:.4f}".format(accuracy_score(y_test, cb_pred)))
        print("CatBoost score, test set: Precision=",precision_recall_fscore_support(y_test, cb_pred, average='macro')[0],
              ", Recall= ",precision_recall_fscore_support(y_test, cb_pred, average='macro')[1],
              ", FScore= ",precision_recall_fscore_support(y_test, cb_pred, average='macro')[2],
              ", AUC= ",roc_auc_score(y_test, cb_clf.predict_proba(X_test)[:, 1]),
              ",\n Classification Report: \n", classification_report(y_test, cb_pred, digits=5))

        # Cross-validation
        #cb_cv = cross_val_score(cb_clf, X_train, y_train, cv=10)
        #print('\n CatBoost model cross-validation : {:.4f} +/- {:.4f}'.format(np.mean(cb_cv), np.std(cb_cv, ddof=1)))

        print('Confusion Matrix CatBoost\n')
        print(confusion_matrix(y_test, cb_pred))

        return cb_pred

    else:
        cb_pred_prob = cb_clf.predict_proba(X_test)[:,1]
        cb_pred_prob = pd.DataFrame(cb_pred_prob, columns=['qso_prob'])
        cb_pred_thre = np.where(cb_pred_prob > threshold, 1, 0)

        print('------------------------------------------------------------------------------')
        print('CatBoost Model with prob > ' + str(threshold) + '\n ----------------------------------\n')
        print("CatBoost accuracy, train set : {:.4f}".format(accuracy_score(y_train, cb_pred_train)))
        print("CatBoost accuracy, test set : {:.4f}".format(accuracy_score(y_test, cb_pred_thre)))
        print("CatBoost score, test set: Precision=",precision_recall_fscore_support(y_test, cb_pred_thre, average='macro')[0],
              ", Recall= ",precision_recall_fscore_support(y_test, cb_pred_thre, average='macro')[1],
              ", FScore= ",precision_recall_fscore_support(y_test, cb_pred_thre, average='macro')[2],
              ",\n Classification Report: \n", classification_report(y_test, cb_pred_thre, digits=5))

        print('Confusion Matrix CatBoost\n')
        print(confusion_matrix(y_test, cb_pred_thre))

        return cb_pred_thre

def LighGBM_clf(type_task, n_estimators, threshold, X_train, X_test, y_train, y_test,plot_feat):
    """LightGBM Classifier for Binary Classification

    Args:
        type_task (string): Type of Classification Task
        n_estimators (int): Number of estimators to use
        threshold (float): Float between 0 and 1, to define threshold percentage
        X_train (DataFrame): DataFrame with training features
        X_test (DataFrame): DataFrame with testing features
        y_train (DataFrame): DataFrame with training target
        y_test (DataFrame): DataFrame with testing target

    Returns:
        DataFrame: DataFrame with prediction class for the test data
    """
    lgb_clf = lgb.LGBMClassifier(n_jobs = -1,n_estimators=n_estimators)
    lgb_clf.fit(X_train, y_train)

    lgb_pred_train = lgb_clf.predict(X_train)

    if plot_feat == 'y':
        lgb.plot_importance(lgb_clf,title='Feature importance for LightGBM')
    else:
        pass

    if type_task == 'Supervised':
        lgb_pred = lgb_clf.predict(X_test)

        print('------------------------------------------------------------------------------')
        print('LightGBM Model\n ----------------------------------\n')
        print("LightGBM accuracy, train set : {:.4f}".format(accuracy_score(y_train, lgb_pred_train)))
        print("LightGBM accuracy, test set : {:.4f}".format(accuracy_score(y_test, lgb_pred)))
        print("LightGBM score, test set: Precision=",precision_recall_fscore_support(y_test, lgb_pred, average='macro')[0],
              ", Recall= ",precision_recall_fscore_support(y_test, lgb_pred, average='macro')[1],
              ", FScore= ",precision_recall_fscore_support(y_test, lgb_pred, average='macro')[2],
              ", AUC= ",roc_auc_score(y_test, lgb_clf.predict_proba(X_test)[:, 1]),
              ",\n Classification Report: \n", classification_report(y_test, lgb_pred, digits=5))

        # Cross-validation
        lgb_cv = cross_val_score(lgb_clf, X_train, y_train, cv=10)
        print('\n LightGBM model cross-validation : {:.4f} +/- {:.4f}'.format(np.mean(lgb_cv), np.std(lgb_cv, ddof=1)))

        print('Confusion Matrix LightGBM\n')
        print(confusion_matrix(y_test, lgb_pred))

        return lgb_pred

    else:
        lgb_pred_prob = lgb_clf.predict_proba(X_test)[:,1]
        lgb_pred_prob = pd.DataFrame(lgb_pred_prob, columns=['qso_prob'])
        lgb_pred_thre = np.where(lgb_pred_prob > threshold, 1, 0)

        print('------------------------------------------------------------------------------')
        print('LightGBM Model with prob > ' + str(threshold) + '\n ----------------------------------\n')
        print("LightGBM accuracy, train set : {:.4f}".format(accuracy_score(y_train, lgb_pred_train)))
        print("LightGBM accuracy, test set : {:.4f}".format(accuracy_score(y_test, lgb_pred_thre)))
        print("LightGBM score, test set: Precision=",precision_recall_fscore_support(y_test, lgb_pred_thre, average='macro')[0],
              ", Recall= ",precision_recall_fscore_support(y_test, lgb_pred_thre, average='macro')[1],
              ", FScore= ",precision_recall_fscore_support(y_test, lgb_pred_thre, average='macro')[2],
              ",\n Classification Report: \n", classification_report(y_test, lgb_pred_thre, digits=5))

        print('Confusion Matrix LightGBM\n')
        print(confusion_matrix(y_test, lgb_pred_thre))

        return lgb_pred_thre

def MLP_clf(task_type, hidden_layer_sizes, threshold, X_train, X_test, y_train, y_test):
    """Multi-Layer Perceptron Classifier for Binary Classification

    Args:
        task_type (string): Type of Classification Task
        hidden_layer_sizes (list): Number of hidden layer
        threshold (float): Float between 0 and 1, to define threshold percentage
        X_train (DataFrame): DataFrame with training features
        X_test (DataFrame): DataFrame with testing features
        y_train (DataFrame): DataFrame with training target
        y_test (DataFrame): DataFrame with testing target

    Returns:
        DataFrame: DataFrame with prediction class for the test data
    """
    mlp_clf = MLPClassifier(activation='relu', solver='adam', alpha=0.0001, hidden_layer_sizes=hidden_layer_sizes,
                                            max_iter=500,learning_rate='constant',random_state=24)
    mlp_clf.fit(X_train, y_train)

    mlp_pred_train = mlp_clf.predict(X_train)

    if task_type == 'Supervised':
        mlp_pred = mlp_clf.predict(X_test)

        print('------------------------------------------------------------------------------')
        print('MLP Model\n ----------------------------------\n')
        print("MLP accuracy, train set : {:.4f}".format(accuracy_score(y_train, mlp_pred_train)))
        print("MLP accuracy, test set : {:.4f}".format(accuracy_score(y_test, mlp_pred)))
        print("MLP score, test set: Precision=",precision_recall_fscore_support(y_test, mlp_pred, average='macro')[0],
              ", Recall= ",precision_recall_fscore_support(y_test, mlp_pred, average='macro')[1],
              ", FScore= ",precision_recall_fscore_support(y_test, mlp_pred, average='macro')[2],
              ", AUC= ",roc_auc_score(y_test, mlp_clf.predict_proba(X_test)[:, 1]),
              ",\n Classification Report: \n", classification_report(y_test, mlp_pred, digits=5))

        # Cross-validation
        mlp_cv = cross_val_score(mlp_clf, X_train, y_train, cv=10)
        print('\n MLP model cross-validation : {:.4f} +/- {:.4f}'.format(np.mean(mlp_cv), np.std(mlp_cv, ddof=1)))

        print('Confusion Matrix MLP\n')
        print(confusion_matrix(y_test, mlp_pred))

        return mlp_pred

    else:
        mlp_pred_prob = mlp_clf.predict_proba(X_test)[:,1]
        mlp_pred_prob = pd.DataFrame(mlp_pred_prob, columns=['qso_prob'])
        mlp_pred_thre = np.where(mlp_pred_prob > threshold, 1, 0)

        print('------------------------------------------------------------------------------')
        print('MLP Model with prob > ' + str(threshold) + '\n ----------------------------------\n')
        print("MLP accuracy, train set : {:.4f}".format(accuracy_score(y_train, mlp_pred_train)))
        print("MLP accuracy, test set : {:.4f}".format(accuracy_score(y_test, mlp_pred_thre)))
        print("MLP score, test set: Precision=",precision_recall_fscore_support(y_test, mlp_pred_thre, average='macro')[0],
              ", Recall= ",precision_recall_fscore_support(y_test, mlp_pred_thre, average='macro')[1],
              ", FScore= ",precision_recall_fscore_support(y_test, mlp_pred_thre, average='macro')[2],
              ",\n Classification Report: \n", classification_report(y_test, mlp_pred_thre, digits=5))

        print('Confusion Matrix MLP\n')
        print(confusion_matrix(y_test, mlp_pred_thre))

        return mlp_pred_thre

def Generalised_Stacking(threshold, list_alg,X_train,X_test,y_train,y_test, path_test_data, data_to_extract):
    """Performes Generalised Stacking with MLP, RF, LGBM, CatBoost, XGBoost and KNN

    Args:
        threshold (float): Threshold to be used, from 0 to 1
        list_alg (list): List of algorithms to be used in GS approach
        X_train (DataFrame): DataFrame with training features
        X_test (DataFrame): DataFrame with testing features
        y_train (DataFrame): DataFrame with training target
        y_test (DataFrame): DataFrame with testing target
        path_test_data (string): Path for the data being classified
        data_to_extract (string): Type of data to extract: misclassifications or correct classifications

    Returns:
        DataFrame: DataFrames with index, probabilities for each algoritms and data for analysis
    """
    print('------------------------------------------------------------------------------')
    print('Starting Generalized Stacking with Meta Learning\n ----------------------------------\n')

    models = {'mlp': MLPClassifier(activation='relu', solver='adam', alpha=0.0001, hidden_layer_sizes=(50,100,50), max_iter=500, random_state=24),
    'rf':RandomForestClassifier(n_estimators=50,n_jobs=-1,random_state=24),
    'lgbm':lgb.LGBMClassifier(n_jobs = -1,n_estimators=50,random_state=24),
    'cb':CatBoostClassifier(logging_level='Silent',thread_count=-1, n_estimators=500),
    'knn': KNeighborsClassifier(n_neighbors=12),
    'xgb':xgb.XGBClassifier(n_estimators=50,n_jobs=-1,tree_method='exact',random_state=24)}

    def train_kf_oof(clf,X_train,y_train,X_test,nkf):
        oof_train = pd.Series([0]).repeat(len(X_train))
        preds = pd.Series([0]).repeat(len(X_test))

        skf = StratifiedKFold(n_splits=nkf, shuffle=True, random_state=24)
        for train_index, oof_index in skf.split(X_train, y_train):
            X_tr = X_train.iloc[train_index]
            y_tr = y_train.iloc[train_index]
            X_te = X_train.iloc[oof_index]

            # detect LGBM and use specific option for it
            if str(type(clf).__name__) == 'LGBMClassifier':
                clf.fit(X_tr, y_tr,verbose=False)
            else:
                clf.fit(X_tr, y_tr)

            # save the OOF class probability predictions and test set class probability predictions
            oof_train.iloc[oof_index] = clf.predict_proba(X_te)[:,1]
            preds += clf.predict_proba(X_test)[:,1]/skf.n_splits

        oof_train.index = X_train.index
        preds.index = X_test.index
        return oof_train, preds

    train_meta = pd.DataFrame(index=X_train.index,columns=[*list_alg])
    test_meta = pd.DataFrame(index=X_test.index, columns=[*list_alg])

    acc_list = []

    for i, model_name in enumerate([*list_alg]):
        if i==0: print('Starting OOF predictions for all models. \n')
        train_meta[model_name], test_meta[model_name] = train_kf_oof(models[model_name],X_train,y_train,X_test,nkf=5)

        # show accuracy for each learner using strat k-fold (test set)
        acc = np.round(accuracy_score(y_test,np.where(test_meta[model_name] > threshold, 1, 0)),4)
        print(str(type(models[model_name]).__name__)+' acc: '+str(np.round(acc,4)))
        acc_list.append(acc)

    # let's take a look at the features for training our meta learner:
    test_meta['average'] = test_meta[[*list_alg]].mean(axis=1)
    train_meta['average'] = train_meta[[*list_alg]].mean(axis=1)
    test_meta['Target'] = y_test.values
    train_meta['Target'] = y_train.values
    print('_________________________________')
    print(train_meta.round(2))

    # look at cases where the predictions are incorrect
    print('Incorrect classifications:')
    test_meta_wrong = test_meta[test_meta['Target'] != np.where(test_meta['average']> threshold, 1, 0)]
    print(test_meta_wrong.round(2).head(),'\n')
    print('_________________________________')
    print('Incorrect classifications were individual learners strongly disagree:')
    test_meta_wrong2 = test_meta_wrong[test_meta_wrong[[*list_alg]].max(1) - test_meta_wrong[[*list_alg]].min(1) >= threshold]
    print(test_meta_wrong2.round(2).head())
    print('_________________________________')

    #apply meta-learner and make new predictions
    print('------------------------------------------------------------------------------')
    print('Starting Meta-learners, training test -> previous predictions \n ----------------------------------\n')
    models2 = {'MLP': MLPClassifier(activation='relu', solver='adam', alpha=0.0001, hidden_layer_sizes=(50,100,50), max_iter=500, random_state=24),
               'MLP2': MLPClassifier(activation='logistic', solver='adam', alpha=0.0001, hidden_layer_sizes=(100,100), max_iter=500, random_state=24),
               'Catboost': CatBoostClassifier(iterations=50,logging_level='Silent',thread_count=-1),
               'LGBM': lgb.LGBMClassifier(n_jobs = -1,n_estimators=50,random_state=24),
               'XGBoost': xgb.XGBClassifier(n_estimators=50,n_jobs=-1,tree_method='exact',random_state=24)
                        }
    cols2 = [*list_alg]
    cols2.append('average')

    acc_previous_best = max(acc_list)
    print('_________________________________')
    print('Meta-learners performance:')

    columns=[*models2]
    data=np.zeros((len(test_meta), len(columns) ))
    for i, model_name in enumerate([*models2]):
        clf = models2[model_name]
        clf.fit(train_meta[cols2], y_train)
        preds_meta = clf.predict(test_meta[cols2])
        preds_proba_meta = clf.predict_proba(test_meta[cols2])[:,1]
        data[:,i]=(preds_proba_meta)
        acc = accuracy_score(y_test,preds_meta)
        print(str(np.round(acc,4))+' ('+str(np.round(acc-acc_previous_best,4))+' change) '+str(type(clf).__name__))


    meta_learn = pd.DataFrame(data, columns=columns)
    meta_learn['meta_average'] = meta_learn.mean(axis=1)
    test_meta['meta_label'] = np.where(meta_learn['meta_average'] > threshold, 1, 0)
    y_test.index = test_meta.index
    test_meta['Target'] = y_test

    test_meta_corrected = test_meta[(test_meta['Target'] != np.where(test_meta['average'] >= threshold, 1, 0)) &
                                           (test_meta['meta_label'] == test_meta['Target'])]

    print('_________________________________')
    print('Corrected misclassifications: ', len(test_meta_corrected),'out of',len(test_meta_wrong),'\n')
    print('_________________________________')

    print('DataFrames with corrected classifications:')
    print(test_meta_corrected[test_meta_corrected['Target'] == 1].round(2))
    print(test_meta_corrected[test_meta_corrected['Target'] == 0].round(2))

    print('------------------------------------------------------------------------------')
    print('Generalised Stacking with Meta Learners\n ----------------------------------\n')
    print("GS with ML accuracy, test set : {:.4f}".format(accuracy_score(y_test, test_meta['meta_label'])))
    print("GS with ML score, test set: Precision=",precision_recall_fscore_support(y_test, test_meta['meta_label'], average='macro')[0],
      ", Recall= ",precision_recall_fscore_support(y_test, test_meta['meta_label'], average='macro')[1],
      ", FScore= ",precision_recall_fscore_support(y_test, test_meta['meta_label'], average='macro')[2],
      ",\n Classification Report: \n", classification_report(y_test, test_meta['meta_label'], digits=5))

    print('Confusion Matrix GS with ML\n')
    print(confusion_matrix(y_test, test_meta['meta_label']))



    print('\n Preparing data to be extracted.\n ')

    if data_to_extract == 'Correct':
        df_wrong = test_meta_wrong.copy()
        df_wrong = df_wrong.drop(test_meta_corrected.index.tolist(),axis='index')
        df_right = X_test.drop(df_wrong.index.tolist())
        index_list = df_right.index
        df_check = pd.read_csv(path_test_data)
        df_check = df_check.iloc[index_list]
        return index_list, test_meta, df_check

    elif data_to_extract == 'Incorrect':
        df_wrong = test_meta_wrong.copy()
        df_wrong = df_wrong.drop(test_meta_corrected.index.tolist(),axis='index')
        index_list = X_test.iloc[df_wrong.index].index
        df_check = pd.read_csv(path_test_data)
        df_check = df_check.iloc[index_list]
        return index_list, test_meta, df_check

    else:
        return test_meta




def ABTestFeatures(X_train, X_test, y_train, y_test):
    """ Performes an AB Test to evaluate feature relevance for model performance

    Args:
        x_train (DataFrame): DataFrame with trainning data, input data
        x_test (DataFrame): DataFrame with testing data, input data
        y_train (DataFrame): DataFrame with trainning data, output data
        y_test (DataFrame): DataFrame with trainning data, output data
    """
    models = {'MLP': MLPClassifier(activation='relu', solver='adam', alpha=0.0001, hidden_layer_sizes=(50,100,50), max_iter=500, random_state=24),
    'RandomForest':RandomForestClassifier(n_estimators=500,n_jobs=-1,random_state=24),
    'lgbm':lgb.LGBMClassifier(n_jobs = -1,n_estimators=500),
    'catboost':CatBoostClassifier(logging_level='Silent',thread_count=-1, n_estimators=500),
    'knn': KNeighborsClassifier(n_neighbors=12),
    'XGBoost':xgb.XGBClassifier(n_estimators=500,n_jobs=-1,tree_method='exact',random_state=24)}

    metrics_pred = np.zeros((len(models),3))
    for i, model_name in enumerate([*models]):
        if i==0: print('Starting baseline predictions for comparision. \n')
        clf = models[model_name].fit(X_train, y_train)
        clf_pred = clf.predict(X_test)
        for t in range(0,3):
            metrics_pred[i,t] = precision_recall_fscore_support(y_test, clf_pred, average='macro')[t]

    print('All data: ',X_train.columns)

    print("Metrics to compare by:\n Precision:", np.mean(metrics_pred[:,0]), " Recall: ", np.mean(metrics_pred[:,1]), " Precision: ", np.mean(metrics_pred[:,2]))

    feat_list = X_train.columns.to_list()

    print("\nTesting model performance with only SDSS broadband magnitudes")

    list_col=[]
    check = ['modelMag_u', 'modelMag_g','modelMag_r','modelMag_i','modelMag_z']
    if any(n in check for n in feat_list):
        list_col = [e for e in feat_list if e in check]


    print('SDSS data: ',list_col)

    X_train1 = X_train[list_col]
    X_test1 = X_test[list_col]

    metrics_pred1 = np.zeros((len(models),3))
    for i, model_name in enumerate([*models]):
        if i==0: print('Starting predictions with SDSS broadband magnitudes. \n')
        clf = models[model_name].fit(X_train1, y_train)
        clf_pred = clf.predict(X_test1)
        for t in range(0,3):
            metrics_pred1[i,t] = precision_recall_fscore_support(y_test, clf_pred, average='macro')[t]

    print("Comparision with baseline predictions:\n Precision:", np.mean(metrics_pred1[:,0]),
          " (",np.mean(metrics_pred1[:,0])-np.mean(metrics_pred[:,0]),
          ")\n Recall: ", np.mean(metrics_pred1[:,1]),
          " (",np.mean(metrics_pred1[:,1])-np.mean(metrics_pred[:,1]),
          ")\n F1-Score: ", np.mean(metrics_pred1[:,2]),
          " (",np.mean(metrics_pred1[:,2])-np.mean(metrics_pred[:,2]),")")


    print("\nTesting model performance with SDSS broadband magnitudes and colours")

    list_col=[]
    check = ['w1mpro','w2mpro','w3mpro','w4mpro']
    if any(n in check for n in feat_list):
        list_col = [e for e in feat_list if e not in check]

    print('SDSS data: ',list_col)


    X_train2 = X_train[list_col]
    X_test2 = X_test[list_col]

    metrics_pred2 = np.zeros((len(models),3))
    for i, model_name in enumerate([*models]):
        if i==0: print('Starting predictions with SDSS broadband magnitudes and colours. \n')
        clf = models[model_name].fit(X_train2, y_train)
        clf_pred = clf.predict(X_test2)
        for t in range(0,3):
            metrics_pred2[i,t] = precision_recall_fscore_support(y_test, clf_pred, average='macro')[t]

    print("Comparision with baseline predictions:\n Precision:", np.mean(metrics_pred2[:,0]),
          " (",np.mean(metrics_pred2[:,0])-np.mean(metrics_pred[:,0]),
          ")\n Recall: ", np.mean(metrics_pred1[:,1]),
          " (",np.mean(metrics_pred2[:,1])-np.mean(metrics_pred[:,1]),
          ")\n F1-Score: ", np.mean(metrics_pred1[:,2]),
          " (",np.mean(metrics_pred2[:,2])-np.mean(metrics_pred[:,2]),")")

    print("\nTesting model performance with only WISE broadband magnitudes")

    list_col=[]
    check = ['w1mpro','w2mpro','w3mpro','w4mpro']
    if any(n in check for n in feat_list):
        list_col = [e for e in feat_list if e in check]

    print('WISE data: ',list_col)

    X_train3 = X_train[list_col]
    X_test3 = X_test[list_col]

    metrics_pred3 = np.zeros((len(models),3))
    for i, model_name in enumerate([*models]):
        if i==0: print('Starting predictions with WISE data. \n')
        clf = models[model_name].fit(X_train3, y_train)
        clf_pred = clf.predict(X_test3)
        for t in range(0,3):
            metrics_pred3[i,t] = precision_recall_fscore_support(y_test, clf_pred, average='macro')[t]

    print("Comparision with baseline predictions:\n Precision:", np.mean(metrics_pred3[:,0]),
          " (",np.mean(metrics_pred3[:,0])-np.mean(metrics_pred[:,0]),
          ")\n Recall: ", np.mean(metrics_pred3[:,1]),
          " (",np.mean(metrics_pred3[:,1])-np.mean(metrics_pred[:,1]),
          ")\n F1-Score: ", np.mean(metrics_pred3[:,2]),
          " (",np.mean(metrics_pred3[:,2])-np.mean(metrics_pred[:,2]),")")
