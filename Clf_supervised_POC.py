from pipeline_data_preprocessing import *
from pipeline_classification_functions import *

path_data_train1 = './data/SDSS_WISE_QSO2_Alexandroff.csv'
path_data_train2 = './data/SDSS_Galaxy_Highz_PCunha.csv'
path_data_test = '' # Can be left empty for supervised tasks
list_features = ['modelMag_u', 'modelMag_g','modelMag_r','modelMag_i','modelMag_z','w1mpro','w2mpro','w3mpro','w4mpro']
list_add_relation = []
target_variable1 = 'QSO2'
target_variable2 = 'Galaxy'
target_variable3 = 'QSO2'
task_type = 'Supervised'
scaling_method = 'Robust'
data_to_extract = ''
plot_feat = 'n' #answer 'y' or 'n'
list_alg=['knn','rf','cb','xgb','lgbm','mlp']

X_train, X_test, y_train, y_test = DataToClassifier(task_type, scaling_method, path_data_train1, path_data_train2, 
                                                    path_data_test, list_features, list_add_relation,
                                                    target_variable1, target_variable2, target_variable3)


knn_clf = KNNeighbours_clf(task_type, 12, 0.8, X_train, X_test, y_train, y_test)

rf_clf = RandomForest_clf(task_type, 50, 2, 0.8, X_train, X_test, y_train, y_test,plot_feat)

xgb_clf = XGBoost_clf(task_type, 50, 0.8, X_train, X_test, y_train, y_test,plot_feat)

cb_clf = CatBoost_clf(task_type, 50,0.8, X_train, X_test, y_train, y_test,plot_feat)

lgb_clf = LighGBM_clf(task_type, 50, 0.8, X_train, X_test, y_train, y_test,plot_feat)

mlp_clf = MLP_clf(task_type, (50,100,50), 0.8, X_train, X_test, y_train, y_test)

test_proba_meta = Generalised_Stacking(0.8,list_alg, X_train, X_test, y_train, y_test, path_data_test, data_to_extract)

ABTestFeatures(X_train, X_test, y_train, y_test)
