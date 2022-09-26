import pandas as pd
from pipeline_data_preprocessing import *
from pipeline_classification_functions import *
from sklearn.impute import KNNImputer

# Metrics
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

#Graphs 
import seaborn as sns
sns.set_style("white")
sns.set(font_scale = 2)


#################################################################################################
# Preprocessing tasks

df_qso = pd.read_csv('./data/SDSS_WISE_QSO2_Alexandroff.csv')
df_qso = df_qso[['modelMag_u', 'modelMag_g','modelMag_r','modelMag_i','modelMag_z', 'w1mpro', 'w2mpro', 'w3mpro', 'w4mpro']].copy()
df_qso.columns = ['u','g','r','i','z','w1','w2','w3','w4']
df_qso['class']='QSO2'
df_qso['redshift']= pd.read_excel('./data/sdss_boss_lines_results.xlsx')[['z']]

df_geral = pd.read_csv('./data/SDSS_Galaxy_Highz_PCunha.csv')
df_gal = df_geral[[ 'modelMag_u', 'modelMag_g','modelMag_r','modelMag_i','modelMag_z', 'w1mpro', 'w2mpro', 'w3mpro', 'w4mpro','z']].copy().sample(len(df_qso), random_state=0)
df_gal.columns = ['u','g','r','i','z','w1','w2','w3','w4','redshift']
df_gal['class'] = 'Not QSO2'

df = pd.concat([df_qso, df_gal])
df = df.reset_index(drop=True)

df_test = df_geral.drop(index=list(df_gal.index))
df_test= df_test[[ 'modelMag_u', 'modelMag_g','modelMag_r','modelMag_i','modelMag_z', 'w1mpro', 'w2mpro', 'w3mpro', 'w4mpro','z']].copy()
df_test.columns = ['u','g','r','i','z','w1','w2','w3','w4','redshift']
df_test['class'] = 'Not QSO2'

def create_colours(data, features):
    """Create dataframe with colour-colour data. Return an Array and list of colours.

    Args:
        data (DataFrame): Dataframe with photometric data
        features (list): List of features from photometric data
    """
    N = len(data)
    F = len(features)
    n=0
    for i in np.linspace(1,len(features),len(features),dtype=int):
    	n = n + (i-1)

    df_features = np.zeros((N, n))
    y=0
    lista=[]
    for z in np.linspace(0,F,F,dtype=int):
    	for x in np.linspace(1,F-1,F-1,dtype=int):
        	if z!=x and z<x:
        		df_features[:,y] = data[features[z]] - data[features[x]]
        		y+=1
        		lista += [features[z]+'-'+features[x]]
        	else:
           		pass
    df_colours = pd.DataFrame(df_features,columns = lista)
    return df_colours


df_colours=create_colours(df,['u','g','r','i','z','w1','w2','w3','w4'])
df_colours_test=create_colours(df_test,['u','g','r','i','z','w1','w2','w3','w4'])

df_train = pd.concat([df, df_colours.reindex(df.index)], axis=1, sort=False)
df_test = pd.concat([df_test, df_colours_test.reindex(df_test.index)], axis=1, sort=False)


targets= ['Target']
df_train['Target'] = pd.Series(np.where(df_train['class'].values == 'QSO2', 1, 0),df_train.index)
df_test['Target'] = pd.Series(np.where(df_test['class'].values == 'QSO2', 1, 0),df_test.index)

sns.histplot(df_test, x='redshift',color='#EF4026',fill=True, hue='Target').set_title('Redshift distribution in Test set', fontsize=20)

features = df_train.columns.values.tolist()

features.remove('class')
features.remove('Target')
features.remove('redshift')


X_train = df_train[features]
X_test = df_test[features]

y_train = pd.DataFrame(df_train['Target'], columns=['Target'])
y_test = pd.DataFrame(df_test['Target'], columns=['Target'])

scaler = MinMaxScaler()
X_train = scaler.fit(X_train).transform(X_train)
X_train = pd.DataFrame(X_train, columns=features)

X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test, columns=features)



print('\n------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------')
print('Starting classification.\n')
print('------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------\n')

task_type = 'Semi-Supervised'
path_test_data = './data/SDSS_Galaxy_Highz_PCunha.csv'
data_to_extract = 'Incorrect'
plot_feat='n'
scaling_method = 'Robust'


knn_clf = KNNeighbours_clf(task_type, 12, 0.9, X_train, X_test, y_train, y_test)

rf_clf = RandomForest_clf(task_type, 50, 10, 0.9, X_train, X_test, y_train, y_test,plot_feat)

xgb_clf = XGBoost_clf(task_type, 50, 0.9, X_train, X_test, y_train, y_test,plot_feat)

cb_clf = CatBoost_clf(task_type, 50,0.9, X_train, X_test, y_train, y_test,plot_feat)

lgb_clf = LighGBM_clf(task_type, 50, 0.9, X_train, X_test, y_train, y_test,plot_feat)

mlp_clf = MLP_clf(task_type, (50,100,50), 0.9, X_train, X_test, y_train, y_test)

list_alg=['knn','rf','cb','xgb','lgbm','mlp']

index_list, test_proba_meta, df_QSO = Generalised_Stacking(0.8,list_alg, X_train, X_test, y_train, y_test,path_test_data, data_to_extract )

# Write file with QSO2 candidates info
df_QSO.to_csv(r'QSO2_cand_highz_thr80_photo.csv',index=None)
