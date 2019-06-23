import time, datetime, uuid, os, random
import numpy as np
import pandas as pd
from skfeature.utility.construct_W import construct_W
from skfeature.function.similarity_based.lap_score import lap_score
from skfeature.function.similarity_based.SPEC import spec
from skfeature.function.similarity_based.fisher_score import fisher_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import xgboost as xgb

config = {
    "best_select_features_size": [25,26,27,28,29,30,31,32,33,34,35],
    "random_states" :[32,41,45,52,65,72,96,97,112,114,128,142],
    "save": True
}

def filter_feature_selection(X,y,method='fisher_score',k_best_features=10,W=None):
    '''

    :param data: Dataframe
    :param maximium_features: Int
    :return: filtered data with selected features
    '''
    try:

        methods = {
            # supervised methods
            "fisher_score": lambda X,y: fisher_score(X.values,y.values.reshape(-1)),
            # unsupervised methods
            "SPEC": lambda X,y: spec(X.values,W=W),
            "Laplacian_score": lambda X,y: lap_score(X.values,W=W),
        }

        selected_features = methods[method](X,y)
        idx = np.argsort(selected_features)[-1 * k_best_features:]
        selected_features_names = X.iloc[:, idx].columns
        selected_features_names = selected_features_names[-1 * k_best_features:]
        return selected_features_names

    except Exception as e:
        raise Exception('Error in Feature Selection method ' + e)


def warpper_feature_selection(X,y,method="sfs",k_best_features=5,knn_neighbors=4):
    '''

    :param data: Dataframe
    :param maximium_features: Int
    :return: filtered data with selected features
    '''

    knn = KNeighborsClassifier(n_neighbors=knn_neighbors)

    if method == 'sfs':
        # preform Sequential Forward Selection
        sfs = SFS(knn,k_features=k_best_features,forward=True,floating=False,scoring='accuracy',cv=5,n_jobs=-1)
        sfs.fit(X,np.ravel(y))
        selected_features = sfs.k_feature_names_
    elif method == 'sbs':
        # preform Sequential Backward Selection
        sbs = SFS(knn,k_features=k_best_features,forward=False,floating=False,scoring='accuracy',cv=5,n_jobs=-1)
        sbs = sbs.fit(X, np.ravel(y))
        selected_features = sbs.k_feature_names_
    else:
        raise Exception("Warning: feature selection not found shared features, returning feature selection according to 'Fisher Score' algorithm")

    return list(selected_features)


def create_features_subsets(X,y):
    '''
    This function gets X and y and rturn
    :param X:
    :param y:
    :return:
    '''

    selected_features_subsets = {}
    feature_selection_algorithms = {
        "sfs": lambda X, y, k_best_features,W: warpper_feature_selection(X, y, method="sfs",k_best_features=k_best_features),
        "sbs": lambda X, y, k_best_features,W: warpper_feature_selection(X, y, method="sbs",k_best_features=k_best_features),
        "fisher_score": lambda X, y, k_best_features,W: filter_feature_selection(X, y, method="fisher_score",k_best_features=k_best_features,W=W),
        "Laplacian_score": lambda X, y, k_best_features, W: filter_feature_selection(X, y, method="Laplacian_score",k_best_features=k_best_features,W=W),
        "SPEC": lambda X, y, k_best_features, W: filter_feature_selection(X, y, method="SPEC",k_best_features=k_best_features,W=W),
    }

    print('>>>>>>>>>>>>>>>>>>>> Creating features selection subsets <<<<<<<<<<<<<<<<<<<')

    # Loop over the feature selection algorithms
    W = construct_W(X=X.values)
    for k_best_features in config.get('best_select_features_size'):
        for feature_selection_algo in feature_selection_algorithms:
            try:
                # perform feature selection
                print('--------------------------------------------')
                print(feature_selection_algo + ' with top selected %d featurs' % k_best_features)
                print('running...')
                selected_features_subsets[(feature_selection_algo,k_best_features)] = feature_selection_algorithms[feature_selection_algo](X, y, k_best_features,W)
                print('done')
                print('--------------------------------------------')
            except Exception as e:
                print("ERROR: ", e)
                continue

    return selected_features_subsets


def load_data(X_train_path,y_train_path):

    # Load data from CSV file
    X = pd.read_csv(X_train_path)
    y = pd.read_csv(y_train_path)

    # setting data index
    X.set_index('Id', inplace=True)
    y.set_index('Id', inplace=True)

    print("Data Shape: (%d,%d)" % X.shape)
    print("Data Balancing:")
    print("Label 1 count: %d, Label 0 count: %d" % (np.count_nonzero(y == 1),np.count_nonzero(y == 0)))

    return X,y


def print_model_start_time(model_id):
    start_time = time.strftime('%Y-%m-%d %H:%M:%S')
    last_time = start_time
    print('##########################################################')
    print('XGBoost Model {} begins running at: {}'.format(model_id,start_time))
    print('##########################################################')
    return start_time


def model_summary_print(model_id,start_time, scores_array):
    scores_array = np.array(scores_array)
    time_now = time.strftime('%Y-%m-%d %H:%M:%S')
    start_time_delta = datetime.datetime.strptime(time_now, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    print('--------------------------------------- Model Summary Print ----------------------------------------------')
    print('Model {} started running at: {}'.format(model_id,start_time))
    print('Mean test score %1.2f' % scores_array.mean())
    print('STD test score %1.2f' % scores_array.std())
    print('Elapsed Time from the beginning: {}'.format(str(start_time_delta)))
    print('----------------------------------------------------------------------------------------------------------')


if __name__=="__main__":

    # loading Schizophrenia Data
    X,y = load_data('data/train_FNC.csv','data/train_labels.csv')

    summary_df = pd.DataFrame(data=None,columns=['Outer_iteration_number','number_of_features','random_state','best_feature_selection_model','xgboost_model_id','AUC_score'])
    idx = 0
    task_id = uuid.uuid4()
    model_id = uuid.uuid4()
    start_time = print_model_start_time(model_id)

    # init XGBoost algorithm
    model = xgb.XGBClassifier(max_depth=3, subsample=0.7, learning_rate=0.01, n_estimators=1200)

    # for seeding in train_test_split function
    iter_number = 1
    random_states = config.get('random_states')
    xgboost_auc_scores = []

    # number of features to filter
    features_subsets = create_features_subsets(X,y)

    print('>>>>>>>>>>>>>>>>>>>> Start Train <<<<<<<<<<<<<<<<<<<')
    # Twelve random states randomly chosen for the outer-MCCV
    for i in random_states:
        print('Iteration number: %d, seed %d' % (iter_number,i))
        # init best model dictionary
        best_feature_selection_model = {
            "best_features": None,
            "best_auc": -np.inf,
            "best_feature_selection_algorithm": None,
            "k_best_features": 0
        }

        # init dictionary to store feature selection scores
        feature_selection_scores = {}

        # splits data to 80% training set and 20% test test like in the paper
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2, random_state=i, stratify=y)

        # initialize sklearn StratifiedKFold algorithm with 5 Folds like in the paper
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)

        for feature_selection_method in features_subsets.keys():

            # get selected features according to feature selection algorithm
            feature_subset = features_subsets[feature_selection_method]

            # perform k-fold cross validation
            score = cross_val_score(model, X=X_train[feature_subset], y=np.ravel(y_train), cv=inner_cv, scoring='roc_auc')

            # updating best model if necessary
            if score.mean() > best_feature_selection_model['best_auc']:
                best_feature_selection_model['best_auc'] = score.mean()
                best_feature_selection_model['best_features'] = feature_subset
                best_feature_selection_model['best_feature_selection_algorithm'] = feature_selection_method[0]
                best_feature_selection_model['k_best_features'] = feature_selection_method[1]

            # store algorithm auc score
            feature_selection_scores[feature_selection_method[0]] = score.mean()

        # fitting XGBoost model
        print('--------------------------------------------')
        print('Selected features:', best_feature_selection_model['best_features'])
        print('Feature Selection methods scores:')
        for m in feature_selection_scores.keys():
            print("     " + m + " auc score: %1.2f" % feature_selection_scores[m])
            col = m+'_score'
            if col not in summary_df.columns:
                summary_df[col] = None
            summary_df.loc[idx,col] = feature_selection_scores[m]
        print('Best feature selection method: ' + best_feature_selection_model['best_feature_selection_algorithm'])
        print('Number of features: ' + repr(best_feature_selection_model['k_best_features']))
        print('Fitting XGBoost model')
        model.fit(X_train[best_feature_selection_model['best_features']], np.ravel(y_train))
        print('Done fitting')

        # predicting and scoring XGBoost model
        y_pred = model.predict(X_test[best_feature_selection_model['best_features']])
        auc = roc_auc_score(np.ravel(y_test), np.ravel(y_pred))
        print('Auc Best Test Score: %1.2f' % auc)
        print('--------------------------------------------')
        xgboost_auc_scores.append(auc)
        iter_number+=1

        #update summary df
        summary_df.loc[idx,'Outer_iteration_number'] = iter_number-1
        summary_df.loc[idx,'number_of_features'] = len(best_feature_selection_model['best_features'])
        summary_df.loc[idx,'random_state'] = i
        summary_df.loc[idx,'best_feature_selection_model'] = best_feature_selection_model['best_feature_selection_algorithm']
        summary_df.loc[idx,'xgboost_model_id'] = model_id
        summary_df.loc[idx,'AUC_score'] = auc
        idx+=1

    # printing model summary
    model_summary_print(model_id,start_time,xgboost_auc_scores)
    model_mean = np.array(xgboost_auc_scores).mean()

    if config.get('save',False):
        summary_df.to_csv(os.path.join('outputs','task_' + str(task_id) + '.csv'))

    print('###################################################')
    print('###################################################')
    print('Final XGBoost Mean AUC Score: %1.2f' % model_mean)
    print('###################################################')
    print('###################################################')
