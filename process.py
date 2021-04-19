# In[]
import sys
from ml_toolkit import encode_str_columns, write_excel_sheet_v2

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# In[]

def fetch_feature_importance( data, feature_importances ):
    feature_importance = sorted( dict( 
        zip( 
            data.drop( columns = [ output_column ]).columns, 
            feature_importances
            )
        ).items(), key = lambda x : x[1], reverse = True
    )
    return feature_importance

def filter_data_for_imp_variables( data, feature_importance, output_column ):
    feature_importance = fetch_feature_importance( data, feature_importance )
    imp_variables = [ x for x,y in feature_importance if y > 0 ]
    imp_variables = imp_variables + [ output_column ]
    return data[ imp_variables ]

# In[]
data = pd.read_csv( "data_/data_clean.csv", dtype = 'category' )
output_column = "Suicide"

# In[]
factorization_technique = 'one-hot'
data_ = encode_str_columns( data.astype( str ), technique = factorization_technique, str_attributes = None, max_unique_values_allowed_for_str = None, drop_extras = False, output_column = output_column )

for x in data_.columns:
    data_[ x ] = data_[x].astype( int )

write_excel_sheet_v2( data_, "data_/data_encoded.xlsx" )

# In[]
scores = {}

# In[]
# split into train-test
X_train, X_test, y_train, y_test = train_test_split( data_.drop( columns = [output_column] ), data_[ output_column ], test_size = 0.25, random_state = 42, stratify = data_[ output_column ] )
# In[]
# XGBoost
model = XGBClassifier( 
    objective = 'binary:logistic',
    seed = 42,
    missing = None,
    min_child_weight = 0, learning_rate = 0.01, n_estimators= 10
)
model.fit( 
    X_train, y_train, 
    verbose = True, early_stopping_rounds = 10, eval_metric = 'aucpr',
    eval_set = [ ( X_test, y_test ) ]
)

scores[ 'raw' ] = accuracy_score( y_test, model.predict( X_test ) )
print( fetch_feature_importance( data_, model.feature_importances_ ) )

# In[]
# round 1
param_grid = {
    "max_depth" : [ 3, 4, 5 ],
    "learning_rate" : [ 0.1, 0.01, 0.05 ],
    "gamma" : [ 0, 0.25, 1.0 ],
    "min_child_weight" : [ 0, 1, 2],
    "colsample_bytree" : [ 0.9, 1.0 ]
}
optimal_params = GridSearchCV(
    estimator = XGBClassifier( objective = 'binary:logistic', seed = 42, 
    subsample = 0.9, colsample_bytree = 0.5 ),
    param_grid = param_grid, scoring = 'roc_auc',
    verbose = 2,
    n_jobs = 10,
    cv = 3
)

optimal_params.fit( X_train, y_train,
    early_stopping_rounds = 10,
    eval_metric = 'auc',
    eval_set = [ ( X_test, y_test ) ],
    verbose = False
)

model = optimal_params.best_estimator_
print( optimal_params.best_params_ )
print( fetch_feature_importance( data_, model.feature_importances_ ) )
scores[ 'cv_1' ] = accuracy_score( y_test, model.predict( X_test ) )
print( "Accuracy: {0}".format( scores[ 'cv_1' ] ) )
data_ = filter_data_for_imp_variables( data_, model.feature_importances_, output_column )
X_train, X_test, y_train, y_test = train_test_split( data_.drop( columns = [output_column] ), data_[ output_column ], test_size = 0.25, random_state = 42, stratify = data_[ output_column ] )

# In[]
# round 2
param_grid = {
    "max_depth" : [ 5, 6, 7 ],
    "learning_rate" : [ 0.025, 0.05, 0.075 ],
    "gamma" : [ 0, 0.01, 0.25,  0.75 ],
    "min_child_weight" : [ 0.75, 1, 1.25 ],
    "colsample_bytree" : [ 0.75, 1.0 ]
}

optimal_params = GridSearchCV(
    estimator = XGBClassifier( objective = 'binary:logistic', seed = 42, 
    subsample = 0.9, colsample_bytree = 0.5 ),
    param_grid = param_grid, scoring = 'roc_auc',
    verbose = 2,
    n_jobs = 10,
    cv = 3
)

optimal_params.fit( X_train, y_train,
    early_stopping_rounds = 10,
    eval_metric = 'auc',
    eval_set = [ ( X_test, y_test ) ],
    verbose = False
)

model = optimal_params.best_estimator_
print( optimal_params.best_params_ )
print( fetch_feature_importance( data_, model.feature_importances_ ) )
scores[ 'cv_2' ] = accuracy_score( y_test, model.predict( X_test ) )
print( "Accuracy: {0}".format( scores[ 'cv_2' ] ) )
data_ = filter_data_for_imp_variables( data_, model.feature_importances_, output_column )
X_train, X_test, y_train, y_test = train_test_split( data_.drop( columns = [output_column] ), data_[ output_column ], test_size = 0.25, random_state = 42, stratify = data_[ output_column ] )

# In[]
optimal_params.best_score_
optimal_params.best_params_

model
# In[]
# train with 1 estimator to plot the tree
model = XGBClassifier( 
    objective = 'binary:logistic',
    seed = 42,
    missing = None,
    gamma = 0.25,
    learning_rate = 0.075,
    max_depth = 5,
    reg_lambda = 0,
    scale_pos_weight = 1,
    n_estimators = 1, min_child_weight = 0.75,
    colsample_bytree = 0.75,
    subsample = 0.9
)
model.fit( 
    X_train, y_train, 
    verbose = True, early_stopping_rounds = 10, eval_metric = 'aucpr',
    eval_set = [ ( X_test, y_test ) ]
)

scores[ 'single_estimator_from_best' ] = accuracy_score( y_test, model.predict( X_test ) )
print( "Accuracy: {0}".format( scores[ 'single_estimator_from_best' ] ) )

print( "Accuracy: {0}".format( accuracy_score( y_test, model.predict( X_test ) ) ) )

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix( model, X_test, y_test, values_format = 'd', display_labels = [ "Not a Suicide", "Suicide" ] )

data_ = filter_data_for_imp_variables( data_, model.feature_importances_, output_column )
write_excel_sheet_v2( data_, "data/data_xg_clean_encoded.csv" )
X_train, X_test, y_train, y_test = train_test_split( data_.drop( columns = [output_column] ), data_[ output_column ], test_size = 0.25, random_state = 42, stratify = data_[ output_column ] )

# In[]
bst = model.get_booster()

for importance_type in ( 'weight', 'gain', 'cover', 'total_gain', 'total_cover' ):
    print( "\n%s: \n\n" % importance_type, bst.get_score( importance_type = importance_type ) )

# In[]
node_params = {
    "shape" : "box",
    "style" : "filled, rounded",
    "fillcolor" : "#78cbe"
}

leaf_params = {
    "shape" : "box",
    "style" : "filled",
    "fillcolor" : "#e48038"
}

# In[]
from xgboost import to_graphviz
to_graphviz(
    model, num_trees = 0, size = "10,10",
    condition_node_params = node_params,
    leaf_node_params = leaf_params
)

# In[]
xg_viz = to_graphviz(
    model, num_trees = 0, size = "10,10",
    condition_node_params = node_params,
    leaf_node_params = leaf_params
)
xg_viz.render( "xgboost_viz", "output", format = 'pdf' )
# In[]


# In[]
print("abc")
