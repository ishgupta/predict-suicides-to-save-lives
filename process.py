# In[]
import sys
sys.path.insert( 0, r"E:\office\GPC\repos\integration automation\data_integration" )
from toolkit import get_data, regulate_data_types, write_excel_sheet_v2
sys.path.insert( 0, r"E:\Upswing Pursuit\Projects\toolkit" )
from ml_toolkit import encode_str_columns

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
# In[]
data = get_data( "data_/data.csv", dtype = str )

# In[]
output_column = "Suicide"
# drop columns which have only one unique value
single_val_cols = [ x for x in data.columns if data[ x ].unique().shape[0] == 1 ]
if len( single_val_cols ) > 0:
    data.drop( columns = single_val_cols, inplace = True )

number_cols = [ "Age", "Follow" ]
data[ number_cols ] = data[ number_cols ].astype( int )

"""data = regulate_data_types( 
    data, 
    enforce_str_on_columns = np.setdiff1d( data.columns, number_cols )
)"""
# In[]
factorization_technique = 'one-hot'
data_ = encode_str_columns( data, technique = factorization_technique, str_attributes = None, max_unique_values_allowed_for_str = None, drop_extras = False, output_column = output_column )
"""scaler = StandardScaler()
data_[ number_cols ] = scaler.fit_transform( data_[ number_cols ] )"""

for x in data_.columns:
    data_[x] = pd.to_numeric( data_[x] )

# split into train-test
X_train, X_test, y_train, y_test = train_test_split( data_.drop( columns = [output_column] ), data_[ output_column ], test_size = 0.25, random_state = 42, stratify = data_[ output_column ] )

# In[]
from sklearn.ensemble import RandomForestClassifier
# RF
model = RandomForestClassifier()
model.fit( X_train, y_train )

sorted( dict( 
    zip( 
        X_train.drop( columns = [ output_column ]).columns, 
        model.feature_importances_
        )
    ).items(), key = lambda x : x[1], reverse = True
)

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

feature_importance = sorted( dict( 
    zip( 
        data.drop( columns = [ output_column ]).columns, 
        model.feature_importances_
        )
    ).items(), key = lambda x : x[1], reverse = True
)
print( feature_importance )

# In[]
# round 1
param_grid = {
    "max_depth" : [ 3, 4, 5 ],
    "learning_rate" : [ 0.1, 0.01, 0.05 ],
    "gamma" : [ 0, 0.25, 1.0 ],
    "reg_lamda" : [ 0, 1.0, 10.0 ],
    "scaled_pos_weight" : [ 1, 3, 5 ]
}
"""
{'gamma': 0.25, 'learning_rate': 0.1, 'max_depth': 4, 'reg_lamda': 0, 'scaled_pos_weight': 1}
"""
# round 2
param_grid = {
    "max_depth" : [ 5, 7 ],
    "learning_rate" : [ 0.1, 0.5, 1 ],
    "gamma" : [ 0, 0.25 ],
    "reg_lamda" : [ 0, 0.1, 0.01 ],
    "scaled_pos_weight" : [ 1, 0.5 ],
    "min_child_weight" : [ 0, 1, 2],
    "colsample_bytree" : [ 0.9, 1.0 ]
}
"""
    {'gamma': 0.25, 'learning_rate': 0.1, 'max_depth': 4, 'reg_lamda': 0, 'scaled_pos_weight': 1}
"""

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

print( optimal_params.best_params_ )
# In[]

model = XGBClassifier( 
    objective = 'binary:logistic',
    seed = 42,
    missing = None,
    gamma = 0.25,
    learning_rate = 0.1,
    max_depth = 4,
    reg_lambda = 0,
    scale_pos_weight = 1,
    n_estimators = 1, subsample = 0.9, colsample_bytree = 0.9
)
model.fit( 
    X_train, y_train, 
    verbose = True, early_stopping_rounds = 10, eval_metric = 'aucpr',
    eval_set = [ ( X_test, y_test ) ]
)

feature_importance = sorted( dict( 
    zip( 
        data.drop( columns = [ output_column ]).columns, 
        model.feature_importances_
        )
    ).items(), key = lambda x : x[1], reverse = True
)
print( feature_importance )

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix( model, X_test, y_test, values_format = 'd', display_labels = [ "Not a Suicide", "Suicide" ] )
# In[]
# validate model
y_pred = model.predict( X_test )
y_results = pd.DataFrame( zip( y_pred, y_test ), columns = [ "prediction", "actual" ] )

y_results[ "match" ] = None
y_results.loc[ y_results[ "prediction" ] == y_results[ "actual" ], "match" ] = "Yes"
y_results.loc[ y_results[ "prediction" ] != y_results[ "actual" ], "match" ] = "No"

y_results[ "match" ].value_counts() / y_results.shape[ 0 ]
# In[]
bst = model.get_booster()

for importance_type in ( 'weight', 'gain', 'cover', 'total_gain', 'total_cover' ):
    print( "\n%s: \n\n" % importance_type, bst.get_score( importance_type = importance_type ) )

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

from xgboost import to_graphviz
to_graphviz(
    model, num_trees = 0, size = "10,10",
    condition_node_params = node_params,
    leaf_node_params = leaf_params
)