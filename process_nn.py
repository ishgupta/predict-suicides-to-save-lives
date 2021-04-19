# In[]
from ml_toolkit import encode_str_columns, write_excel_sheet_v2

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F

import datetime
# In[]
output_column = 'Suicide'
# In[]
data = pd.read_csv( "data_/11.csv", dtype = str )
data.columns = [ x.title() for x in data.columns ]
# suicide
data[ 'Cause113' ] = data[ 'Cause113' ].astype( int )

data[ "Suicide" ] = 0
data.loc[ data[ "Cause113" ].isin( [ 105, 106 ] ), "Suicide" ] = 1
data.loc[ data[ "Cause113" ].isin( [ 100, 104, 107, 108, 109 ] ), "Suicide" ] = 2

data[ "Working" ] = 1
data.loc[ data[ "Occ" ].isnull(), "Working" ] = 0

data.drop( columns = [ "Record", "Wt", "Hhid", "Occ", "Dayod", "Cause113", "Inddea", "Indalg" ], inplace = True )
data.drop( columns = [ "Hosp", "Hospd" ], inplace = True )

# type cast integer columns as int
number_cols = [ "Age", "Follow" ]
data[ number_cols ] = data[ number_cols ].astype( int )

# clean data
"""drop columns which have only one unique value"""
single_val_cols = [ x for x in data.columns if data[ x ].unique().shape[0] == 1 ]
if len( single_val_cols ) > 0:
    print( "dropping '{0}' for they had only one unique value respectively".format( ", ".join( single_val_cols ) ) )
    data.drop( columns = single_val_cols, inplace = True )
else:
    print( "no change, all the columns have at least 2 unique values in the data" )

# drop columns with more than 40% of blank values
null_values = data.isnull().sum().reset_index()
null_values.columns = [ "column", "blanks" ]
null_values = null_values.loc[ ( null_values[ "blanks" ] / data.shape[0] ) > 0.4 ]
blank_cols = null_values['column'].tolist()
null_values_df = data[ blank_cols + [ output_column ] ].copy( deep = True )
for col in blank_cols:
    null_values_df.loc[ null_values_df[ col ].notnull(), col ] = 1
    null_values_df.loc[ null_values_df[ col ].isnull(), col ] = 0
null_values_df.groupby( output_column ).sum()
if null_values.shape[ 0 ] > 0:
    print( "dropping columns: '{0}', for they had more than 40% of missing values in the data".format( null_values['column'].tolist() ) )
    data.drop( columns = null_values['column'].tolist(), inplace = True )
else:
    print( "all okay. all the columns have at least 60% of values available in the column" )

# lets transform follow up days into years
data['Follow'] = ( round( data['Follow'] / 365, 0 ) ).astype( int )

age_groups = [ 0, 6, 13, 19, 31, 60, 91 ]
age_labels = [ "infant", "kid", "teenager", "adult", "man", "senior" ]
data['Age'] = pd.cut( data['Age'] , bins = age_groups, labels = age_labels, right = False )

data = data.astype( 'category' )

from missingpy import MissForest

imputer = MissForest()
imputed_cols = data.drop( columns = [ 'Age', 'Suicide' ] ).columns

data_copy = imputer.fit_transform( data.drop( columns = [ 'Age', 'Suicide' ] ) )
data_copy = pd.DataFrame( data_copy, columns = imputed_cols )
data_copy = data_copy.astype(int)
data_copy = data_copy.astype( 'category' )
data_copy = pd.concat( [ data_copy, data[ [ 'Age', 'Suicide' ] ] ], axis = 1 )
data = data_copy

# In
write_excel_sheet_v2( data, "data_/11_data_imputed_miss_forest.csv")
write_excel_sheet_v2( data, "data_/11_data_imputed_miss_forest.xlsx")

# In[]
# data = pd.read_csv( "data_/data_clean.csv", dtype = str )
# data = pd.read_csv( "data/data_xg_clean_encoded.csv" )
# training_acc_results = None
# test_acc_results = None

output_column = "Suicide"

model_choice = [ "user", "sequential", "log_soft_max", "four" ]
model_choice = input( "select a model: {0}".format( ", ".join( model_choice ) ) )

# In[]
"""
======================================
"""

# In[]
# prepare data

factorization_technique = 'one-hot'
data_ = encode_str_columns( data.astype( str ), technique = factorization_technique, str_attributes = None, max_unique_values_allowed_for_str = None, drop_extras = False, output_column = output_column )
"""scaler = StandardScaler()
data_[ number_cols ] = scaler.fit_transform( data_[ number_cols ] )"""

for x in data_.columns:
    # data_[ x ] = pd.to_numeric( data_[x] )
    data_[ x ] = data_[x].astype( int )

X_train, X_test, y_train, y_test = train_test_split( data_.drop( columns = [output_column] ), data_[ output_column ], test_size = 0.25, random_state = 42, stratify = data_[ output_column ] )

X_train = torch.tensor( np.stack( X_train.values, axis = 0 ), dtype = torch.float )
X_test = torch.tensor( np.stack( X_test.values, axis = 0 ), dtype = torch.float )
y_train = torch.tensor( np.stack( y_train.values, axis = 0 ), dtype = torch.long )
y_test = torch.tensor( np.stack( y_test.values, axis = 0 ), dtype = torch.long )

# In[]


epochs = 100

# instantiate the model
input_features = X_train.shape[ 1 ]
hidden_1_features = 64
hidden_2_features = 64
output_features = np.unique( y_test ).shape[ 0 ]

nn_model = None
torch.manual_seed( 42 )

if model_choice == "user":
    # create model
    class Model( nn.Module ):
        def __init__( self, input_features, hidden_1, hidden_2, output_features = 2 ):
            super().__init__()
            self.f_connected_1 = nn.Linear( input_features, hidden_1 )
            self.f_connected_2 = nn.Linear( hidden_1, hidden_2 )
            self.out = nn.Linear( hidden_2, output_features )
            self.batchnorm_1 = nn.BatchNorm1d( hidden_1 )
            self.batchnorm_2 = nn.BatchNorm1d( hidden_2 )
            self.dropout = nn.Dropout( p = 0.2 )
            
        def forward( self, x ):
            x = F.relu( self.f_connected_1( x ) )
            x = self.batchnorm_1( x )
            x = F.relu( self.f_connected_2( x ) )
            x = self.batchnorm_2( x )
            x = self.dropout( x )
            x = self.out( x )
            return x
    
    nn_model = Model( input_features, hidden_1_features, hidden_2_features, output_features )
elif model_choice == "sequential":
    nn_model = torch.nn.Sequential(
            torch.nn.Linear( input_features, hidden_1_features ),
            torch.nn.BatchNorm1d( hidden_1_features ),
            torch.nn.Dropout( p = 0.3 ),
            torch.nn.ReLU( hidden_1_features ),
            torch.nn.Linear( hidden_1_features, hidden_2_features ),
            torch.nn.BatchNorm1d( hidden_2_features ),
            torch.nn.Dropout( p = 0.3 ),
            torch.nn.ReLU( hidden_2_features ),
            torch.nn.Linear( hidden_2_features, output_features ),            
            torch.nn.Sigmoid()
        )
elif model_choice == 'log_soft_max':
    nn_model = torch.nn.Sequential(
        nn.Linear( input_features, hidden_1_features ),
        nn.ReLU(),
        nn.Linear( hidden_1_features, hidden_2_features),
        nn.ReLU(),
        nn.Linear( hidden_2_features, output_features ),
        nn.LogSoftmax( dim = 1 )
    )
elif model_choice == "four":
    class NN_Classifier( nn.Module ):
        def __init__( self, input_features, hidden_1_features, hidden_2_features, output_features ):
            super().__init__()
            self.fc1 = nn.Linear( input_features, hidden_1_features )
            self.fc2 = nn.Linear( hidden_1_features, hidden_2_features )
            self.fc3 = nn.Linear( hidden_2_features, output_features )

        def forward( self, x ):
            x = F.relu( self.fc1( x ) )
            x = F.relu( self.fc2( x ) )
            x = F.log_softmax( F.relu( self.fc3( x ) ), dim = 1 )
            return x
    nn_model = NN_Classifier( input_features, hidden_1_features, hidden_2_features, output_features )
elif model_choice == "five":
    class NN_Classifier( torch.nn.Module ):
        def __init__(self, input_features, hidden_1_features, hidden_2_features, output_features ):
            # Define layers
            super(NN_Classifier, self).__init__()
            self.fc1 = torch.nn.Linear(input_features, hidden_1_features)
            self.bn1 = torch.nn.BatchNorm1d(hidden_1_features)
            self.fc2 = torch.nn.Linear(hidden_1_features, hidden_2_features)
            self.bn2 = torch.nn.BatchNorm1d(hidden_2_features)
            self.fc3 = torch.nn.Linear(hidden_2_features, output_features)
            
        def forward(self, x):
            # Define sequence of layers
            x = self.fc1(x) # Fully connected layer
            x = self.bn1(x) # Batch normalisation
            x = F.dropout(x, p = 0.35 ) # Apply dropout
            x = F.relu(x) # ReLU activation
            x = self.fc2(x) # Fully connected layer
            x = self.bn2(x) # Batch normalisation
            x = F.dropout(x, p = 0.35 ) # Apply dropout
            x = F.relu(x) # ReLU activation
            x = self.fc3(x) # Fully connected layer
            x = torch.sigmoid(x) # Sigmoid output (0-1)
            return x
    nn_model = NN_Classifier( input_features, hidden_1_features, hidden_2_features, output_features )

# In[]
# backward propagation - define loss function, and the optimizer
if model_choice in [ "log_soft_max", "four"]:
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD( nn_model.parameters(), lr = 0.01 )
else:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam( nn_model.parameters(), lr = 0.01 )

"""loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD( nn_model.parameters(), lr = 0.01 )"""
# In[]
final_losses = []
for epoch in range( epochs ):
    epoch += 1
    y_pred = nn_model.forward( X_train )
    loss = loss_fn( y_pred, y_train )
    final_losses.append( loss )
    if epoch % 25 == 0:
        print( "Epoch# {0}, loss: {1}".format( epoch, loss.item() ) )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# In[]

# map the loss function

import matplotlib.pyplot as plt
# %matplotlib inline
plt.plot( range( epochs ), final_losses )
plt.ylabel( 'Loss' )
plt.xlabel( 'Epoch' )

# In[]
# predict
# In[]
predictions = []
with torch.no_grad():
    predictions = nn_model( X_test )
if model_choice in [ "log_soft_max", "four"]:
    predictions = [ torch.exp( x ).argmax().item() for x in predictions ]
else:
    predictions = [ x.argmax().item() for x in predictions ]

"""with torch.no_grad():
    for index, input_data in enumerate( X_test ):
        input_data = input_data.view( 1, input_data.shape[ 0 ] )
        pred = nn_model( input_data )
        predictions.append( pred.argmax().item() )"""

print( pd.Series( predictions ).value_counts() )
# In[]
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

cm = confusion_matrix( y_test, predictions )
score = accuracy_score( y_test, predictions )

plt.figure( figsize = ( 10, 6 ) )
sns.heatmap( cm, annot = True )
plt.xlabel( "Actual" )
plt.ylabel( "Prediction" )

# In[]
# save the model
model_file_name = "output/nlms_nn_model{0}_{1}_{2}.pt".format( str( datetime.date.today() ).replace( "-", "_" ), model_choice, round( score, 4 ) )
torch.save( nn_model, model_file_name )
"""nn_model = torch.load( model_file_name )"""
nn_model.eval()

# In[]
# Set up lists to hold results
if training_acc_results:
    training_acc_results = training_acc_results
    test_acc_results = test_acc_results
else:
    training_acc_results = []
    test_acc_results = []

with torch.no_grad():
    tmp_predictions = nn_model( X_train )
    tmp_predictions = [ x.argmax().item() for x in tmp_predictions ]
training_acc_results.append( np.mean( pd.Series( tmp_predictions ) == pd.Series( y_train ) ) )

with torch.no_grad():
    tmp_predictions = nn_model( X_test )
    tmp_predictions = [ x.argmax().item() for x in tmp_predictions ]
test_acc_results.append( np.mean( pd.Series( tmp_predictions ) == pd.Series( y_test ) ) )

x_for_box = [ training_acc_results, test_acc_results ]
labels = [ "train", "test" ]
fig = plt.figure( figsize = (5, 5) )
ax1 = fig.add_subplot( 111 )
ax1.boxplot( x_for_box, widths = 0.7 )
ax1.set_xticklabels( labels )
ax1.set_ylabel( "Accuracy" )
plt.show()

# In[]
print( dict(
    zip(
        training_acc_results, test_acc_results
    )
) )
# In[]
# predict new data points
with torch.no_grad():
    print( model( data_point ).agmax().item() )

# In[]
"""
========================================================================
"""





# In[]

"""
    implement a neural network to predict suicides
"""
# In[]
from torch import nn, optim
import torch.nn.functional as F
import helper
from collections import OrderedDict

class Network( nn.Module ):
    def __init__( self ):
        super().__init__()
        self.hidden = nn.Linear( 278, 10 )
        self.output = nn.Linear( 10, 2 )

    def forward( self, x ):
        x = F.relu( self.hidden( x ) )
        x = F.softmax( self.output( x ), dim = 1 )
        return x

nn_model = Network()
nn_model.hidden.bias.fill_( 0 )
nn_model.hidden.weight.data.normal_( std = 0.01 )


nn_model = nn.Sequential(
    OrderedDict(
        [
            ( 'hidden_1', nn.Linear( input_size, hidden_sizes[ 0 ] ) ),
            ( 'relu1', nn.ReLU() ),
            ( 'hidden_2', nn.Linear( hidden_sizes[ 0 ], hidden_sizes[ 1 ] ) ),
            ( 'relu2', nn.ReLU() ),
            ( 'output', nn.Linear( hidden_sizes[ 1 ], output_size ) ),
            ( 'softmax', nn.LogSoftmax( dim = 1 ) )
        ]
    )
)

print( nn_model )

criterion = nn.NLLLoss()
optimizer = optim.SGD( nn_model.parameters(), lr = 0.01 )
epochs = 5

"""from torchvision import transforms
transform = transforms.Compose( [ transforms.ToTensor(), transforms.Normalize( ( 0.5, ), ( 0.5, ) ), ] )
"""

# In[]
# save the model
"""for model_choice in model_choices:
    model_file_name = "output/nlms_nn_model{0}_{1}_{2}_epochs_{3}.pt".format( 
        str( datetime.date.today() ).replace( "-", "_" ), 
        model_choice, 
        round( accuracy_scores[ model_choice ], 4 ), 
        epochs
    )
    torch.save( nn_model[model_choice], model_file_name )
    """nn_model = torch.load( model_file_name )"""
    nn_model[model_choice].eval()"""