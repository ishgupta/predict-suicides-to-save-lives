

# In[]
# prepare data for tensors
"""
...
"""

data[ np.setdiff1d( data.columns, number_cols ) ] = data[ np.setdiff1d( data.columns, number_cols ) ].astype( 'category' )
data_ = data.copy( deep = True )
for x in data_.select_dtypes( 'category' ).columns:
    data_[ x + "_codes" ] = data_[ x ].cat.codes.values
    data_[ x ] = data_[ x ].cat.codes.values
write_excel_sheet_v2( data_, "output/data_factors.xlsx" )
data_.drop( columns = data_.columns[ data_.columns.str.endswith( "_codes" ) ], inplace = True )

X_train, X_test, y_train, y_test = train_test_split( 
    data_.drop( columns = [output_column] ), 
    data_[ output_column ], 
    test_size = 0.25, 
    random_state = 42, 
    stratify = data_[ output_column ]
)

train_categorical_data = np.stack(
    X_train[ np.setdiff1d( data.select_dtypes( 'category' ).columns, output_column ) ].values, 
    axis = 0
)
train_categorical_data = torch.tensor( train_categorical_data, dtype = torch.int64 )
test_categorical_data = np.stack(
    X_test[ np.setdiff1d( data.select_dtypes( 'category' ).columns, output_column ) ].values, 
    axis = 0
)
test_categorical_data = torch.tensor( test_categorical_data, dtype = torch.int64 )

numerical_data = torch.tensor( np.stack(
    data_[ data.select_dtypes( 'number' ).columns ].values, axis = 0
), dtype = torch.float )

train_numerical_data = torch.tensor( np.stack(
    X_train[ data.select_dtypes( 'number' ).columns ].values, axis = 0
), dtype = torch.float )

test_numerical_data = torch.tensor( np.stack(
    X_test[ data.select_dtypes( 'number' ).columns ].values, axis = 0
), dtype = torch.float )

train_outputs = torch.tensor( y_train.values ).flatten()
test_outputs = torch.tensor( y_test.values ).flatten()

print( train_categorical_data.shape, train_numerical_data.shape, train_outputs.shape )
print( test_categorical_data.shape, test_numerical_data.shape, test_outputs.shape )

categorical_column_sizes = [ len( data[ column ].cat.categories ) for column in np.setdiff1d( data.columns, number_cols ) ]
categorical_embedding_sizes = [ ( col_size, min( 50, ( col_size + 1 ) // 2 ) ) for col_size in categorical_column_sizes ]
print( categorical_embedding_sizes )

# In[]

# prepare model

class Model( nn.Module ):
    """
        embedding_size : embedding size of the categorical columns
        num_numerical_cols : number of numerical columns
        output_size : the number of possible outputs
        layers : 
        p : dropout with default value, 0.5

    """
    def __init__( self, embedding_size, num_numerical_cols, output_size, layers, p = 0.4 ):
        super().__init__()
        self.all_embeddings = nn.ModuleList( [ nn.Embedding( ni, nf ) for ni, nf in embedding_size ] )
        self.embedding_dropout = nn.Dropout( p )
        self.batch_norm_num = nn.BatchNorm1d( num_numerical_cols )

        all_layers = []
        num_categorical_cols = sum( ( nf for ni, nf in embedding_size ) )
        input_size = num_categorical_cols + num_numerical_cols

        for layer in layers:
            all_layers.append( nn.Linear( input_size, layer ) )
            all_layers.append( nn.ReLU( inplace = True ) )
            all_layers.append( nn.BatchNorm1d( layer ) )
            all_layers.append( nn.Dropout( p ) )
            input_size = layer

        all_layers.append( nn.Linear( layers[ -1 ], output_size ) )
        self.layers = nn.Sequential( *all_layers )

    def forward( self, x_categorical, x_numerical ):
        embeddings = []
        for index, embedding in enumerate( self.all_embeddings ):
            embeddings.append( embedding( x_categorical[ :, index ] ) )
        x = torch.cat( embeddings, 1 )
        x = self.embedding_dropout( x )

        x_numerical = self.batch_norm_num( x_numerical )
        x = torch.cat( [ x, x_numerical ], 1 )
        x = self.layers( x )
        return x

nn_model = Model( categorical_embedding_sizes, train_numerical_data.shape[ 1 ], 2, [ 200, 100, 50 ], 0.4 )
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam( nn_model.parameters(), lr = 0.001 )

# In[]
epochs = 300
aggregated_losses = []

for epoch in range( epochs ):
    epoch += 1
    y_pred = nn_model( train_categorical_data, train_numerical_data )
    single_loss = loss_fn( y_pred, train_outputs )
    aggregated_losses.append( single_loss )

    if epoch%25 == 1:
        print( f'epoch: { epoch : 3 } loss: { single_loss.item():10.8f }' )
    optimizer.zero_grad()
    single_loss.backward()
    optimizer.step()

print( f'epoch: {epoch : 3 } loss: { single_loss.item() : 10.10f }' )