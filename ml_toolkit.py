import pandas as pd
from pandas import get_dummies
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from itertools import chain
from time import time

from pathlib import Path
from collections import OrderedDict

"""
    todo:
    Base N Encoding using category_encoders package
    utilize ppscore package for linear/non-linear correlation between variables
"""
"""
    find outliers in the data attribute beyond the first and third quartile
    data : pandas Series or a list of integer values
"""
def find_outliers( data ):
    data = np.sort(data)
    q3, q1 = np.percentile( data, [ 75, 25 ])
    iqr = q3 - q1
    ceil = ( iqr * 1.5 ) + q3
    floor = q1 - ( iqr * 1.5 )
    outliers = data[ list( np.where( (data > ceil) )[0] ) + list( np.where( ( data < floor ) )[0] ) ]
    outliers = np.unique( outliers )
    return outliers

"""
    drop rows with null values for the said columns in data
    data : pandas data frame
    columns: columns to be checked, if left an empty list, all the attributes are utilized
"""
def remove_rows_with_null_values( data, columns = [] ):
    if len( columns ) > 1:
        data = data.loc[
            data[ columns ].notnull().all( axis = 1 )
        ].reset_index( drop = True )
    else:
        data = data.loc[
            data.notnull().all( axis = 1 )
        ].reset_index( drop = True )
    return data

"""
    impute numeric columns for missing values, and create a new column with suffix, 'was_missing' to identify the respective data location was imputed
    data : data frame
    strategy : most_frequent - to populate missing values with the most frequent value in the column
                mean - to impute missing values with the mean of data attribute
                median - to impute missing values with the median of data attribute
                constant - to impute missing values with a custom fill value
"""
def impute_number_attributes( data, strategy = 'most_frequent', fill_value = None, use_copy = True ):
    if use_copy:
        data = data.copy( deep = True )
    missing_val_cols = [ x for x in data.select_dtypes( [ 'integer', 'float' ] ).columns if data[x].isnull().sum() > 0 ]
    if len( missing_val_cols ) > 0:
        if strategy == 'constant':
            imputer = SimpleImputer( strategy = strategy, fill_value = fill_value )
        else:
            imputer = SimpleImputer( strategy = strategy )
        for x in missing_val_cols:
            data[ x + "_was_missing"] = data[ x ].isnull()
        data[ missing_val_cols ] = imputer.fit_transform( data[ missing_val_cols ] )
    return data

"""
    drop string columns from the data
"""
def drop_str_attributes_from_data( data ):
    str_attributes = data.select_dtypes( include = [ 'string', 'object' ] ).columns.tolist()
    return data.drop( columns = str_attributes )

"""
    encode categorical columns in the data using one-hot encoding or factorization
    data : pandas data frame
    technique : one-hot - create new columns as per the unique values in a string column, and encode them with 1/0 to flag the presence
                factorize - encode the same data column with integers ranging from -1 to the max number of unique values in the data column. 
                            This will also generate an factors file in the output directory to refer factors in future.
    str_attributes: list of str attributes to encode. If left None, all the str attributes in the data frame will be encoded
    max_unique_values_allowed_for_str : the column with number of unique values <= max_unique_values_allowed_for_str will be encoded. 
                                        If left None, it will be encoded if number of unique values <= 100
    drop_extras: any column ( in str_attributes) exceeding the unique values more than max_unique_values_allowed_for_str will be dropped
    output_column : if provided, the output_column will not be encoded, and return as is
    model : name of the model being used, it is only used to refer in the factors  file exported when technique = 'factorize'
    copy : if True, the original df is not affected, but its copy
"""
def encode_str_columns( data, technique = 'one-hot', str_attributes = None, max_unique_values_allowed_for_str = 10, drop_extras = False, output_column = None, model = None, copy = True ):
    if copy:
        data = data.copy( deep = True )
    # use Base N Encoding in category_encoders package
    if str_attributes is None:
        str_attributes = data.select_dtypes( include = [ 'string', 'object' ] ).columns.tolist()
        if output_column is not None:
            if output_column in str_attributes:
                str_attributes.remove( output_column )
        print( "string columns found: {0}".format( str_attributes ) )
    if technique == 'one-hot':
        if len( str_attributes ) > 0:
            str_uniques = [ ( x, data[x].unique().shape[0] ) for x in str_attributes ]
            str_uniques = sorted( str_uniques, key = lambda x : x[ 1 ])
            str_uniques = [ ( x[0], (x[1] / data.shape[0] * 100) ) for x in str_uniques ]
            if max_unique_values_allowed_for_str is not None:
                str_uniques = [ x[ 0 ] for x in str_uniques if x[ 1 ] <= max_unique_values_allowed_for_str ]
            else:
                max_unique_values_allowed_for_str = 100
                extras = [ x[0] for x in str_uniques if x[1] > 100 ]
                if len( extras ) > 0:
                    print( "cannot encode data column to more than 100 unique values, rejecting {0}".format( ", ".join( extras ) ) )
                    drop_extras = True
                str_uniques = [ x[ 0 ] for x in str_uniques if x[1] <= 100 ]
            if drop_extras:
                str_attributes = np.setdiff1d( str_attributes, str_uniques ).tolist()
                if len( str_attributes ) > 0:
                    print( "dropping {0}".format( ", ".join( str_attributes ) ) )
                    data.drop( columns = str_attributes, inplace = True )
            print( "encoding {0} using {1}" .format( ", ".join( str_uniques ), technique ) )
            start_time = time()
            dummies = pd.get_dummies( data[ str_attributes ] )
            data = pd.concat( [ data.drop( columns = str_attributes ),  dummies ],
                        sort = False, axis = 1
                    )
            end_time = time()
            time_taken = end_time - start_time
            print( "time taken to create dummy variables: {0:.4f} secs".format( time_taken ) )
    elif technique == "factorize":
        str_attributes = data.select_dtypes( include = [ 'string', 'object' ] ).columns.tolist()
        factors = pd.DataFrame()
        for x in str_attributes:
            tmp_labels, _ = data[ x ].factorize( na_sentinel = -1 )
            tmp_factors = pd.DataFrame( [ data[x].tolist(), tmp_labels.tolist() ] ).T.drop_duplicates()
            tmp_factors.columns = [ x, "{0}_factors".format( x ) ]
            factors_columns = factors.columns.tolist() + tmp_factors.columns.tolist()
            factors = pd.concat( [ pd.DataFrame( factors.values ), pd.DataFrame( tmp_factors.values ) ], sort = False, ignore_index= True, axis = 1 )
            factors.columns = factors_columns
            data[ x ] = tmp_labels
            tmp_factors = None
            tmp_labels = None
        write_excel_sheet_v2( factors, "output/data_factors_{0}.xlsx".format( model if model is not None else "" ) )
    return data

"""
    fetch train test split of the data frame
    data: pandas data frame
    output_column  : the dependent attribute(s)
    test size: partition factor of train/test split
    random_state : seed for the operation
    stratify : if True, the train/test split is stratified basis the output column(s)

"""
def fetch_train_test_split( data, output_column, test_size = 0.25, random_state = 42, stratify = True ):
    # take out the output variable from data
    y = data[ output_column ]
    data = data.drop( columns = [ output_column ] )
    # split data into train/test
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split( data, y, test_size = test_size, random_state = random_state, stratify = y )
    else:
        X_train, X_test, y_train, y_test = train_test_split( data, y, test_size = test_size, random_state = random_state )
    return X_train, X_test, y_train, y_test

"""
    write worksheets using pandas dataframe or a dict of dataframes
"""
def write_excel_sheet_v2( data, output_file_name= "output.xlsx", collate_data = False, collated_sheet_str = None,
                      single_sheet_name = "All", format = None, header = True, mode = 'w', index = False ):
    # validate data
    is_data_frame_empty = is_empty_df( data )
    if is_data_frame_empty is None:
        print("invalid data format.. Please use it for pandas DataFrame, or a dict of dataframe objects")
        return None
    elif is_data_frame_empty:
        print("nothing to write, empty data frame. . .")
        return None
    # check if data is a dictionary of data frames (which is the case when input or output is a collection of worksheets, example: .xlsx )
    is_data_in_more_than_one_sheet = ( isinstance( data, dict ) or isinstance(data, OrderedDict) ) and ( len( data.keys() ) > 1 )
    # finalize data format and write
    output_file_name = Path( output_file_name.strip() ) if isinstance( output_file_name, str ) else output_file_name
    # output_file_name.stem = output_file_name.stem.strip()
    # output_file_name.parent.mkdir( parents = True, exist_ok = True )
    create_parent_directories( output_file_name )
    if format is None:
        format = output_file_name.suffix
    # finalize data (to write)
    if collate_data:
        if is_data_in_more_than_one_sheet:
            # if more than one sheet is available in data, and collation is requested- collate the data
            print( "collating data in one dataframe" )
            data = collate_all_sheets( data, collated_sheet_str )
        elif ( not is_data_in_more_than_one_sheet ) and ( ( isinstance( data, dict ) or isinstance( data, OrderedDict ) ) ):
            # if it is requested to collate, but the data does not contain more than one sheet
            collate_data = False
            print( "nothing to collate" )
            data = data[ list( data.keys() )[0] ]
    else:
        if is_data_in_more_than_one_sheet and ( format not in [ '.xlsx', '.xls' ] ):
            print( "data available in more than one data frames, collating everything. . . \n . . use collate_date = True from next time in such a case." )
            data = collate_all_sheets( data )
    if format in [ '.xls', '.xlsx' ]:
        if format == '.xls':
            output_file_name = Path( str( output_file_name ).replace( output_file_name.suffix, '.xlsx' ) )
        writer = pd.ExcelWriter( str( output_file_name ), engine='xlsxwriter' )
        if collate_data:
            if data.shape[ 0 ] > 1048576:
                tmp_output_file_name = Path( str( output_file_name ).replace( output_file_name.suffix, '.csv' ) )
                print( "data frame is too huge, cannot be written to an .xls(x) file, writing a csv instead: {0}. . .".format( tmp_output_file_name ) )
                write_excel_sheet_v2( data, tmp_output_file_name , collate_data, collated_sheet_str, single_sheet_name, ".csv", header, mode, index )
            data.to_excel( writer, sheet_name = single_sheet_name, index = index, header = header )
            writer.save()
        else:
            if ( isinstance( data, dict ) or isinstance(data, OrderedDict) ):
                sheets = list( data.keys() )
                for sheet in sheets:
                    if data[ sheet ].shape[ 0 ] > 1048576:
                        tmp_output_file_name = Path( output_file_name.name.replace( output_file_name.suffix, "" ) + "__" + sheet + '.csv' )
                        print( "data frame is too huge, cannot be written to .xls(x), writing a csv instead : {0}. . .".format( tmp_output_file_name ) )
                        format = ".csv"
                        write_excel_sheet_v2( data[ sheet ], tmp_output_file_name, collate_data, collated_sheet_str, single_sheet_name, ".csv", header, mode, index )
                    else:
                        if len( sheet ) > 31:
                            print("truncating sheet name from '{0}' to : '{1}'".format( sheet, sheet[ : 31 ] ))
                            data[ sheet ].to_excel( writer, sheet_name = sheet[ : 31 ], index = index )
                        else:
                            data[ sheet ].to_excel( writer, sheet_name = sheet, index = index )
                writer.save()
            else:
                if data.shape[ 0 ] > 1048576:
                    tmp_output_file_name = Path( output_file_name.name.replace( output_file_name.suffix, "" ) + '.csv' )
                    print( "data frame is too huge, cannot be written to .xls(x), writing a csv instead : {0}. . .".format( tmp_output_file_name ) )
                    write_excel_sheet_v2( data, tmp_output_file_name, collate_data, collated_sheet_str, single_sheet_name, ".csv", header, mode, index )
                else:
                    data.to_excel( output_file_name, index = index, header = header )
        print( "finally writing {0} file: '{1}'".format( ".xlsx", output_file_name.name ) ) 
    elif format == '.csv':
        print( "finally writing {0} file: '{1}'".format( format, output_file_name.name ) )
        data.to_csv( output_file_name, index = index, header = header, mode = mode )
    else:
        print("{0} format not supported. . . Please use .csv, or .xls(x) file format to export")
        return None
    return output_file_name

"""
    check if the provided data frame is empty
"""
def is_empty_df( data ):
    is_data_frame_empty = True
    if isinstance(data, dict):
        data_sheets = list( data.keys() )
        for sheet in data_sheets:
            is_data_frame_empty = is_data_frame_empty and is_empty_df( data[ sheet ] )
    elif isinstance(data, pd.DataFrame):
        is_data_frame_empty = ( data.shape[0] == 0 and data.shape[1] == 0)
    else:
        print("invalid data format.. Please use it for pandas DataFrame, or a dict of dataframe objects")
        return None
    return is_data_frame_empty

def collate_all_sheets(data, sheet_name_column = None, save_sheet_in_title_case = True):
    """
        #generic
        method to collate all data sheets into one. returns a single data frame which is collated form of all the data frames available in dictionary as values
        
        :param data: data provided as a dictionary, where key is the reference name for data, and value is the data frame itself
        :type data: dict

        :rtype: pd.DataFrame
    """
    final_df = None
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, dict):
        sheets = list( data.keys() )
        for sheet in sheets:
            if save_sheet_in_title_case:
                if isinstance(sheet, str):
                    sheet_str = sheet.title()
            if not isinstance( data[ sheet ], pd.DataFrame ):
                data[ sheet ] = pd.DataFrame.from_dict( data[ sheet ] )
                if sheet_name_column is not None:
                    data[ sheet ][sheet_name_column] = sheet_str
            if data[ sheet ].shape[0] > 0:
                # find differential columns which need to be added per sheet
                columns_to_be_added = list( np.setdiff1d( np.unique(list(chain(* [ data[x].columns.tolist() for x in sheets ] ))), data[ sheet ].columns ) )
                if sheet_name_column is not None:
                    if sheet_name_column in columns_to_be_added:
                        columns_to_be_added.remove( sheet_name_column )

                if len(columns_to_be_added) > 0:
                    print("columns to be added in sheet : '{0}' are : \t {1}".format(sheet, columns_to_be_added) )
                else:
                    print("no new columns to be added in sheet : '{0}'".format(sheet) )

                for x in columns_to_be_added:
                    data[ sheet ][x] = None

                columns_to_be_added = None

                if sheet_name_column is not None:
                    print("saving sheet name in column: '{0}'".format( sheet_name_column ) )
                    data[ sheet ][sheet_name_column] = sheet
                data[ sheet ][ data[ sheet ].select_dtypes( [ 'string' ] ).columns ] = data[ sheet ].select_dtypes( [ 'string' ] ).astype('str').values
            else:
                print("no data to collate in the sheet : '{0}'".format(sheet) )
        print( [ (sheet, len(data[ sheet ].columns.tolist() )) for sheet in data.keys() ] )
        
        final_df = pd.concat( [ data[ sheet ] for sheet in sheets ], ignore_index = True, sort = False)
        final_df = final_df.convert_dtypes()
    return final_df

"""
    create parent directories for a file, if not exist already
"""
def create_parent_directories( file_path ):
    file_path = Path( file_path ) if not isinstance(file_path, Path) else file_path
    if is_file( file_path ):
        if not Path( file_path.parent ).exists():
            Path( file_path.parent ).mkdir( parents=True, exist_ok=True )
        return True
    else:
        if file_path.exists():
            return True
        else:
            Path( file_path ).mkdir( parents=True, exist_ok=True )
            return True
    return False

def is_file( file_path ):
    file_path = Path( file_path ) if not isinstance(file_path, Path) else file_path
    if file_path.is_file():
        return True
    elif "." in file_path.name:
        return True
    return False