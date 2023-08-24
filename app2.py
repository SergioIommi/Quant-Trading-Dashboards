from dash import Dash, html, dcc, Input, Output, State, callback, no_update
from dash_dangerously_set_inner_html import DangerouslySetInnerHTML
import dash_mantine_components as dmc
import plotly.express as px
import pymongo
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

app = Dash(__name__)

db_params={'db_uri': 'mongodb://localhost:27017/',
           'db_name': 'stocks_db',
           'db_collection_name': 'daily'}
db_uri = db_params['db_uri']
db_name = db_params['db_name']
db_collection_name = db_params['db_collection_name']
        
# Connect to MongoDB
client = pymongo.MongoClient(db_uri)
db = client[db_name]
db_collection = db[db_collection_name]  # Get/create collection for specific type of stocks data (daily, monthly, etc.)


def get_all_index_symbols(db_collection):
    """
    Get list of stock index symbols stored in the database
    """
    index_symbols = [elem.replace('Index_','^') for elem in list(db_collection.find_one().keys()) if elem.startswith('Index_')]
    # Database query to extract pairs {'Symbol', 'Name'} for each Index Symbol in the database
    cursor = db_collection.aggregate([{"$match": {"Symbol": {"$in": index_symbols}
                                             }
                                  },
                                  { '$group': {'_id'   : '$Symbol',
                                               'Symbol': { '$first': '$Symbol' },
                                               'Name'  : { '$first': '$Name'   }
                                              }
                                  },
                                  { '$project': {'_id':     0,
                                                 'Symbol':  1,
                                                 'Name':    1,
                                                }
                                  }
                                 ])
    query_result = list(cursor)
    # Transform pairs {'Symbol', 'Name'} into full strying, ex. convert {^NDX, NASDAQ 100} into '^NDX - NASDAQ 100'
    index_symbols_names = [f'{value1} - {value2}' for value1, value2 in [list(dict_.values()) for dict_ in query_result]]
    return index_symbols_names


def extract_index_symbol(selected_index_symbol_name: str):
    """
    Extract Index Symbol from string combination 'Index_Symbol - Index_Name' (ex. get ^NDX from '^NDX - NASDAQ 100')
    """
    selected_index_symbol = selected_index_symbol_name.split(' - ', 1)[0]
    return selected_index_symbol


def extract_stock_symbols(selected_stock_symbols_names: list):
    """
    Extract Symbols from list of strings combinations ['Stock_Symbol_1 - Stock_Name_1', 'Stock_Symbol_2 - Stock_Name_2', ...]
    ex. get ['ZM', 'ZS', 'XEL', ...] from ['ZM - Zoom Video Communications, Inc.', 'ZS - Zscaler, Inc.', 'XEL - Xcel Energy Inc.', ...]
    """
    selected_stock_symbols = [selected_stock_symbol_name.split(' - ', 1)[0] for selected_stock_symbol_name in selected_stock_symbols_names]
    return selected_stock_symbols


def get_all_stocks_for_index(selected_index_symbol):
    """
    Get list of all stocks (stored in the database) for a specific stock index symbol
    """
    selected_index_symbol = selected_index_symbol.replace('^', '')
    # Query database to get all stocks contained in an index
    cursor = db_collection.aggregate([{ '$match' : { f'Index_{selected_index_symbol}': True}
                                      }, 
                                      { '$group': {'_id'   : '$Symbol',
                                                   'Symbol': { '$first': '$Symbol' },
                                                   'Name'  : { '$first': '$Name'   }
                                                  }
                                      },
                                      { '$project': {'_id':     0,
                                                     'Symbol':  1,
                                                     'Name':    1,
                                                    }
                                      }
                                     ])
    query_result = list(cursor)
    return query_result


def get_stock_index_data(selected_index_symbol):
    """
    Get data (from the database) for a specific stock index symbol
    """
    # Query database to get closing prices for selected stock index:
    selected_index_cursor = db_collection.find({ 'Symbol': selected_index_symbol },
                                                     {'_id':    0,
                                                      'Date':   1,
                                                      'Symbol': 1,
                                                      'Close': 1
                                                     }
                                                     )
    query_result_selected_index = list(selected_index_cursor)
    # Build pandas dataframe with stock index data got from database.
    # ex. 
    #    Date (df index)         ^GSPC (stock index symbol)
    #    2023-07-25 04:00:00     4567.459961
    #    2023-07-26 04:00:00     4566.750000
    #    2023-07-27 04:00:00     4537.410156
    #    2023-07-28 04:00:00     4582.229980
    #    2023-07-31 04:00:00     4588.959961
    #    2023-08-01 04:00:00     4577.919922
    #
    data_stock_index = pd.DataFrame(query_result_selected_index)
    data_stock_index = data_stock_index.rename(columns={'Close': data_stock_index['Symbol'].iloc[0]})
    data_stock_index.drop(columns=['Symbol'], inplace=True)
    
    return data_stock_index

def get_stocks_data(selected_stock_symbols):
    """
    Get data (from the database) for a list of stock symbols
    """
    # Query database to get closing prices for selected stocks:
    selected_stocks_cursor = db_collection.find({ 'Symbol': {"$in": selected_stock_symbols}},
                                                     {'_id':    0,
                                                      'Date':   1,
                                                      'Symbol': 1,
                                                      'Close': 1
                                                     }
                                                     )
    query_result_selected_stocks = list(selected_stocks_cursor)
    # Build pandas dataframe with stocks data got from database (structure it to have name of column containing closing price for specific stock, equal to the symbol of the stock)
    # ex. 
    #    Date  (df index)       XEL          ZM           ZS => stock symbols 
    #    2023-07-25 04:00:00    64.959999    69.330002    155.210007
    #    2023-07-26 04:00:00    65.050003    71.099998    156.619995
    #    2023-07-27 04:00:00    62.869999    72.389999    155.250000
    #    2023-07-28 04:00:00    62.889999    73.080002    157.490005
    #    2023-07-31 04:00:00    62.730000    73.349998    160.380005
    #    2023-08-01 04:00:00    62.930000    72.485001    163.940002
    #
    data_stocks_query = pd.DataFrame(query_result_selected_stocks)
    gb = data_stocks_query.groupby('Symbol')
    data_stocks = pd.DataFrame()
    for x in gb.groups:
        if data_stocks.empty:
            df_temp = gb.get_group(x)
            df_temp = df_temp.rename(columns={'Close': df_temp['Symbol'].iloc[0]})
            df_temp.drop(columns=['Symbol'], inplace=True)
            data_stocks = df_temp
            del df_temp
        else:
            df_temp = gb.get_group(x)
            df_temp = df_temp.rename(columns={'Close': df_temp['Symbol'].iloc[0]})
            df_temp.drop(columns=['Symbol'], inplace=True)
            data_stocks = pd.merge(data_stocks, df_temp, on='Date')
            del df_temp
    
    return data_stocks


app.layout = dmc.MantineProvider(
    theme={
        #"fontFamily": "'Inter', sans-serif",
        "primaryColor": "indigo",
        "components": {
            "Button": {"styles": {"root": {"fontWeight": 400}}},
            "Alert": {"styles": {"title": {"fontWeight": 500}}},
            "AvatarGroup": {"styles": {"truncated": {"fontWeight": 500}}},
         },
    },
    inherit=True,
    withGlobalStyles=True,
    withNormalizeCSS=True,
    children=[
    dmc.Space(h=20),
    dmc.Grid([
        dmc.Col([dmc.Title(f"Multi-Variate Index Regression", order=4),
                 dmc.Select(label='Select Stock Index',
                            placeholder='You can select only 1 stock index',
                            id='stock-indices-dropdown',
                            data=get_all_index_symbols(db_collection=db_collection),
                            value='^NDX - NASDAQ 100',
                            style={"width": 300}
                           ),
                 dmc.MultiSelect(label='Select Stocks',
                                 description='You can select between 1 and 10 stocks',
                                 id='stocks-dropdown',
                                 #value=['ZM - Zoom Video Communications, Inc.'],
                                 value=[],
                                 maxSelectedValues=10,
                                 style={"width": 300},
                                 clearable=True,
                                 searchable=True),
                 dmc.Space(h=10),
                 dmc.Alert("Unfortunately there was a problem in \nestimating the regression model with the \nselected stocks probably for lack of data \nfor some symbols. Please remove the last \nselected stock(s) from the box or try a different group of stocks.",
                           id='regression-model-alert',
                           title="ALERT!",
                           color="red",
                           duration=5000,
                           hide=True,
                           style={"width": 350,
                                  "white-space": "pre"}),
                 dmc.Space(h=30),
                 dmc.Text('Press the below button to automatically select the \nstocks (10 or less) that best explain the index \n(selected above):',
                          align="left",
                          size="sm",
                          style={"width": 350,
                                 "white-space": "pre"}),
                 dmc.Button("Select Best Stocks", id='button-best-stocks'),
                 dmc.Space(h=10),
                 dmc.Alert("I'm Selecting the Stocks! Please wait.",
                           id='computing-stock-selection-alert',
                           title="INFO",
                           color="yellow",
                           hide=True),
                 dmc.Space(h=10),
                 dmc.Alert("Unfortunately there was a problem in \nselecting the stocks that best explain \nthe index. \nPlease select them manually in the box.",
                           id='best-stocks-alert',
                           title="ALERT!",
                           color="red",
                           duration=5000,
                           hide=True,
                           style={"width": 350,
                                  "white-space": "pre"}),
                ], span=3, offset=0.25),
        dmc.Col([dcc.Graph(figure={}, id="scatter-plot"),
                ], span=5, offset=-0.75),
        dmc.Col([html.Div(id='ols-summary-table'),
                ], span=4),
    ]),
])

@app.callback(Output('stocks-dropdown', 'data'),
              Output('ols-summary-table', 'children'),
              Output('scatter-plot', 'figure'),
              Input('stock-indices-dropdown', 'value'),
              prevent_initial_call=True)
def get_all_stocks_for_index_callback(selected_index_symbol_name):
    selected_index_symbol = extract_index_symbol(selected_index_symbol_name)
    query_result = get_all_stocks_for_index(selected_index_symbol=selected_index_symbol)

    # Convert list of stock symbols into list of strings in the format 'Stock_Symbol - Company_Name' (ex. 'ZM - Zoom Video Communications, Inc.' for symbol 'ZM') and drop Index Symbol from list
    symbols_names = [value for value in [f'{value1} - {value2}' for value1, value2 in [list(dict_.values()) for dict_ in query_result]] if value != selected_index_symbol_name]
    # Reset OLS regression model results table
    ols_table_children = []
    # Reset scatterplot figure
    figure = {}
    return symbols_names, ols_table_children, figure


@app.callback(Output("stocks-dropdown", "error"),
              Input("stocks-dropdown", "value"))
def select_value(value):
    return "Select at least 1 stock" if len(value) < 1 else ""


@app.callback(Output('ols-summary-table', 'children', allow_duplicate=True),
              Output('scatter-plot', 'figure', allow_duplicate=True),
              Output("regression-model-alert", "hide"),
              State('stock-indices-dropdown', 'value'),
              Input('stocks-dropdown', 'value'),
              State("regression-model-alert", "hide"),
              prevent_initial_call=True)
def multiple_linear_regression(selected_index_symbol_name, selected_stock_symbols_names, hide):
    # Make sure the list of symbols is not empty
    if not selected_stock_symbols_names:
        return DangerouslySetInnerHTML('''<br>'''), {}, hide
    
    # Query database for data
    # Query for stock index data
    selected_index_symbol = extract_index_symbol(selected_index_symbol_name)
    data_stock_index = get_stock_index_data(selected_index_symbol=selected_index_symbol)
    # Query for stocks data
    selected_stock_symbols = extract_stock_symbols(selected_stock_symbols_names)
    data_stocks = get_stocks_data(selected_stock_symbols=selected_stock_symbols)
    
    # Fill NANs by forward filling
    data_stock_index.ffill(inplace=True)
    data_stocks.ffill(inplace=True)
    
    # Make sure the dataframes for the stock index and the single stocks have the same samples to avoid problems in the estimation of the regression model
    data_stock_index = pd.merge(data_stock_index, data_stocks[['Date']], how='inner', on='Date')
    data_stocks = pd.merge(data_stock_index[['Date']], data_stocks, how='inner', on='Date')
    
    # Set 'Date' as index on both dataframes
    data_stock_index.set_index('Date', inplace=True)
    data_stocks.set_index('Date', inplace=True)
    
    try:
        y = data_stock_index
        # define predictor variables
        x = data_stocks
        # add constant to predictor variables
        x = sm.add_constant(x)
        # fit linear regression model (I drop missing values)
        model = sm.OLS(y, x, missing='drop').fit()

        predictions = model.predict(x).to_frame()
        predictions.rename(columns={predictions.columns[0]:'model_prediction'}, inplace=True)
        df = pd.merge(y, predictions, on='Date')

        fig = px.scatter(df, x="model_prediction", y=selected_index_symbol)
        result = model.summary().as_html()
        result = DangerouslySetInnerHTML(result)
        
        return result, fig, hide
    except:
        hide = False
        return no_update, no_update, hide


@app.callback(Output("computing-stock-selection-alert", "hide", allow_duplicate=True),
              Input("button-best-stocks", "n_clicks"),
              State("computing-stock-selection-alert", "hide"),
              prevent_initial_call=True)
def button_pressed_callback(n_clicks_button, hide_computing_stock_selection_alert):
    hide_computing_stock_selection_alert = False
    return hide_computing_stock_selection_alert


@app.callback(Output('stocks-dropdown', 'value', allow_duplicate=True),
              Output("best-stocks-alert", "hide"),
              Output("computing-stock-selection-alert", "hide"),
              State('stock-indices-dropdown', 'value'),
              Input("button-best-stocks", "n_clicks"),
              State("best-stocks-alert", "hide"),
              State("computing-stock-selection-alert", "hide"),
              prevent_initial_call=True)
def select_10_best_stocks(selected_index_symbol_name, n_clicks, hide_best_stocks_alert, hide_computing_stock_selection_alert):
    # Query database for data
    selected_index_symbol = extract_index_symbol(selected_index_symbol_name)
    data_stock_index = get_stock_index_data(selected_index_symbol=selected_index_symbol)
    
    # Query for stocks data:
    # Get all stock symbols for selected stock index.
    # Query result will contain a list like:
    #  [{'Symbol': 'ZM', 'Name': 'Zoom Video Communications, Inc.'},
    #   {'Symbol': 'ZS', 'Name': 'Zscaler, Inc.'},
    #   {'Symbol': '^NDX', 'Name': 'NASDAQ 100'},
    #   {'Symbol': 'XEL', 'Name': 'Xcel Energy Inc.'},
    #   ...]
    query_result = get_all_stocks_for_index(selected_index_symbol=selected_index_symbol)
    stock_index_symbol_name_list = get_all_index_symbols(db_collection=db_collection)
    stock_index_symbols_list = [extract_index_symbol(stock_index_symbol_name) for stock_index_symbol_name in stock_index_symbol_name_list]
    # Make sure to remove stock index symbol from list of stocks
    selected_stock_symbols = [symbol_name_dict['Symbol'] for symbol_name_dict in query_result if symbol_name_dict['Symbol'] not in stock_index_symbols_list]
    data_stocks = get_stocks_data(selected_stock_symbols=selected_stock_symbols)
    
    # Fill NANs by forward filling
    data_stock_index.ffill(inplace=True)
    data_stocks.ffill(inplace=True)
    
    # Make sure the dataframes for the stock index and the single stocks have the same samples to avoid problems in the estimation of the regression model
    data_stock_index = pd.merge(data_stock_index, data_stocks[['Date']], how='inner', on='Date')
    data_stocks = pd.merge(data_stock_index[['Date']], data_stocks, how='inner', on='Date')
    
    # Set 'Date' as index on both dataframes
    data_stock_index.set_index('Date', inplace=True)
    data_stocks.set_index('Date', inplace=True)
    
    try:
        feature_selection = 'SelectFromModel'
        #feature_selection = 'SequentialFeatureSelector'
        
        # Select the best 10 stocks/features (or less)
        if feature_selection == 'SelectFromModel':
            # Feature/stock selection done using an XGBoost model and the SelectFromModel function from scikit-learn (it uses the feature importance from the model passed as parameter)
            # I use some preset hyperparameters' values for the XGBoost but a grid search could be applied to optimize these hyperparameters for the specific problem (e.g., by using bayesian hyperparameter
            # optimization, random grid search or a full grid search)
            # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
            model = xgb.XGBRegressor(objective='reg:squarederror',
                                     booster='gbtree',
                                     n_estimators=100,
                                     random_state=42,
                                     n_jobs=-1)
            sfm_selector = SelectFromModel(estimator=model,
                                           max_features=10,  # we want to select max 10 features
                                           prefit=False)
            sfm_selector.fit(X=data_stocks,
                             y=data_stock_index.values.ravel())
            best_stocks = sfm_selector.get_feature_names_out()
        elif feature_selection == 'SequentialFeatureSelector':
            # Feature/stock selection done with the SequentialFeatureSelector. This method adds (forward selection) or removes (backward selection) features to form a feature subset in a greedy fashion.
            # It should give better results than the SelectFromModel approach but it takes longer to execute, in particular if the number of stocks/features is of decent size.
            # I use a linear regression model to speed up the computation.
            # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html
            model = LinearRegression()
            # We want to select 10 features/stocks if possible but if there are less in the dataset we set the 'n_features_to_select' variable to 'auto'
            if data_stocks.shape[1] < 10:
                n_features_to_select = 'auto'
            else:
                n_features_to_select = 10
            sfs_selector = SequentialFeatureSelector(estimator=model,
                                                     n_features_to_select=n_features_to_select,
                                                     tol=None,
                                                     direction='forward',
                                                     #direction='backward',
                                                     scoring='r2',
                                                     n_jobs=-1)
            sfs_selector.fit(X=data_stocks,
                             y=data_stock_index.values.ravel())
            best_stocks = sfs_selector.get_feature_names_out()

        # Once we have selected the best stocks as list (e.g., ['ZM', 'ZS', 'XEL', ...]) we build back the list containing the stock names as well
        # (e.g., ['ZM - Zoom Video Communications, Inc.', 'ZS - Zscaler, Inc.', 'XEL - Xcel Energy Inc.', ...]) and pass this to the Dash component with id 'stocks-dropdown'
        # which will trigger the callback to estimate the linear regression model.
        
        # Get stock names linked to stock symbols
        best_stocks_symbols_names_list_of_dict = [symbol_name_dict for symbol_name_dict in query_result if symbol_name_dict['Symbol'] in best_stocks]
        # Convert list of stock symbols into list of strings in the format 'Stock_Symbol - Company_Name' (ex. 'ZM - Zoom Video Communications, Inc.' for symbol 'ZM') and drop Index Symbol from list
        best_stocks_symbols_names_list = [value for value in [f'{value1} - {value2}' for value1, value2 in [list(dict_.values()) for dict_ in best_stocks_symbols_names_list_of_dict]]]

        hide_computing_stock_selection_alert = True
        return best_stocks_symbols_names_list, hide_best_stocks_alert, hide_computing_stock_selection_alert
    except:
        hide_computing_stock_selection_alert = True
        hide_best_stocks_alert = False
        return no_update, hide_best_stocks_alert, hide_computing_stock_selection_alert


if __name__ == '__main__':
    app.run_server(debug=False)
