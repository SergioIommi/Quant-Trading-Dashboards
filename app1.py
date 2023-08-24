from dash import Dash, dcc, Input, Output, State, callback, no_update, dash_table
import dash_mantine_components as dmc
import plotly.express as px
import pymongo
import numpy as np
import pandas as pd
import datetime
import statsmodels.api as sm
from pykalman import KalmanFilter
import quantstats as qs

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
db_collection = db[db_collection_name]

def get_db_min_max_dates(min_max_str='min'):
    """
    Get min/max dates stored in database
    """
    cursor = db_collection.find({},
                                {'_id':    0,
                                 'Date':   1,
                                })
    query_result = list(cursor)
    db_dates = pd.DataFrame(query_result)
    if min_max_str == 'min':
        return db_dates['Date'].min().date()
    elif min_max_str == 'max':
        return db_dates['Date'].max().date()

def get_data(date_start, date_end):
    """
    Get data from database
    """
    date_end = date_end+datetime.timedelta(days=1)  # increment 1 day to include the day in the query, otherwise it is discarded by MongoDB
    cursor = db_collection.find({'Date': {'$gt': date_start,
                                          '$lt': date_end
                                         }
                                },
                                {'_id':    0,
                                 'Date':   1,
                                 'Symbol': 1,
                                 'Close': 1
                                })
    query_result = list(cursor)

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
    data_stocks_query = pd.DataFrame(query_result)
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
            data_stocks = pd.merge(data_stocks, df_temp, on='Date', how='outer')
            del df_temp
    data_stocks.set_index('Date', inplace=True)
    return data_stocks


def select_pairs_correlation(data_stocks, n_pairs_to_select=5):
    """
    Select the stock pairs with the highest correlation
    """
    # n_pairs_to_select: variable to specify the number of the most highly correlated pairs to select
    
    # In this step I pre-select the most highly correlated pirs of stocks (including stock indices as well) to later compute the model for the mean-reverting spread (using OLS regression or the
    #    Kalman filter for a state-space model) only on the these selected pairs. The main aim of this procedure is to reduce the computational time/costs and avoid estimating models for pairs that aren't
    #    promising candidates.

    # Compute correlation matrix (matrix containing correlation for each pair of stocks) and convert it to a more usable format to select the top correlated pairs according to absolute correlation
    #    ex.
    #        SymbolA  SymbolB  Corr       AbsCorr   SignCorr
    #    0   ^GSPC    ^NDX     0.972411   0.972411  1.0
    #    1   ZD       ZIP      0.949294   0.949294  1.0
    #    2   ZEUS     ^RUT     0.942167   0.942167  1.0
    #    3   YELP     ^RUT     0.930411   0.930411  1.0
    #    4   WY       ZD       0.923649   0.923649  1.0
    corr = data_stocks.corr(method='pearson')
    corr = corr.where(np.triu(np.ones(corr.shape)).astype(bool))
    corr = corr.stack().reset_index()
    corr.columns = ['SymbolA','SymbolB','Corr']
    corr = corr[corr['SymbolA'] != corr['SymbolB']]
    corr['AbsCorr'] = corr['Corr'].abs()
    corr['SignCorr'] = np.sign(corr['Corr'])
    corr.sort_values('AbsCorr', inplace=True, ascending=False)
    corr.reset_index(inplace=True, drop=True)

    # Select most highly correlated pairs
    selected_pairs = corr.iloc[0:n_pairs_to_select]

    return selected_pairs
    

def half_life(spread):
    """
    Compute the half-life of the spread model which is an auto-regressive AR(1) model
    https://mathtopics.wordpress.com/2013/01/10/half-life-of-the-ar1-process/
    """
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    spread_ret = spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    spread_lag2 = sm.add_constant(spread_lag)
    model = sm.OLS(spread_ret, spread_lag2)
    res = model.fit()
    halflife = int(round(-np.log(2) / res.params[1],0))
    if halflife <= 0:
        halflife = 1
    return halflife

    
def estimate_spread_model(data_stocks, selected_model, selected_pairs):
    """
    Estimate the parameters alpha/beta for the spread model (Spread = StockB - beta*StockA - alpha) with one of the following 2 models/methods:
        - linear regression model with OLS estimator (beta/alpha fixed): Spread_t = StockB_t - beta*StockA_t - alpha (t=time)
        - state-space model with Kalman filter estimation algorithm (beta_t/alpha_t dynamic): Spread_t = StockB_t - beta_t*StockA_t - alpha_t (t=time)
    """
    # Build a nested dictionary to store the information related to each of the selected pairs (symbols, correlation, half-life/speed of mean reversion, parameters of the OLS/Kalman filter models to estimate, etc.)
    selected_pairs_dict = dict()
    for i, row in selected_pairs.iterrows():
        selected_pairs_dict[f'Pair_{i}'] = {'SymbolA': row['SymbolA'],
                                            'SymbolB': row['SymbolB'],
                                            'Corr': row['Corr'],
                                            'Half-Life': None
                                            }

        # Build Model for the (Mean-Reverting) Spread of the form (t=time):
        #        OLS Linear Regression:  Spread_t = PriceStockA_t - beta * PriceStockA_t - alpha
        #        SSM with Kalman Filter: Spread_t = PriceStockA_t - beta_t * PriceStockA_t - alpha_t
        # The main difference between the 2 models is that the parameters of the OLS Linear Regression are fixed meanwhile the ones estimated using the Kalman Filter (in the state-space modelling framework) are
        # dynamic and change in time

        # I use directly the prices to estimate the regression model but other approaches could be tested as well (log prices, returns, etc.)
        #x = np.log(data_stocks[row['SymbolA']])
        x = data_stocks[row['SymbolA']]
        #y = np.log(data_stocks[row['SymbolB']])
        y = data_stocks[row['SymbolB']]
        
        # Fill NANs by forward filling
        x.ffill(inplace=True)
        y.ffill(inplace=True)
        
        # Make sure the dataframes for the stocks have the same samples to avoid problems in the estimation of the models
        # Convert series to dataframe
        x = x.to_frame()
        y = y.to_frame()
        x.reset_index(inplace=True)
        y.reset_index(inplace=True)
        # Inner merge to have same dates between the 2 series of prices
        x = pd.merge(x, y[['Date']], how='inner', on='Date')
        y = pd.merge(x[['Date']], y, how='inner', on='Date')
        # Set 'Date' as index on both dataframes
        x.set_index('Date', inplace=True)
        y.set_index('Date', inplace=True)
        # Convert back to series
        x = x.squeeze()
        y = y.squeeze()

        if selected_model == 'ols':
            # First estimate OLS linear regression model to get alpha and beta using the prices of the 2 selected stocks/stock indices. The alpha and beta will be used to build the model for the spread.
            x_const = sm.add_constant(x)  # add constant to predictor variable
            model = sm.OLS(y, x_const, missing='drop').fit()
            # Get OLS parameters
            alpha = model.params[0]
            beta = model.params[1]
        elif selected_model =='kalman':
            # Estimate a model for the spread (for the prices of the 2 selected stocks/stock indices) directly by making use of a state-space modelling approach and estimating the alpha
            # and beta of the spread model by using a Kalam filter.
            obs_mat = sm.add_constant(x.values, prepend=False)[:, np.newaxis]
            delta = 1e-5
            trans_cov = delta / (1 - delta) * np.eye(2)
            kf = KalmanFilter(n_dim_obs=1,  # y is 1-dimensional
                              n_dim_state=2,  # (alpha, beta) is 2-dimensional
                              initial_state_mean=np.ones(2),
                              initial_state_covariance=np.ones((2, 2)),
                              transition_matrices=np.eye(2),
                              observation_matrices=obs_mat,
                              observation_covariance=0.5,
                              transition_covariance=trans_cov)
            state_means, state_covs = kf.filter(y.values)
            alpha=state_means[:, 1]
            beta=state_means[:, 0]
        # Build the spread time-series using the parameters alpha/beta estimated via OLS/Kalman Filter and store it as a pandas dataframe
        # This spread, hopefully mean-reverting, can be used as the main input to build the signal for the trading strategy later on
        df_spread = pd.DataFrame(y - x*beta - alpha, index = data_stocks.index)
        df_spread.columns = ['spread']
        # Store the resulting spread in the selected_pairs_dict dictionary. I serialize the dataframe to JSON to share it between Dash callbacks (using dcc.Store).
        selected_pairs_dict[f'Pair_{i}']['spread'] = df_spread['spread'].to_json()
        # Compute half-life/speed of mean reversion and add it to selected_pairs_dict dictionary
        selected_pairs_dict[f'Pair_{i}']['Half-Life'] = half_life(df_spread)
        # Store beta which is equal to the hedge-ratio (to compute the strategy returns in the backtesting)
        if isinstance(beta, np.ndarray):
            beta = beta.tolist()  # Convert numpy ndarray to list to make it serializable to JSON to share it between Dash callbacks (using dcc.Store).
        selected_pairs_dict[f'Pair_{i}']['Beta'] = beta
        # Store the the prices for both stocks in the selected_pairs_dict dictionary (to be used later for computing the backtests). I could have queried them again from the database in the later steps
        # and avoid duplicate data, but I prefer storing them in the dictionary and then in the browser session (in memory) for simplicity and with the assumption that the number of selected pairs aren't
        # that big. I serialize the dataframes as JSON to share them between Dash callbacks (using dcc.Store).
        selected_pairs_dict[f'Pair_{i}']['x'] = data_stocks[row['SymbolA']].to_json()
        selected_pairs_dict[f'Pair_{i}']['y'] = data_stocks[row['SymbolB']].to_json()
        
    return selected_pairs_dict


def run_backtest(selected_pairs_dict, zscore_entry = 1.5, zscore_exit = 0):
    """
    Run simple backtest on each of the selected stock pairs and compute:
        - time-series of the cumulative returns
        - Sharpe ratio
        - CAGR (compound annual growth rate)
    """
    for key, pair in selected_pairs_dict.items():
        # Deserialize JSON pandas series stored in dict (loaded from dcc.Store)
        df_backtest = pd.read_json(pair['spread'], typ='series').to_frame('spread')
        # To compute the returns of the strategy we need to compute the actual change in the spread, and to do that we need the 'beta' of the estimated models and the prices of the 2 stocks used to build the spread
        df_backtest['beta'] = pair['Beta']
        # Deserialize JSON pandas series stored in dict (loaded from dcc.Store)
        df_backtest['x'] = pd.read_json(pair['x'], typ='series').to_frame('x')
        df_backtest['y'] = pd.read_json(pair['y'], typ='series').to_frame('y')
        epsilon = 1e-6  # small value to be used in the divisions to make sure we don't divide by zero

        # The core of the trading strategy is the alpha signal, which in this case it the z-score of the spread (or the z-score normalized spread):
        #    z-score(spread) = ( spread - mean(spread) ) / std(spread)
        # I use the z-score of the spread instead of the actual spread as the input of the alpha signal because each pair of stocks could have a spread with a standard deviation that
        # is rather different, hence the entry and exit (that are based on how 'far' we move away from the spread mean) for the trading strategy would need to be adapted for each pair of stocks.
        # By using the z-score instead the spread is 'normalized' so that the entry and exit for each pair can be compared across all the pairs, and in particular the range of values for the entry
        # and exit proposed to the user (with 2 sliders in the GUI) can be the same for all pairs.
        # In particular there are few possibilities to compure the z-score, a dynamic one which uses the rolling_mean and rolling_std (with a time-window of predefined width), or a fixed vcersion
        # with the mean and std computed over the entire period.
        # In a normal setting of designing a trading strategy for live trading, using the rolling version is to be preferred because it is dynamic and it is more suited to adapt to changing market
        # regimes. With a fixed mean and std computed over historical data there could be the problem that they're unable to capture new market regimes (in terms of these 2 statistics) and there
        # would be the need for some monitoring and an actual discretionary decision (from a quant trader/researcher) as to when update the mean and std used in the computation of the z-score to
        # account for the new market regime observed.
        
        # In the following I use a fixed z-score with fixed mean and std.
        # Someting to underline is the fact that by using these values I introduce a look-ahead bias in the trading strategies, hence the results of the backtests have to be discounted for this.
        # For this specific project I consider it ok given that the goal isn't that of building full proof trading strategies to be deployed live but to evaluate other skillsets.
        spread_mean = df_backtest['spread'].mean()
        spread_std  = df_backtest['spread'].std()
        # For completeness, I leave the code to compute the rolling window z-score of the spread (with window width equal to half-life period)
        #halflife = pair['Half-Life']
        #spread_mean = df_backtest['spread'].rolling(window=halflife).mean()
        #spread_std  = df_backtest['spread'].rolling(window=halflife).std()
        
        # Compute the z-score
        df_backtest['zscore'] = (df_backtest['spread'] - spread_mean) / (spread_std + epsilon)
        # Save z-score on dictionary for plotting later
        selected_pairs_dict[key]['zscore'] = df_backtest['zscore'].to_json()

        # Longs:
        # Compute the instant/day when we open/close each long:
        df_backtest['long_open'] = np.where(np.logical_and(df_backtest['zscore'] < -zscore_entry, df_backtest['zscore'].shift(1) > -zscore_entry), True, False)
        df_backtest['long_close'] = np.where(np.logical_and(df_backtest['zscore'] > -zscore_exit, df_backtest['zscore'].shift(1) < -zscore_exit), True, False)
        # Compute for each instant/day if a long position is open or not (1 for open long position and 0 for no-long position)
        df_backtest['long_position'] = np.where(df_backtest['long_open'], 1,
                                      (np.where(df_backtest['long_close'], 0, np.nan)))
        df_backtest['long_position'] = df_backtest['long_position'].ffill().fillna(0)

        # Shorts:
        # Compute the instant/day when we open/close each short
        df_backtest['short_open'] = np.where(np.logical_and(df_backtest['zscore'] > zscore_entry, df_backtest['zscore'].shift(1) < zscore_entry), True, False)
        df_backtest['short_close'] = np.where(np.logical_and(df_backtest['zscore'] < zscore_exit, df_backtest['zscore'].shift(1) > zscore_exit), True, False)
        # Compute for each instant/day if a short position is open or not (1 for open short position and 0 for no-short position)
        df_backtest['short_position'] = np.where(df_backtest['short_open'], -1,
                                       (np.where(df_backtest['short_close'], 0, np.nan)))
        df_backtest['short_position'] = df_backtest['short_position'].ffill().fillna(0)

        # Compute trading strategy return
        df_backtest['long_short_position'] = df_backtest['long_position'] + df_backtest['short_position']
        df_backtest['spread_change_pct'] = (df_backtest['spread'] - df_backtest['spread'].shift(1)) / ((df_backtest['x'] * abs(df_backtest['beta'])) + df_backtest['y'])
        df_backtest['return'] = df_backtest['spread_change_pct'] * df_backtest['long_short_position'].shift(1)
        df_backtest['return'].fillna(0, inplace=True)
        start_portfolio_value = 1
        df_backtest['cumulative_return'] = start_portfolio_value + df_backtest['return'].cumsum()
        # Save cumulative return on dictionary for plotting later
        selected_pairs_dict[key]['cumulative_return'] = df_backtest['cumulative_return'].to_json()

        # Compute Sharpe ratio
        sharpe = np.round(np.sqrt(252) * df_backtest['return'].mean() / ( df_backtest['return'].std() + epsilon ), 2)

        # Compute CAGR (compound annual growth rate)
        end_portfolio_value = df_backtest['cumulative_return'].iloc[-1]
        trading_days = (df_backtest.index[-1].date() - df_backtest.index[0].date()).days
        cagr = (end_portfolio_value / start_portfolio_value) ** (252/trading_days) - 1

        # Update dictionary with Sharpe and CAGR
        selected_pairs_dict[key]['Sharpe'] = sharpe
        selected_pairs_dict[key]['CAGR (%)'] = np.round(cagr*100,1)
        
    return selected_pairs_dict

def run_backtest_details(df_cumret):
    """
    Run detailed backtest on each of the selected stock pairs to compute few risk/return metrics
    """
    # Compute risk-return metrics and store them in a pandas dataframe
    data = [qs.stats.cagr(df_cumret)*100,
            qs.stats.sharpe(df_cumret),
            qs.stats.sortino(df_cumret),
            qs.stats.max_drawdown(df_cumret)*100,
            qs.stats.volatility(df_cumret)*100,
            qs.stats.skew(df_cumret),
            qs.stats.kurtosis(df_cumret)
           ]
    data = [np.round(value,2) for value in data]
    index=['CAGR (%)',
           'Sharpe',
           'Sortino',
           'Max Drawdown (%)',
           'Volatility (%)',
           'Skew',
           'Kurtosis']
    risk_return_metrics = pd.DataFrame(data=data,
                                       index=index,
                                       columns=['Risk-Return Metrics'])
    
    risk_return_metrics
    risk_return_metrics.reset_index(inplace=True)
    risk_return_metrics.rename(columns={'index': 'Risk-Return Metrics', 'Risk-Return Metrics': 'Value'}, inplace=True)
    
    return risk_return_metrics


models_to_select = [['ols', 'OLS Linear Regression'], ['kalman', 'Kalman Filter (State-Space Model)']]
    
app.layout = dmc.MantineProvider(
    theme={
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
    # To share data between Dash callbacks I initialise the dcc.Store to store JSON data in the browser session (in memory)
    dcc.Store(id='browser-session-memory-storage', storage_type='memory'),
    dmc.Space(h=20),
    dmc.Grid([
        dmc.Col([dmc.Title(f"Equities Pair-Trading", order=4),
                 dmc.DateRangePicker(id="date-range-picker",
                                     label="Select date range (max interval pre-selected):",
                                     minDate=get_db_min_max_dates(min_max_str='min'),
                                     maxDate=get_db_min_max_dates(min_max_str='max'),
                                     value=[get_db_min_max_dates(min_max_str='min'), get_db_min_max_dates(min_max_str='max')],
                                     style={"width": 300},
                                     clearable=False,
                                    ),
                 dmc.Space(h=10),
                 dmc.RadioGroup([dmc.Radio(l, value=k) for k, l in models_to_select],
                                id="model-selector",
                                #value='ols',
                                label="Select model to estimate alpha/beta for the spread \n(Spread = StockB - beta*StockA - alpha):",
                                size="sm",
                                mt=10,
                                style={"width": 270,
                                       "white-space": "pre"},
                               ),
                 dmc.Space(h=10),
                 dmc.Alert("Estimating Models! Please wait.",
                           id='model-estimation-alert',
                           title="INFO",
                           color="yellow",
                           hide=True),
                 dmc.Space(h=10),
                 dmc.Alert("Unfortunately there was a problem in \nestimating the models probably for a lack of data \nfor some symbols. Please try changing the \ndate-interval (making it wider or moving it).",
                            id='model-estimation-exception-alert',
                            title="ALERT!",
                            color="red",
                            duration=5000,
                            hide=True,
                            style={"width": 350,
                                   "white-space": "pre"}),
                 dmc.Space(h=10),
                 dash_table.DataTable(id='correlation-table',
                                      style_data={'whiteSpace': 'normal',
                                                  'height': 'auto',
                                                  'lineHeight': '15px'
                                                 },
                                     ),
                ], span=2, offset=0.25),
        dmc.Col([
            dmc.Text("Select Z-Score entry/exit values (1.5 and 0 preselected are suggested as starting ones):", align="left", size="lg"),
            dmc.Space(h=10),
            dmc.Text("Select Z-Score entry value:", align="left", size="sm"),
            dmc.Slider(id="drag-slider-zscore-entry",
                       min=0,
                       max=3,
                       step=0.1,
                       value=1.5,
                       updatemode="drag",
                       marks=[{"value": value, 'label': f'{value}'} for value in np.arange(0,3.1,0.5)]+[{"value": value} for value in np.arange(0,3.1,0.1)]
                      ),
            dmc.Space(h=20),
            dmc.Text(id='drag-slider-zscore-entry-output', align="left", size="sm"),
            dmc.Space(h=20),
            dmc.Text("Select Z-Score trade exit value (must be smaller than entry Z-Score):", align="left", size="sm"),
            dmc.Slider(id="drag-slider-zscore-exit",
                       min=0,
                       max=3,
                       step=0.1,
                       value=0,
                       updatemode="drag",
                       marks=[{"value": value, 'label': f'{value}'} for value in np.arange(0,3.1,0.5)]+[{"value": value} for value in np.arange(0,3.1,0.1)]
                      ),
            dmc.Space(h=20),
            dmc.Text(id='drag-slider-zscore-exit-output', align="left", size="sm"),
            dmc.Space(h=10),
            dmc.Alert("Selected ZScore exit > ZScore entry! Please check and update the values selected.",
                      id='drag-slider-zscore-exit-alert',
                      title="ALERT!",
                      color="red",
                      hide=True),
            dmc.Space(h=10),
            dmc.Text(id='backtest-table-descr', align="left", size="sm"),
            dash_table.DataTable(id='backtest-table',
                                 #style_as_list_view=True,
                                 row_selectable='single',
                                 selected_rows=[],
                                 style_data={'whiteSpace': 'normal',
                                             'height': 'auto',
                                             'lineHeight': '15px'
                                            },
                                ),
        ], span=3, offset=0.5),
        dmc.Col([
            dmc.Text(id='backtest-details-table-descr', children='', align="left", size="sm"),
            dash_table.DataTable(id='backtest-details-table',
                                 data=[],
                                 columns=[],
                                 #style_as_list_view=True,
                                 style_data={'whiteSpace': 'normal',
                                             'height': 'auto',
                                             'lineHeight': '15px',
                                            },
                                ),
            dcc.Graph(figure={}, id="cumulative-return-plot"),
            dcc.Graph(figure={}, id="zscore-plot"),
        ], span=5, offset=0.5),
    ]),
])


@app.callback(Output("model-estimation-alert", "hide"),
              Output('backtest-table-descr', 'children', allow_duplicate=True),
              Output('backtest-table', 'data', allow_duplicate=True),
              Output('backtest-table', 'columns', allow_duplicate=True),
              Output('backtest-details-table-descr', 'children', allow_duplicate=True),
              Output('backtest-details-table', 'data', allow_duplicate=True),
              Output('backtest-details-table', 'columns', allow_duplicate=True),
              Output('zscore-plot', 'figure', allow_duplicate=True),
              Output('cumulative-return-plot', 'figure', allow_duplicate=True),
              Input("model-selector", "value"),
              State("model-selector", "value"),
              State("model-estimation-alert", "hide"),
              prevent_initial_call=True)
def model_estimation_alert_callback(selected_model, previous_selected_model, hide_model_estimation_alert):
    if hide_model_estimation_alert == True and (selected_model or selected_model != previous_selected_model):
        # Clear all the dash items that need a refresh
        return not hide_model_estimation_alert, [], [], [], [], [], [], {}, {}

    
@app.callback(Output('browser-session-memory-storage', 'data'),
              Output('correlation-table', 'data'),
              Output('correlation-table', 'columns'),
              Output("model-estimation-alert", "hide", allow_duplicate=True),
              Output("model-estimation-exception-alert", "hide"),
              State('date-range-picker', 'value'),
              Input('model-selector', 'value'),
              State("model-estimation-exception-alert", "hide"),
              prevent_initial_call=True)
def select_stocks_callback(selected_dates, selected_model, hide_model_exception_alert):
    date_start, date_end = selected_dates
    date_format = '%Y-%m-%d'
    date_start = datetime.datetime.strptime(date_start, date_format)
    date_end = datetime.datetime.strptime(date_end, date_format)
    
    # Get the stocks data from the database soon as the user select the date-range and the model estimation method (OLS or Kalman). If the user is going to change the model once already selected there will be
    # another query to the database to get the same initial data. This could be separated in a different Dash callback but the approaches (described here https://dash.plotly.com/sharing-data-between-callbacks)
    # to then share this data betweek callbacks could be less efficient than a new query to the database, and given the performance of MongoDB and the small amount of data to query (daily vs intraday) my approach
    # is probably better. That said, the approaches suggested by Dash could be explored and tested.
    # That said, I have used the dcc.Store to store JSON data in the browser session (in memory) to share other data between other callbacks of this application.
    data_stocks = get_data(date_start, date_end)
    
    selected_pairs = select_pairs_correlation(data_stocks=data_stocks,
                                              n_pairs_to_select=20)
    
    try:
        selected_pairs_dict = estimate_spread_model(data_stocks=data_stocks,
                                                    selected_model=selected_model,
                                                    selected_pairs=selected_pairs)

        # Build table with relevant data for stock pairs and convert to format compatible for dash_table.DataTable
        selected_pairs_df = pd.DataFrame()
        for dict_ in selected_pairs_dict.values():
            dict_selected_keys = {key: [dict_[key]] for key in ['SymbolA', 'SymbolB', 'Corr', 'Half-Life']}
            df_temp = pd.DataFrame.from_dict(dict_selected_keys)
            selected_pairs_df = selected_pairs_df.append(df_temp)
        selected_pairs_df.reset_index(inplace=True, drop=True)
        selected_pairs_df = selected_pairs_df.round(3)
        selected_pairs_dash_table_data = selected_pairs_df.to_dict('records')
        selected_pairs_dash_table_columns = [{"name": i, "id": i} for i in selected_pairs_df.columns]

        # I return the data I need to share with the trading strategy backtester (callback run_backtest_callback) to store it in the browser session.
        # The data is in a dictionary so I don't need to serialise it in JSON beforehand (apart from the pandas dataframes that are already serialized
        # to JSON at the time I created the dictionary).
        hide_model_estimation_alert=True
        hide_model_exception_alert=True
        return selected_pairs_dict, selected_pairs_dash_table_data, selected_pairs_dash_table_columns, hide_model_estimation_alert, hide_model_exception_alert
    except:
        hide_model_estimation_alert=True
        hide_model_exception_alert = False
        return [], [], [], hide_model_estimation_alert, hide_model_exception_alert


@app.callback(Output("drag-slider-zscore-entry-output", "children"),
          Input("drag-slider-zscore-entry", "value")
)
def zscore_entry_callback(zscore_entry):
    return f"Selected value: {np.round(zscore_entry,2)}"


@app.callback(Output("drag-slider-zscore-exit-output", "children"),
          Input("drag-slider-zscore-exit", "value")
)
def zscore_exit_callback(zscore_exit):
    return f"Selected value: {np.round(zscore_exit,2)}"


@app.callback(Output("drag-slider-zscore-exit-alert", "hide"),
              Input("drag-slider-zscore-entry", "value"),
              Input("drag-slider-zscore-exit", "value"),
              State("drag-slider-zscore-exit-alert", "hide"),
              prevent_initial_call=True)
def zscore_alert_callback(zscore_entry, zscore_exit, hide):
    # Show alert in case of invalid selection of ZScore values
    if hide == True and zscore_entry <= zscore_exit:
        return not hide
    elif hide == False and zscore_entry > zscore_exit:
        return not hide
    else:
        return hide

    
@app.callback(Output('browser-session-memory-storage', 'data', allow_duplicate=True),
              Output('backtest-table-descr', 'children'),
              Output('backtest-table', 'data'),
              Output('backtest-table', 'columns'),
              Input('browser-session-memory-storage', 'data'),
              Input("drag-slider-zscore-entry", "value"),
              Input("drag-slider-zscore-exit", "value"),
              prevent_initial_call=True)
def run_backtest_callback(selected_pairs_dict, zscore_entry, zscore_exit):
    # Run basic backtest only for valid values selected for the z-score entry/exit
    if zscore_entry <= zscore_exit or not selected_pairs_dict:
        return no_update
    else:
        selected_pairs_dict = run_backtest(selected_pairs_dict=selected_pairs_dict,
                                           zscore_entry=zscore_entry,
                                           zscore_exit=zscore_exit)

        # Build table with relevant data for stock pairs and convert to format compatible for dash_table.DataTable
        selected_pairs_df = pd.DataFrame()
        for dict_ in selected_pairs_dict.values():
            dict_selected_keys = {key: [dict_[key]] for key in ['SymbolA', 'SymbolB', 'Corr', 'Half-Life', 'Sharpe', 'CAGR (%)']}
            df_temp = pd.DataFrame.from_dict(dict_selected_keys)
            selected_pairs_df = selected_pairs_df.append(df_temp)
        selected_pairs_df.reset_index(inplace=True, drop=True)
        selected_pairs_df = selected_pairs_df.round(3)
        selected_pairs_dash_table_data = selected_pairs_df.to_dict('records')
        selected_pairs_dash_table_columns = [{"name": i, "id": i} for i in selected_pairs_df.columns]
        
        backtest_table_descr = 'Select row/pair in table to compute full risk/return metrics (please change selection to refresh full-metrics table if you update z-score values):'
    
    return selected_pairs_dict, backtest_table_descr, selected_pairs_dash_table_data, selected_pairs_dash_table_columns


@app.callback(Output('backtest-details-table-descr', 'children'),
              Output('backtest-details-table', 'data'),
              Output('backtest-details-table', 'columns'),
              Output('cumulative-return-plot', 'figure'),
              Output('zscore-plot', 'figure'),
              State('browser-session-memory-storage', 'data'),
              Input("backtest-table", "selected_rows"),
              prevent_initial_call=True
             )
def run_backtest_details_callback(selected_pairs_dict, selected_pair):
    # Extract pair of stocks from dictionary and store values in selected_pair_dict to compute all risk/return metrics (detailed backtest) and charts
    selected_pair_dict = list(selected_pairs_dict.values())[selected_pair[0]]
    
    # Deserialize JSON pandas series stored in dict
    df_cumret = pd.read_json(selected_pair_dict['cumulative_return'], typ='series')
    df_cumret_frame = df_cumret.to_frame('cumulative_return')
    df_cumret_frame['cumulative_return'] = 100*(df_cumret_frame['cumulative_return'] - 1)
    df_cumret_frame.index.names = ['Date']

    fig1 = px.line(df_cumret_frame, x=df_cumret_frame.index, y=df_cumret_frame['cumulative_return'])
    
    # Deserialize JSON pandas series stored in dict
    df_stock_a = pd.read_json(selected_pair_dict['x'], typ='series').to_frame('StockA')
    df_stock_b = pd.read_json(selected_pair_dict['y'], typ='series').to_frame('StockB')
    df_stock_a.index.names = ['Date']
    df_stock_b.index.names = ['Date']
    df_stock = pd.merge(df_stock_a, df_stock_b, on='Date')
    df_stock_zscore = pd.read_json(selected_pair_dict['zscore'], typ='series').to_frame('zscore')
    df_stock_zscore.index.names = ['Date']
    df_stock = pd.merge(df_stock, df_stock_zscore, on='Date')
    df_stock['rolling_correlation'] = df_stock['StockA'].rolling(10).corr(df_stock['StockB'])

    fig2 = px.line(df_stock, x=df_stock.index, y=['zscore', 'rolling_correlation'])
    
    risk_return_metrics = run_backtest_details(df_cumret=df_cumret)
    risk_return_metrics_dash_table_data = risk_return_metrics.to_dict('records')
    risk_return_metrics_dash_table_columns = [{"name": i, "id": i} for i in risk_return_metrics.columns]
    
    backtest_details_table_descr = 'Full risk/return metrics:'

    return backtest_details_table_descr, risk_return_metrics_dash_table_data, risk_return_metrics_dash_table_columns, fig1, fig2

if __name__ == '__main__':
    app.run_server(debug=False)
