import os
import datetime
import pandas as pd
import pymongo
from pymongo import MongoClient
import yfinance as yf

class DatabaseBackendMongoDB():
    
    def __init__(self,
                 db_params: dict,
                 stock_data_params: dict):
        # Declare class parameters storing database URI, database name and database-collection name
        self.db_uri = db_params['db_uri']
        self.db_name = db_params['db_name']
        self.db_collection_name = db_params['db_collection_name']
        
        # Declare class parameters storing list of stock indices, start-date (for initial date to download data) and local path storing CSVs containing stock symbols
        self.stock_indices = stock_data_params['stock_indices']
        self.start_date_creation = stock_data_params['start_date_creation']
        self.path_csv_symbols = stock_data_params['path_csv_symbols']

    def run_database_backend(self):
        try:
            # Connect to MongoDB
            self.client = pymongo.MongoClient(self.db_uri)
            
            # Check if stocks database and collection exists in MongoDB, if it exists it means we have already created and populated it with some initial data,
            # so this function call is to update/replenish the data, otherwise this is the inital call so we need to download the initial data and store it in the database.
            if self.db_name in self.client.list_database_names():  # check if database name for stocks exists in MongoDB
                # Get database containing stocks data
                self.db = self.client[self.db_name]
                if self.db_collection_name in self.db.list_collection_names():  # check if collection for daily data exists in stocks database
                    # Update collection in database
                    self.update_database()
            else:
                # Create database containing stocks data
                self.db = self.client[self.db_name]
                # Create and populate collection in database
                self.create_database()
        except Exception as e:
            print(f"An error occurred while connecting and creating/updating the database: {e}")
        finally:
            # Close connection to database
            self.client.close()
            
    def create_database(self):
        print('Creating and populating collection in database')
        
        # Load symbols for stock indices from CSVs stored in folder '.../symbols' and create a dictionary
        # index_symbols_dict:
        #   {'GSPC': ['AAPL', 'GOOG', ...],
        #    'NDX': [...]
        #    ...}
        index_symbols_dict = dict()
        for index in self.stock_indices:
            index_symbols = pd.read_csv(os.path.join(self.path_csv_symbols,f'{index}.csv'))
            index_symbols_dict[index] = list(index_symbols['Symbol'])

        # Create pandas dataframe with symbols as rows and stock indices as columns with true (false) boolean value if symbol is (is not) among constituents of specific stock index.
        # This is required to avoid storing duplicate data on the database (same stock for different stock indices).
        # index_symbols_bool_df:
        #        GSPC  NDX   RUT
        #   AAPL True True False
        #   ...
        values = list(set([ x for y in index_symbols_dict.values() for x in y]))
        data = {}
        for key in index_symbols_dict.keys():
            data[key] = [True if value in index_symbols_dict[key] else False for value in values]
        index_symbols_bool_df = pd.DataFrame(data, index=values).sort_index()
        index_symbols_bool_df.index.name = 'Symbol'
        # Save resulting dataframe as CSV (index_symbols_bool.csv) in folder '.../symbols' for later use when we need to update the database. This is done for efficiency reasons: avoid recreating such
        # table or reading the database to extract the list of unique symbols every time we update the database with new data.
        index_symbols_bool_df.to_csv(os.path.join(self.path_csv_symbols,'index_symbols_bool.csv'))

        # Download (from Yahoo Finance)
        data = self.download_transform_data(start_date = self.start_date_creation,
                                            index_symbols_bool_df=index_symbols_bool_df)

        # Write data to database
        self.write_data_to_database(data=data)

    def update_database(self):
        print('Updating collection in database')
        
        # Query database to get last stored date for each symbol
        # Get collection storing stocks data
        db_collection = self.db[self.db_collection_name]
        # Get last date for each symbol
        cursor = db_collection.aggregate([{'$group': {'_id': "$Ticker", 'Date': {'$last': '$Date'}}}])
        symbols_last_date_df = pd.DataFrame(list(cursor))
        # Get most recent date in the database and add 1 day to download from that date (to avoid duplicates). I assume the data for all the stocks in the collection has been downloaded up to the last
        # date for all of them.
        #TODO: Add a check to make sure that the above is true.
        start_date_update = symbols_last_date_df['Date'].max() + pd.DateOffset(1)
        
        # get index_symbols_bool_df from CSV file 'index_symbols_bool.csv' generated and saved at the time of the database creation
        index_symbols_bool_df = pd.read_csv(os.path.join(self.path_csv_symbols,'index_symbols_bool.csv')).set_index(keys='Symbol', drop=True)
        
        # Download (from Yahoo Finance)
        data_update = self.download_transform_data(start_date = start_date_update,
                                                   index_symbols_bool_df=index_symbols_bool_df)

        # Write data to database
        self.write_data_to_database(data=data_update)

    def download_transform_data(self,
                                start_date: str or datetime,  # string (YYYY-MM-DD) or datetime
                                index_symbols_bool_df: pd.DataFrame):
        print('Downloading data from Yahoo Finance')
        all_data = pd.DataFrame()  # dataframe containing data for all stocks
        symbol_data_list = []
        index_symbols = index_symbols_bool_df.columns
        num_symbols = len(index_symbols_bool_df.index)

        #for i, symbol in enumerate(index_symbols_bool_df.index[-50:],1):  # test for downloading only a subset of 50 stocks
        for i, symbol in enumerate(index_symbols_bool_df.index,1):
            try:
                print(f'Downloading data for symbol {symbol} (symbol {i} of {num_symbols}) from date {start_date}')
                # Retrieve stock data for symbol
                symbol_yf = yf.Ticker(symbol)
                symbol_data = symbol_yf.history(start=start_date,
                                                actions=False)
                # Add column with symbol
                symbol_data.insert(0, 'Symbol', symbol)
                # Add name of company/stock index
                symbol_data.insert(1, 'Name', symbol_yf.info['longName'])
                # Add column with date
                #symbol_data.insert(2, 'Date', symbol_data.index.tz_localize(None))
                symbol_data.insert(2, 'Date', symbol_data.index)
                # Add one column for each stock index and set value to true (false) if symbol is (is not) among constituents of specific stock index
                #symbol_data.loc[:,index_symbols] = list(index_symbols_bool_df.loc[symbol,:])
                cols = [f'Index_{index_symbol}' for index_symbol in index_symbols]
                symbol_data.loc[:,cols] = list(index_symbols_bool_df.loc[symbol,:])
                # Append data to dataframe containing data for all stocks
                all_data = pd.concat([all_data, symbol_data])
            except Exception as e:
                print(f"An error occurred while retrieving data for {symbol} from Yahoo Finance: {e}")
        
        return all_data
    
    def write_data_to_database(self,
                               data: pd.DataFrame):
        print('Inserting data in database')
        
        # Transform data from dataframe to list of dictionaries to make it compatible for writing into MongoDB
        data_dict = data.to_dict(orient='records')

        try:
            # Get/create collection for specific type of stocks data (daily, monthly, etc.)
            db_collection = self.db[self.db_collection_name]

            # Insert/update data into the collection
            result = db_collection.insert_many(data_dict,
                                               ordered=True)
            print(f"Inserted {len(result.inserted_ids)} records into the collection")
            print(f"last modified {datetime.datetime.utcnow()}")
        except Exception as e:
            print(f"An error occurred while inserting data into the database: {e}")

if __name__ == '__main__':
    stock_data_params = {'stock_indices': ['GSPC', 'NDX', 'RUT'],
                         #'start_date_creation': '2020-01-01',
                         'start_date_creation': '2022-01-01',
                         'path_csv_symbols': f"{os.path.abspath('')}/symbols"}
    db_params={'db_uri': 'mongodb://localhost:27017/',
               'db_name': 'stocks_db',
               'db_collection_name': 'daily'}
    db_backend_instance = DatabaseBackendMongoDB(db_params=db_params,
                                                 stock_data_params=stock_data_params)
    db_backend_instance.run_database_backend()
