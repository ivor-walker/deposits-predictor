import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from imblearn.over_sampling import RandomOverSampler

"""
Class representing the data for the project
"""
class Data:
    """
    Constructor: read data from csv and preprocess
    @param path: path to the csv file
    """
    def __init__(self,
        path = "data/bank-additional-full.csv"
    ):
        # Data comes with semicolon separator
        self.data = pd.read_csv(path, sep=';');

    """
    Preprocessing data
    """
    def preprocess(self):
        self.__convert_response();

        self.__create_day_ids(); 

        self.__convert_to_categorical();
        
        self.__merge("loan", "housing");
        self.__merge("poutcome", "previous");
        
        self.__bin_continuous();
        self.__bin_categorical();

        self.__encode_categorical();

        self.__remove_unfair_predictors();
        
        self.__split_data();

    """
    Convert response to binary
    """
    def __convert_response(self):
        self.data["response"] = self.data["y"].replace({"no": 0, "yes": 1});
        
        # Convert response to binary
        self.data["response"] = self.data["response"] == 1;

        # Drop original response column
        self.data.drop(columns=["y"], inplace=True);

    """
    Convert all string columns and some numerical to categorical (according to EDA)
    """
    def __convert_to_categorical(self,
        numerical_to_categorical = ["previous"]
    ):
        # Get list of columns to convert
        cols_to_convert = self.data.select_dtypes(include='object').columns;
        cols_to_convert = cols_to_convert.union(numerical_to_categorical);

        # Convert columns to categorical
        for col in cols_to_convert:
            self.data[col] = self.data[col].astype('category');

    """
    Create inferred day IDs from changes in day of week
    """
    def __create_day_ids(self):
        # Create helper column to mark the first row where the day of week changes
        self.data["new_day"] = self.data["day_of_week"] != self.data["day_of_week"].shift(1);

        # Create day IDs
        self.data["day_id"] = self.data["new_day"].cumsum();

        # Remove helper column
        self.data.drop(columns=["new_day"], inplace=True);
    
    """
    Bin all continuous columns except economic indicators (according to EDA)
    """
    def __bin_continuous(self):
        # Group age into 0-29, 30-57 and 58+
        self.data["age_group"] = pd.cut(self.data["age"], bins=[0, 29, 57, 100], labels=["0-29", "30-57", "58+"]);
        
        # Group pdays into a binary variable: contacted or not
        self.data["pdays_group"] = self.data["pdays"].replace({999: "not_contacted", -1: "contacted"});

        # Group campaign into 1, 2, 3, 4, 5, 6, 7, 8 and 9+
        self.data["campaign_group"] = pd.cut(self.data["campaign"], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 100], labels=["1", "2", "3", "4", "5", "6", "7", "8", "9+"]);
    
    """
    Scale all continuous columns
    """
    def __scale_continuous(self):
        # Use MinMax scaling as continuous columns are not normally distributed
        scaler = MinMaxScaler();
        cols_to_scale = self.data.select_dtypes(include='number').columns;

        for col in cols_to_scale: 
            self.data[col] = scaler.fit_transform(self.data[[col]]);

    """
    One-hot encode categorical columns 
    """
    def __encode_categorical(self):
        self.data = pd.get_dummies(self.data);
        
    """
    Bin some categorical columns (according to EDA)
    """
    def __bin_categorical(self):
        # Group default into no and unknown_or_yes
        self.data["default_group"] = self.data["default"].replace({"unknown": "unknown_or_yes", "yes": "unknown_or_yes"});

    """
    Merge two categorical columns into one
    @param col_1: first column name to merge
    @param col_2: second column name to merge
    @param col_suffix: suffix to add to the new column name
    """
    def __merge(self, col_1, col_2,
        col_suffix = "merge"
    ):
        new_col_name = col_1 + "_" + col_2 + "_" + col_suffix;

        # Concatenate the two columns
        self.data[new_col_name] = self.data[col_1].astype(str) + "_" + self.data[col_2].astype(str);

        # Transform the new column to categorical
        self.data[new_col_name] = self.data[new_col_name].astype('category');

    """
    Remove predictors that cannot be known before the call
    @param unfair_predictors: list of predictors to remove
    """
    def __remove_unfair_predictors(self,
        unfair_predictors = ["duration"]
    ):
        self.data.drop(columns=unfair_predictors, inplace=True);

    """
    Split data into training and test sets, based on days, and validation set, based on random sampling
    @param validate_size: proportion of training set to use for validation
    """
    def __split_data(self,
        validate_size = 0.2,
    ):
        self.get_split_day(); 
        
        # Get locations of train and test data
        split_locs = self.data["day_id"] <= self.split_day;
        train_indices = split_locs[split_locs == True].index; 
        test_indices = split_locs[split_locs == False].index;
        
        # Scale continuous columns after determining split
        self.__scale_continuous();
        
        # Split predictors and response dummies
        X = self.data.drop(columns="response");
        y = self.data["response"];
                 
        # Create test sets
        test_X = X.loc[test_indices];
        self.insensitive_test_X = self.__select_insensitive_data(test_X);
        self.sensitive_test_X = self.__select_sensitive_data(test_X);
        self.test_prop = test_X.shape[0] / self.data.shape[0]; 
        self.test_y = y.loc[test_indices]; 

        # Create initial training set
        train_X = X.loc[train_indices]; 
        self.train_prop = train_X.shape[0] / self.data.shape[0];
        train_y = y.loc[train_indices];

        # Oversample the minority class in the training set
        oversampler = RandomOverSampler();
        train_X, train_y = oversampler.fit_resample(train_X, train_y);
        
        # Create final train sets from oversampled training set
        self.insensitive_train_X = self.__select_insensitive_data(train_X);
        self.sensitive_train_X = self.__select_sensitive_data(train_X);
        self.train_y = train_y
        
        # Create validation set from sample of training set
        validate_indices = train_X.sample(frac=validate_size).index; 
        validate_X = train_X.loc[validate_indices];
        self.insensitive_validate_X = self.__select_insensitive_data(validate_X); 
        self.sensitive_validate_X = self.__select_sensitive_data(validate_X);
        self.validate_y = train_y.loc[validate_indices];
        
    """
    Determine split day     
    @param train_size: proportion of days to use for training
    @param split_type: type of split to use
    """
    def get_split_day(self,
        train_size = 0.93,
        split_type = "quantile"
    ):
        # Choose split day based on a proportion of days
        if split_type == "max":
            self.split_day = self.data["day_id"].max() * train_size;
        
        # Choose split day based on a proportion of data
        elif split_type == "quantile":
            self.split_day = self.data["day_id"].quantile(train_size);

        else:
            raise ValueError("Invalid data train/test split type");

    """
    Select data for a sensitive model (e.g GAM): all encoded dummy columns of categorical variables and some continuous
    @param data: data to select from
    @param continuous: list of continuous indicators to include
    """
    def __select_sensitive_data(self, data,
        continuous = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "day_id"]
    ):
        return data[
            [col for col in data.columns
             if col in continuous
             or data[col].dtype == 'bool'
            ]
        ];
    """
    Select data for a less sensitive model (e.g RF): all continuous columns and all non-merge and grouped categorical columns
    @param suffixes: list of suffixes to exclude
    @param data: data to select from
    """
    def __select_insensitive_data(self, data,
        suffixes = ["group", "merge"]
    ):
        return data[
            [col for col in data.columns 
            if not any(
                 [suffix in col for suffix in suffixes]
            )]
        ];
