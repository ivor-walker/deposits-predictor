import pandas as pd

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

        self.__convert_to_categorical();
        
        self.__merge("loan", "housing");
        self.__merge("poutcome", "previous");

        self.__create_day_ids(); 
        
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
        
        # Drop original response column
        self.data.drop(columns=["y"], inplace=True);

    """
    Convert all string columns and some numerical to categorical (according to EDA)
    """
    def __convert_to_categorical(self):
        # Convert string columns to categorical
        for col in self.data.select_dtypes(include='object').columns:
            self.data[col] = self.data[col].astype('category');
        
        numerical_to_categorical = ["previous"];
        for col in numerical_to_categorical:
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
    Split data into training and test sets, based on days
    @param train_size: proportion of days to use for training
    """
    def __split_data(self,
        train_size = 0.8
    ):
        # Split predictors and response
        self.X = self.data.drop(columns=["response"]);
        self.y = self.data["response"];

        # Determine split day and train/test indices
        split_day_id = round(train_size * max(self.data["day_id"]));
        train_indices = self.data["day_id"] <= split_day_id;
        test_indices = self.data["day_id"] > split_day_id;

        # Split data
        self.insensitive_train_X = self.__select_insensitive_data(self.X[train_indices]);
        self.sensitive_train_X = self.__select_sensitive_data(self.X[train_indices]);
        self.train_y = self.y[train_indices];

        self.insensitive_test_X = self.__select_insensitive_data(self.X[test_indices]); 
        self.sensitive_test_X = self.__select_sensitive_data(self.X[test_indices]);
        self.test_y = self.y[test_indices];

    """
    Select data for a sensitive model: all categorical columns and some continuous
    @param data: data to select from
    @param continuous: list of continuous indicators to include
    """
    def __select_sensitive_data(self, data,
        continuous = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "day_id"]
    ):
        return data.select_dtypes(include='category').join(data[continuous]);

    """
    Select data for a less sensitive model: all continuous columns and all non-merge and grouped categorical columns
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

    """
    One-hot encode categorical columns 
    """
    def __encode_categorical(self):
        self.data = pd.get_dummies(self.data);
