# Feature engineering
# 1. label encoding
# 2. count encoding
# 3. target encoding
# 4. catboost encoding
# 5. create interaction 
# 6. transform numerical features
# 7. univariate feature selection
# 8. L1 regularization

# import library
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn import metrics
import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# load data
ks = pd.read_csv('/Users/Jing/Desktop/Temp/ks-projects-201801.csv',
                 parse_dates=['deadline', 'launched'])

# prepare target column
# (1) drop projects that are "live"
# (2) count "successful" states as outcome = 1
# (3) count every other state as outcome = 0
ks['state'].value_counts()
ks = ks.query('state!="live"')
ks = ks.assign(outcome=(ks['state']=='successful').astype(int))

# create new features (of type int64) based on launched column (of type datetime64)
ks = ks.assign(hour=ks.launched.dt.hour,
               day=ks.launched.dt.day,
               month=ks.launched.dt.month,
               year=ks.launched.dt.year)

# Label encoding
# assign an integer to each value of a categorical feature
# apply label encoder to three categorical variables: category, country, currency
# LightGBM models work with label encoded features and thus no need for one-hot encoding
cat_features = ['category', 'currency', 'country']
encoder = LabelEncoder()
encoded = ks[cat_features].apply(encoder.fit_transform)

# prepare data for modeling
# split data into train/validation/test
# check the proportion of successful outcomes in each dataset
data = ks[['goal', 'hour', 'day', 'month', 'year', 'outcome']].join(encoded)

def get_data_splits(dataframe, valid_fraction=0.1):
    valid_size = int(len(data)*valid_fraction)
    train = data[:-2*valid_size]
    valid = data[-2*valid_size:-valid_size]
    test = data[-valid_size:]
    return train, valid, test

train, valid, test = get_data_splits(data)
for each in [train, valid, test]:
    print("Fraction of successful outcomes: " + str(each.outcome.mean()))

# train a LightGBM model
# (1) drop the target
# (2) define the training data set and validation data set
# (3) set the hyperparameters
feature_cols = train.columns.drop('outcome')
dtrain = lgb.Dataset(train[feature_cols], label=train['outcome'])
dvalid = lgb.Dataset(valid[feature_cols], label=valid['outcome'])
param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
bst = lgb.train(param, 
                dtrain, 
                num_round, 
                valid_sets=[dvalid], 
                early_stopping_rounds=10, 
                verbose_eval=False)

# make predictions
y_pred= bst.predict(test[feature_cols])

# evaluate the model
score = metrics.roc_auc_score(test['outcome'], y_pred)

# Count encoding
# assign each categorical value the number of times it appears in the dataset
count_encoder = ce.CountEncoder(cols=cat_features)
count_encoder.fit(train[cat_features])
train = train.join(count_encoder.transform(train[cat_features]).add_suffix('_count'))
valid = valid.join(count_encoder.transform(valid[cat_features]).add_suffix('_count'))

# Target encoding
# assign each categorical value the average value of the target for that value of the feature
# learn the target encodings from the training dataset only and apply it to the other datasets
target_encoder = ce.TargetEncoder(cols=cat_features)
target_encoder.fit(train[cat_features], train['outcome'])
train = train.join(target_encoder.transform(train[cat_features]).add_suffix('_target'))
valid = valid.join(target_encoder.transform(valid[cat_features]).add_suffix('_target'))

# CatBoost encoding
# similar to target encoding
# The difference is that with CatBoost, the target probability is calculated only
# from the rows before it.
catboost_encoder = ce.CatBoostEncoder(cols=cat_features)
catboost_encoder.fit(train[cat_features], train['outcome'])
train = train.join(catboost_encoder.transform(train[cat_features]).add_suffix('_cb'))
valid = valid.join(catboost_encoder.transform(valid[cat_features]).add_suffix('_cb'))

# create a new feature by combining categorical variables
interactions = ks['category'] + "_" + ks['country']

# count the number of projects in the preceeding week
# (1) create a series with the launched column as the index and the index as the values
# (2) create a rolling window that contains all the data in the previous 7 days using the .rolling method on a series
# (3) the window contains the current record, so subtract 1 to exclude the current one
# (4) set index and adjust it to match the original dataset
launched = pd.Series(ks.index, index=ks.launched, name="count_7_days").sort_index()
count_7_days = launched.rolling('7d').count() - 1
count_7_days.index = launched.values
count_7_days = count_7_days.reindex(ks.index)

# calculate the time since the last project in the same category
# (1) sort the data in increasing order of launched
# (2) group the data by category and apply the .transform method; this method takes a function and apply it to each group
# (3) replacing missing value with the median in that category
# (4) reorder the resultant dataset to match the index of the original dataset
def time_since_last_project(dataframe):
    return dataframe.diff().dt.total_seconds()/3600

df = ks[['category', 'launched']].sort_values('launched')
timedeltas = df.groupby('category').transform(time_since_last_project)
timedeltas = timedeltas.fillna(timedeltas.median()).reindex(ks.index)


# check normality of a distrubtion using histogram
# it is necessary for linear model or neural network but not for tree-based models 
plt.hist(ks.goal, range=(0,100000), bins=50);

# apply square root transformation
plt.hist(np.sqrt(ks.goal), range=(0,400), bins=50);

# apply log transformation
plt.hist(np.log(ks.goal), range=(0,25), bins=50);

# select the five best features based on F-value
# feature selection should use training data only to avoid leakage
# (1) drop the target column from the dataset
# (2) split the dataset into training, validation and testing
# (3) create a feature selector
# (4) apply the feature selector to the training dataset
# (5) get a dataframe with the same index and columns as the training data but the unselected columns are filled with zeros
# (6) find selected columns by choosing features with nonzero variance
feature_cols = data.columns.drop('outcome')
train, valid, test = get_data_splits(data)
selector = SelectKBest(f_classif, k=6)
X_new = selector.fit_transform(train[feature_cols], train['outcome'])
selected_features = pd.DataFrame(selector.inverse_transform(X_new), 
                                 index=train.index, 
                                 columns=feature_cols)
selected_columns = selected_features.columns[selected_features.var()!=0]

# L1 regularization
# feature selection using L1 regularization should use training data only
# (1) split the data into training, validation and testing
# (2) drop the target column
# (3) fit a logistic regressio model to the training dataset (the smaller the parameter C the more penalty)
# (4) select the nonzero coefficients using .SelectFromModel method
# (5) select features based on the nonzero coefficients
# (6) get a dataframe with the same index and columns as the training data but the unselected columns are filled with zeros
# (7) find selected columns by choosing features with nonzero variance
train, valid, test = get_data_splits(data)
X, y = train[train.columns.drop("outcome")], train['outcome']
logistic = LogisticRegression(C=0.00001, penalty="l1", random_state=7).fit(X, y)
model = SelectFromModel(logistic, prefit=True)
X_new = model.transform(X)
selected_features = pd.DataFrame(model.inverse_transform(X_new),
                                 index=X.index,
                                 columns=X.columns)
selected_columns = selected_features.columns[selected_features.var()!=0]
