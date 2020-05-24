# Pandas

# import library
import pandas as pd

# create a dataframe for a two-dimensional table
fruit_sales = pd.DataFrame({"Apples": [35, 41], "Bananas": [21, 34]}, 
                            index=["2017 Sales", "2018 Sales"])

# create a series for a list
ingredients = pd.Series(["4 cups", "1 cup","2 large","1 can"],
                        index=["Flour","Milk","Eggs","Spam"],
                        name="Dinner")

# load data from csv file
reviews = pd.read_csv("/Users/Jing/Desktop/Temp/winemag-data_first150k.csv")

# check the size of the file
reviews.shape

# the first five rows
reviews.head()

# save data to csv file
reviews.to_csv("/Users/Jing/Desktop/Temp/wine_reviews_copy.csv")

# access a column of a dataframe
reviews.country
reviews['country']

# access a specific value of a dataframe
reviews['country'][0]

# index based selection - iloc
# access the first row of a dataframe
reviews.iloc[0]

# access the first column of a dataframe
reviews.iloc[:, 0]

# access the first three entries of the second column
reviews.iloc[0:3, 1]

# label based selection - loc
# access the first three entries in column country
reviews.loc[0:2, 'country']

# access three columns
reviews.loc[:, ['country', 'points', 'price']]

# set index
reviews.set_index("title")

# select relevant rows
reviews.loc[(reviews.country=='Italy') & (reviews.points>=90)]
reviews.loc[(reviews.country=='Italy') | (reviews.points>=90)]
reviews.loc[reviews.country.isin(['Italy', 'France'])]

# filter out rows lacking a price
reviews.loc[reviews.price.notnull()]

# obtain statistic summary of a column
reviews.points.describe()

# get the mean of a column
reviews.points.mean()

# get a list of unique values in a column
reviews.country.unique()

# get a list of unique values and the frequency in a column
reviews.country.value_counts()

# return a new Series of transformed data in a column using map function
# the original data unchanged
review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean)

# find the number of rows with description column containing keyword 'tropical'
n_trop = reviews.description.map(lambda x:"tropical" in x).sum()

# return a new dataframe of transformed data using apply function
# the original data unchanged
# apply the transform function to each row with axis='columns'
# apply the transform function to each column with axis='index'
def remean_points(row):
    row.points = row.points - review_points_mean
    return row

reviews.apply(remean_points, axis='columns')

# find the wine with highest point per unit price
bargin_index = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargin_index, 'winery']

# find the minimum price for each point
reviews.groupby('points').price.min()

# find the first price for each winery
reviews.groupby('winery').apply(lambda x: x.price.iloc[0])

# find the wine with the highest point for each country and province
reviews.groupby(['country', 'province']).apply(lambda x: x.loc[x.points.idxmax()])

# apply several functions to a dataframe simultaneously using agg function
reviews.groupby(['country']).price.agg([len, min, max])

# create a series whose index is country and whose values count how many reviews each country has
reviews_by_country = reviews.groupby('country').size()

# create a series whose index is wine price and whose value is the highest point a wine
# costing that much was given in a review; sort the values by price in ascending order
best_rating_per_price = reviews.groupby('price')['points'].max().sort_index()

# create a dataframe whose index is  variety of wine and whose value is the min and max price
price_extremes = reviews.groupby('variety').price.agg([min, max])

# create a dataframe with multi-index
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
type(countries_reviewed.index)

# reset index
countries_reviewed = countries_reviewed.reset_index()

# sort the data by column len (in ascending order by default)
countries_reviewed = countries_reviewed.sort_values(by='len')

# sort the data by column len in decreasing order
countries_reviewed = countries_reviewed.sort_values(by='len', ascending=False)

# sort by index
countries_reviewed = countries_reviewed.sort_index()

# sort the data by two columns 
countries_reviewed = countries_reviewed.sort_values(by=['country', 'len'])

# get the data type of the price column
reviews.price.dtype

# get the data type of every column of a dataframe
reviews.dtypes

# change the data type of the points column from int64 to float64
reviews.points = reviews.points.astype('float64')

# entries that have missing values are given the value NaN (Not a Number)
# NaN values are always of the float64 type
# select rows where the country value is missing
reviews[pd.isnull(reviews.country)]

# count the number of rows with missing price
n_missing_prices = reviews.price.isnull().sum()

# replace NaN with "Unknown"
reviews.country.fillna("Unknown",inplace=True)

# replace "Unknown" with "NA" in country column
reviews.country.replace("Unknown", "NA", inplace=True)

# rename colulmn points to score
reviews.rename(columns={'points': 'score'}, inplace=True)

# rename index from 'rows' to 'winse'
reviews = reviews.rename_axis('winse', axis='rows')

# combine multiple dataframes with the same fields vertically using concat function
canadian_youtube = pd.read_csv("../CAvideos.csv")
british_youtube = pd.read_csv("../GBvideos.csv")
pd.concat([canadian_youtube, british_youtube])

# combine multiple dataframes with the same index horizontally using join function
left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])
left.join(right, lsuffix='_CAN', rsuffix='_UK')

