# data visualization

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
spotify_data = pd.read_csv("/Users/Jing/Desktop/Temp/spotify.csv", index_col="Date", parse_dates=True)
flight_data = pd.read_csv("/Users/Jing/Desktop/Temp/flight_delays.csv", index_col="Month")
insurance_data = pd.read_csv("/Users/Jing/Desktop/Temp/insurance.csv")
iris_data = pd.read_csv("/Users/Jing/Desktop/Temp/iris.csv", index_col="Id")
iris_set_data = pd.read_csv("/Users/Jing/Desktop/Temp/iris_setosa.csv", index_col="Id")
iris_ver_data = pd.read_csv("/Users/Jing/Desktop/Temp/iris_versicolor.csv", index_col="Id")
iris_vir_data = pd.read_csv("/Users/Jing/Desktop/Temp/iris_virginica.csv", index_col="Id")

# line chart
plt.figure(figsize=(14,6))
plt.title("Daily Global Streams of Popular Songs in 2017-2018")
sns.lineplot(data=spotify_data)
plt.xlabel("Date")

# bar chart
# create a bar chart showing the average arrival delay for Spirit Airlines
plt.figure(figsize=(10,6))
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")
sns.barplot(x=flight_data.index, y=flight_data['NK'])
plt.ylabel("Arrival delay (in minutes)")

# heatmap 
plt.figure(figsize=(10,5))
plt.title("Average Arrival Delay for Each Airline, by Month")
sns.heatmap(data=flight_data, annot=True)
plt.xlabel("Airline")

# scatter plot
# display relationship between two variables
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
sns.scatterplot(x="bmi", y="charges", data=insurance_data)

# scatter plot with regression line 
# display the strength of the linear relationship between bmi and charges
sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
sns.regplot(x="bmi", y="charges", data=insurance_data)

# scatter plot with color coding
# display relationship between three variables
# with the third variables assigned to argument hue
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
sns.scatterplot(x="bmi", y="charges", hue="smoker", data=insurance_data)

# scatter plot with color coding and regression line
sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)

# categorical scatter plot
# scatter plot between a categorical variable and a continuous variable
sns.swarmplot(x=insurance_data['smoker'], y=insurance_data['charges'])
sns.swarmplot(x="smoker", y="charges", data=insurance_data)

# histogram
sns.distplot(a=iris_data['Petal Length (cm)'], kde=False)

# density plot
# smoothed histogram
sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)

# joint density plot
sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")

# histogram for three species in one plot
sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)
sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)
sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)
plt.title("Histogram of Petal Lengths by Species")
plt.legend()

# density plot for three species in one plot
sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label="Iris-setosa", shade=True)
sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", shade=True)
sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", shade=True)
plt.title("Distribution of Petal Lengths by Species")
