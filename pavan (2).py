

# importing all of the necessary library  and components.

# It is a library written in Python that is used to perform operations on datasets..
import pandas as PD 
# The array manipulation job is done with the help of a numpy module.
import numpy as np 
# K-means is a method for grouping data points without being given specific instructions on how to do so. The data points are clustered into K groups using the algorithm, which does this by minimising the amount of variance that exists between each group..
from sklearn.cluster import KMeans  
# matplotlib import for visulaization
import matplotlib.pyplot as plots

# importing warnings.
import warnings 
warnings.filterwarnings('ignore')

"""# K Means Clustering"""

# Create the function that will be used to analyse the dataset.
def data_read(new_file):
    data = PD.read_csv(new_file, skiprows=4) # utilising pandas to read the data and skipping the first four rows of the data.
    data1 = data.drop(['Unnamed: 66', 'Indicator Code',  'Country Code'],axis=1) # removing the columns 
    data2 = data1.set_index("Country Name")  #set the index
    data2= data2.T #transform our data
    data2.reset_index(inplace=True)  
    data2.rename(columns = {'index':'Year'}, inplace = True) 
    return data1, data2 #return the dataset

# define the path of Agriculture data.
new_file = 'API_AG.LND.AGRI.ZS_DS2_en_csv_v2_4772688.csv'  
Agriculture_fdata, Transpose_data = data_read(new_file)   #different data in different variable
Agriculture_fdata.head() # show start 5 rows.

Transpose_data.head() #show rows of transpose data

# Using a function to extract information from a 20-year time period.
def Dataset(data): 
    F_data = data[['Country Name', 'Indicator Name','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010',
                                    '2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']] 
    final_dataset = F_data.dropna() # drop null values from data.
    return final_dataset

# calling the function to extract the data. 
Final_agr_Dataset = Dataset(Agriculture_fdata) 
Final_agr_Dataset.head(10) # shows starting rows from data.

Final_agr_Dataset.isnull().sum()

#creating function for preprocessing 
def pre_process(data):
  print(data.shape)
  data.dropna(axis=1,inplace=True)
  print(data.isnull().sum())
  return data

final_data=pre_process(Final_agr_Dataset)

# importing label encoder from scikit learn. 
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()# define classifier for encoder.
final_data['Country Name'] = encoder.fit_transform(final_data['Country Name']) 
final_data.head(10) # showing 5 rows from data.

X = final_data.drop(['Country Name','Indicator Name'], axis=1)
y = final_data['Country Name']  

# importing minmax scaler for normalize the data.
from sklearn.preprocessing import MinMaxScaler
Min_max_scaler = MinMaxScaler()# define classifier.
Min_max_scaled = Min_max_scaler.fit_transform(X)# fit classifier with data.

"""# Elbow Method """

# using the elbow method to find out the clusters.
from scipy.spatial.distance import cdist 
Cluster = range(10) 
Meandist = list()

for k in Cluster:
    model = KMeans(n_clusters=k+1) 
    model.fit(Min_max_scaled) 
    Meandist.append(sum(np.min(cdist(Min_max_scaled, model.cluster_centers_, 'euclidean'), axis=1)) / Min_max_scaled.shape[0]) 

# setting all the parameter and ploting the graph.

# define font size.
plots.rcParams.update({'font.size': 20})
# define figure size.
plots.figure(figsize=(10,7))
# set parameter for graph.
plots.plot(Cluster, Meandist, marker="o") 
# define xlabel.
plots.xlabel('Numbers of Clusters')
# define ylabel.
plots.ylabel('Average distance') 
# define title for graph.
plots.title('Choosing k with the Elbow Method');

# define classifier for clustering.
kmean_model = KMeans(n_clusters=4, max_iter=100, n_init=10,random_state=10)
# fit classifier with data.  
kmean_model.fit(Min_max_scaled) 
# predict model to getting the label.
predictions = kmean_model.predict(Min_max_scaled)

predictions

# define color for all clusters.
color_map = {0 : 'r', 1 : 'b', 2 : 'g',3:'m'} 
def color(x):  
    return color_map[x]  
colors = list(map(color, kmean_model.labels_))   

# plotting the graph.

# define font size.
plots.rcParams.update({'font.size': 20})
# define figure size.
plots.figure(figsize=(10,7))
# set parameter for scatter plot.
plots.scatter(x=X.iloc[:,0], y=X.iloc[:,2], c=colors)  
# define xlabel.
plots.xlabel('2000')
# define ylabel.  
plots.ylabel('2002') 
# define title for graph. 
plots.title('Scatter plot for 4 Clusters');

# Getting the Centroids and label.
centroids = kmean_model.cluster_centers_
u_labels = np.unique(predictions) 
centroids

# plotting the results.
plots.figure(figsize=(10,7))
for i in u_labels:
    plots.scatter(Min_max_scaled[predictions == i , 0] , Min_max_scaled[predictions == i , 1] , label = i)  

# define parameter for graph like color, data etc.
plots.scatter(centroids[:,0] , centroids[:,1] , s = 40, color = 'r') 
# define xlabel.
plots.xlabel('2000')
# define ylabel.
plots.ylabel('2002')
# define title for graphs.
plots.title('Scatter plot for 4 Clusters with Centroids') 
# define legend for graph.
plots.legend()  
plots.show()

# creating the lists to extract all the cluster.
first_cluster=[]
second_cluster=[] 
third_cluster=[] 
fourth_cluster=[]

# with the help of loop find out the data availabel in each cluster.
for i in range(len(predictions)):
    if predictions[i]==0:
        first_cluster.append(Agriculture_fdata.loc[i]['Country Name']) 
    elif predictions[i]==1:
         second_cluster.append(Agriculture_fdata.loc[i]['Country Name'])
    elif predictions[i]==2:
         third_cluster.append(Agriculture_fdata.loc[i]['Country Name'])
    else:
        fourth_cluster.append(Agriculture_fdata.loc[i]['Country Name'])

# showing the data present in first cluster.
print('*'*80)
First_cluster = np.array(first_cluster)
print(First_cluster)
print('*'*80)

# showing the data present in second cluster.
print('*'*80)
Second_cluster = np.array(second_cluster)
print(Second_cluster)
print('*'*80)

# showing the data present in third cluster.
print('*'*80)
Third_cluster = np.array(third_cluster)
print('*'*80)
print(Third_cluster)

Fourth_cluster = np.array(fourth_cluster)
print('*'*80)
print(Fourth_cluster)
print('*'*80)

first_cluster = First_cluster[0] 
print('Country name :', first_cluster)
arab_country = Final_agr_Dataset[Final_agr_Dataset['Country Name']==8]  
arab_country = np.array(arab_country)  
arab_country = np.delete(arab_country,1) 
arab_country

second_cluster = Second_cluster[2] 
print('*'*80)
print('Country name :', second_cluster) 
afganistan_country = Final_agr_Dataset[Final_agr_Dataset['Country Name']==2] 
afganistan_country = np.array(afganistan_country)  
afganistan_country = np.delete(afganistan_country,1) 
afganistan_country
print('*'*80)

third_cluster = Third_cluster[0] 
print('Country name :', third_cluster) 
print('*'*80)
American_country = Final_agr_Dataset[Final_agr_Dataset['Country Name']==3] 
American_country = np.array(American_country)  
American_country = np.delete(American_country,1) 
American_country
print('*'*80)

fourth_cluster = Fourth_cluster[0] 
print('Country name :', fourth_cluster) 
African_country = Final_agr_Dataset[Final_agr_Dataset['Country Name']==5] 
African_country = np.array(African_country)  
African_country = np.delete(African_country,1) 
African_country
print('*'*80)

# plotting the line graph for different clusters.
year=list(range(2000,2022))
# define figure size for graph.
plots.figure(figsize=(22,5))

plots.subplot(131)
plots.xlabel('Years')
plots.ylabel('Agriculture land') 
plots.title('Arab Country') 
plots.plot(year,afganistan_country, color='g');

plots.subplot(132)
plots.xlabel('Years')
plots.ylabel('Agriculture land') 
plots.title('African Country') 
plots.plot(year,African_country);

plots.subplot(133) 
plots.xlabel('Years') 
plots.ylabel('Agriculture land')
plots.title(' American country') 
plots.plot(year,American_country, color='r');

"""# Curve Fitting"""

# calling the function to extract the data. 
Final_agr_Dataset4 = Dataset(Agriculture_fdata) 
Final_agr_Dataset4.head(10)

#shape of data
Final_agr_Dataset4.shape

#checking null value
Final_agr_Dataset4.isnull().sum()

def columns_and_convert_into_array(data,country_name):
  x = np.array(data.columns) 
# dropping some columns.
  x = np.delete(x,0) 
  x = np.delete(x,0) 
# # convert data type as int.
  x = x.astype(np.int)

# # selecting all the data for urban population and india.
  curve_fit = data[(data['Indicator Name']=='Agricultural land (% of land area)') & (data['Country Name']==country_name)]   

# convert into array.
  y = curve_fit.to_numpy()
  # dropping some columns.
  y = np.delete(y,0) 
  y = np.delete(y,0)

# convert data type as int.
  y = y.astype(np.int) 
  return x , y

a,b=columns_and_convert_into_array(Final_agr_Dataset4,'Vietnam')

# import scipy.
import scipy
# it is python library which is used to work with arrays.
import numpy as np 
# importing curve fit from scipy.
from scipy.optimize import curve_fit
# Matplotlib is a Python library that lets you make rigid, animated, and interactive visualisations. Matplotlib makes things that are easy and things that are hard possible.
import matplotlib.pyplot as plots
from scipy import stats 

# Define the function to be fitted (linear function y = mx + c)
def line_fun(x, m, c):
    return m*x + c

def curve_fitting(x,y): 
    labels=x
 

    # Perform curve fitting
    popt, pcov = curve_fit(line_fun, a, b) 

    # Extract the fitted parameters and their standard errors
    m, c = popt
    m_err, c_err = np.sqrt(np.diag(pcov)) 

    # Calculate the lower and upper limits of the confidence range
    conf_int = 0.95  # set the confidence interval as 95%
    alpha = 1.0 - conf_int 
    m_low, m_high = scipy.stats.t.interval(alpha, len(x)-2, loc=m, scale=m_err)
    c_low, c_high = scipy.stats.t.interval(alpha, len(x)-2, loc=c, scale=c_err)

    # Plot the best-fitting function and the confidence range.
    plots.figure(figsize=(12,6)) #define figure size.
    plots.rcParams.update({'font.size': 20}) #define fontsize.
    plots.plot(x, y, 'bo', label='Data') #set data for graph.
    plots.plot(x, line_fun(x, m, c), 'g', label='Fitted function')
    plots.fill_between(x, line_fun(x, m_low, c_low), line_fun(x, m_high, c_high), color='gray', alpha=0.5, label='Confidence range') # set all the parameter.
    plots.title('Curve Fitting') # define title for graph.
    plots.xlabel('Years') # define xlabel.
    plots.xticks(x,labels,rotation='vertical')
    plots.ylabel('Population') # define ylabel. 
    plots.legend() # set legend in graph.
    plots.show()

curve_fitting(a,b)

