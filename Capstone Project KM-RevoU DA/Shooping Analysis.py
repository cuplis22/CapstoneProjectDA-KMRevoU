#!/usr/bin/env python
# coding: utf-8

# # **Install Module**

# In[1]:


pip install kmodes


# In[2]:


pip install yellowbrick


# In[3]:


pip install mlxtend


# In[4]:


pip install missingno


# # **Import library**

# In[1]:


import pandas as pd
import os
import numpy as np
import seaborn as sns
import warnings
from scipy import stats
from matplotlib import pylab as plt
from statsmodels.graphics.gofplots import qqplot
from IPython.core.interactiveshell import InteractiveShell
import plotly.express as px
from operator import attrgetter
import matplotlib.colors as mcolors

from plotnine import *
import plotnine

from kmodes.kprototypes import KPrototypes

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)
InteractiveShell.ast_node_interactivity = 'all'

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
set_seed()

pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 30)

print("setup-complete!")


# # **Read-in Data**

# In[2]:


# Accomodate raw path to variables
raw_customer, raw_orders = "customers.csv", "orders.csv"
raw_products, raw_sales = "products.csv", "sales.csv"

# Read-in data
customer, order = pd.read_csv(raw_customer), pd.read_csv(raw_orders)
product, sales_data = pd.read_csv(raw_products), pd.read_csv(raw_sales)


# # **Cleaning Dataset**

# In[3]:


product.rename(columns={"product_ID" : "product_id"}, inplace=True)
product


# **Dataset Information**

# In[95]:


# check dataset information
data = [{'df':customer,'name':'customer dataset'},
        {'df':sales_data,'name':'sales_data dataset'},
        {'df':product,'name':'product dataset'},
        {'df':order,'name':'order dataset'},]

for item in data:
  print(f"Checking dataset information on {item['name']}")
  print(item['df'].info())
  print('')


# **check data duplication**

# In[96]:


data = [{'df':customer,'name':'customer dataset'},
        {'df':sales_data,'name':'sales_data dataset'},
        {'df':product,'name':'product dataset'},
        {'df':order,'name':'order dataset'},]

for item in data:
  print(f"number of duplications {item['name']}")
  print(item['df'].isnull().sum())
  print('')


# **Check Null Values**

# In[97]:


data = [{'df':customer,'name':'customer dataset'},
        {'df':sales_data,'name':'sales_data dataset'},
        {'df':product,'name':'product dataset'},
        {'df':order,'name':'order dataset'},]

for item in data:
  print(f"Checking Null value on {item['name']}")
  print(item['df'].isnull().any())
  print('')


# **Change Data Type**

# In[4]:


def change_datatype(df,list_changes):
  for item in list_changes:
    if item['type'] == 'datetime':
      df[item['column']] = pd.to_datetime(df[item['column']])
    elif item['type'] == 'int':
      df[item['column']] = df[item['column']].astype('int')
    elif item['type'] == 'int64':
      df[item['column']] = df[item['column']].astype(np.int64)
    elif item['type'] == 'float':
      df[item['column']] = df[item['column']].astype(float)

  return df


# In[5]:


## change datatype
data = [{'df':order,
         'list_changes': [
             {'column':'order_date','type':'datetime'}
             , {'column':'delivery_date','type':'datetime'}]
         }]
for el in data:
  el['df'] = change_datatype(el['df'],el['list_changes'])


# **Remove Extra Whitespaces**

# In[38]:


def remove_extra_whitespaces(df):
  for col in df.columns:
    if df[col].dtypes == 'object' :
      df[col] = df[col].str.strip()

  return df


# In[39]:


# remove extra whitespace on customers dataset
list_df = [customer,order,sales_data,product]
for df in list_df:
  df = remove_extra_whitespaces(df)


# **Checking Unique Value**

# In[102]:


def check_nunique(df,name):
  print(f'Checking number of unique value in {name}')
  for col in df.columns:
    nuniq = len(df[col].unique())
    print(f'- {col} : {nuniq} nunique')
  print('')


# In[103]:


data = [{'df':customer,'name':'customer dataset'},
        {'df':sales_data,'name':'sales_data dataset'},
        {'df':product,'name':'product dataset'},
        {'df':order,'name':'order dataset'},]

for item in data:
  check_nunique(item['df'],item['name'])


# **Checking Outliers**

# In[104]:


def check_outliers(df_col,col_name):
  print(f"checking outliers on {col_name}")

  # Find Q1, Q3, IQR
  Q1 = df_col.quantile(0.25)
  Q3 = df_col.quantile(0.75)
  IQR = Q3 - Q1

  # Find Bottom Fence and Upper Fence
  boxplot_min = Q1 - 1.5 * IQR
  boxplot_max = Q3 + 1.5 * IQR

  #Show the calculation
  print('Q1:',Q1)
  print('Q3:',Q3)
  print('IQR:',IQR)
  print('Min (Lower inner fence):',boxplot_min)
  print('Max (Upper inner fence):',boxplot_max)

  # Filter value that <Bottom fence and >Upper fence
  filter_min = df_col < boxplot_min
  filter_max = df_col > boxplot_max

  # drop outlier using loc. So it will show only value exclude (~) bottom fence and Upper fence
  df_outlier = customer.loc[(filter_min | filter_max)]

  # Check data information
  df_outlier.count()

  print("Result:")
  if df_outlier.shape[0] == 0:
    print("no outliers found")
  else:
    print(f"there are {df_outlier.shape[0]} outliers found")
  print('')


# In[105]:


data = [{'df':customer['age'],'name':'customers dataset on field age'},
        {'df':sales_data['total_price'],'name':'sales dataset on field total_price'},
        {'df':sales_data['quantity'],'name':'sales dataset on field quantity'}]

for item in data:
  check_outliers(item['df'],item['name'])


# Checking Outliers using bloxplot

# In[106]:


# List Check
list_check = [{'df':customer,'column':'age'},
        {'df':sales_data,'column':'total_price'},
        {'df':sales_data,'column':'quantity'}]

# adjust chart position and chart size
fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(20,10))

# for 0 (i) , Income (el)
for i,el in enumerate(list_check):
    a = el['df'].boxplot(el['column'], ax=axes.flatten()[i],grid=False)

# Show Chart
plt.show()


# **Customer**

# In[107]:


customer.describe(include="all")


# In[108]:


customer.groupby(by="gender").agg({
    "customer_id": "nunique",
    "age": ["max", "min", "mean", "std"]
})


# In[109]:


customer.groupby(by="city").customer_id.nunique().sort_values(ascending=False)


# In[110]:


customer.groupby(by="state").customer_id.nunique().sort_values(ascending=False)


# **Order**

# In[111]:


delivery_time = order["delivery_date"] - order["order_date"]
delivery_time = delivery_time.apply(lambda x: x.total_seconds())
order["delivery_time"] = round(delivery_time/86400)


# In[112]:


order.describe(include="all")


# **Order and Customer**

# In[113]:


customer_id_in_order =  order.customer_id.tolist()
customer["status"] = customer["customer_id"].apply(lambda x: "Active" if x in customer_id_in_order else "Non Active")
customer.sample(5)


# In[114]:


customer.groupby(by="status").customer_id.count()


# In[115]:


order_customer = pd.merge(
    left=order,
    right=customer,
    how="left",
    left_on="customer_id",
    right_on="customer_id"
)
order_customer.head()


# **Number of order by city**

# In[116]:


order_customer.groupby(by="city").order_id.nunique().sort_values(ascending=False).reset_index().head(10)


# **Number of order based on state**

# In[117]:


order_customer.groupby(by="state").order_id.nunique().sort_values(ascending=False)


# **Number of order based on gender**

# In[118]:


order_customer.groupby(by="gender").order_id.nunique().sort_values(ascending=False)


# **Number of order based on age group**

# In[119]:


order_customer["age_group"] = order_customer.age.apply(lambda x: "Youth" if x <= 24 else ("Seniors" if x > 64 else "Adults"))
order_customer.groupby(by="age_group").order_id.nunique().sort_values(ascending=False)


# **Product and Sales**

# In[120]:


product.describe(include="all")


# In[121]:


sales_data.describe(include="all")


# In[122]:


product.sort_values(by="price", ascending=False)


# In[123]:


product.groupby(by="product_type").agg({
    "product_id": "nunique",
    "quantity": "sum",
    "price":  ["min", "max"]
})


# In[124]:


product.groupby(by="product_name").agg({
    "product_id": "nunique",
    "quantity": "sum",
    "price": ["min", "max"]
})


# In[125]:


sales_data_product = pd.merge(
    left=sales_data,
    right=product,
    how="left",
    left_on="product_id",
    right_on="product_id"
)
sales_data_product.head()


# In[126]:


sales_data_product.groupby(by="product_type").agg({
    "sales_id": "nunique",
    "quantity_x": "sum",
    "total_price": "sum"
})


# In[127]:


sales_data_product.groupby(by="product_name").agg({
    "sales_id": "nunique",
    "quantity_x": "sum",
    "total_price": "sum"
}).sort_values(by="total_price", ascending=False)


# **ALL DATA**

# In[128]:


all_df = pd.merge(
    left=sales_data_product,
    right=order_customer,
    how="left",
    left_on="order_id",
    right_on="order_id"
)
all_df.head()


# In[129]:


all_df.groupby(by=["state", "product_type"]).agg({
    "quantity_x": "sum",
    "total_price": "sum"
})


# In[130]:


all_df.groupby(by=["gender", "product_type"]).agg({
    "quantity_x": "sum",
    "total_price": "sum"
})


# In[131]:


all_df.groupby(by=["age_group", "product_type"]).agg({
    "quantity_x": "sum",
    "total_price": "sum"
})


# # **Merging Dataset**

# In[6]:


cust_order = pd.merge(left=customer, right=order, 
                      left_index=True, right_index=True) # merging
cop = pd.merge(left=cust_order, right=product, 
                    left_index=True, right_index=True) # merging


# In[7]:


df = customer.merge(order,how='inner',left_on='customer_id',right_on='customer_id')
df = df.merge(sales_data,how='inner',left_on='order_id',right_on='order_id')
df = df.merge(product,how='inner',left_on='product_id',right_on='product_id')


# In[8]:


df['order_day'] = df['order_date'].dt.dayofweek
df['order_week'] = df['order_date'].dt.strftime('%U').astype(np.int64)
df['order_month_str'] = df['order_date'].dt.strftime('%Y-%m')
df['order_month'] = df['order_date'].dt.to_period('M')


# In[9]:


cop['order_day'] = df['order_date'].dt.dayofweek
cop['order_week'] = df['order_date'].dt.strftime('%U').astype(np.int64)
cop['order_month_str'] = df['order_date'].dt.strftime('%Y-%m')
cop['order_month'] = df['order_date'].dt.to_period('M')


# In[136]:


df.to_excel('Clean dataset shopping cart1.xlsx')


# In[137]:


cop.to_excel('Clean dataset shopping cart2.xlsx')


# **Check Data**

# In[138]:


df.head(3)


# In[139]:


df.rename(columns={"quantity_x" : "quantity","quantity_y" : "stock"}, inplace=True)
df


# In[140]:


cop


# In[141]:


df.info()


# In[142]:


cop.info()


# In[143]:


df.isnull().sum().sum()


# In[144]:


cop.isnull().sum().sum()


# In[145]:


df[df.duplicated()]


# In[146]:


cop[cop.duplicated()]


# **Sorting data based on age**

# In[147]:


df.sort_values(by=["age"],
axis=0, ascending=True,
inplace=True)
df["age"]
df.head()


# In[148]:


cop.sort_values(by=["age"],
axis=0, ascending=True,
inplace=True)
cop["age"]
cop.head()


# # **EDA**

# In[10]:


monthly_sales = df.groupby('order_month_str',as_index=False)['total_price'].sum()

fig = px.line(monthly_sales
              , x='order_month_str'
              , y=['total_price']
              , title='Monthly Sales Trend')

fig.update_layout(
    yaxis_title="Total Sales"
    , xaxis_title="Month")


# # **Clustering**

# **Using K-Means**

# In[38]:


from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler


# In[39]:


def elbow_method(df_clus_main):
  # For each n_clusters between 1 and 11, we calculate the distortion value
  print('Finding best n_cluster using elbow method')
  distortions = []
  K = range(2,11)
  for n_clusters in K:
    print('n_clusters =',n_clusters,end=' ~ ')
    kmeanModel = KMeans(n_clusters, random_state = 42)
    kmeanModel.fit(df_clus_main)
    distortions.append(kmeanModel.inertia_)
    print('kmeanModel.inertia_ =', kmeanModel.inertia_)
    
  plt.figure(figsize=(16,8))
  plt.figure()
  plt.plot(K, distortions, 'b*-')
  plt.xlabel('k')
  plt.ylabel('Inertia')
  plt.title('The Elbow Method showing the optimal k')
  plt.show()


# In[40]:


def silhoutte_method(df_clus_main):
  # Silhouette score plot
  K = range(2,11)
  max_K = max(K)
  fig, ax = plt.subplots(int(np.ceil(max_K/2)), 2, figsize = (15,30))

  for n_clusters in K:
    kmeanModel = KMeans(n_clusters)

    q, mod = divmod(n_clusters,2)
    sil = SilhouetteVisualizer(kmeanModel, is_fitted = False, ax = ax[q-1][mod])
    sil.fit(df_clus_main)
    sil.finalize()
    print(f"For k={n_clusters}, the average silhouette score is {sil.silhouette_score_}")


# In[41]:


# df_clus = df.groupby(['customer_id','age','quantity','price_per_unit'],as_index=False)['total_price','quantity'].agg([''])
df_clus1 = df.groupby(['customer_id'],as_index=False)['order_id'].nunique()
df_clus1 = df_clus1.rename(columns={'order_id':'order'})
df_clus2 = df.groupby(['customer_id'],as_index=False)['quantity','total_price'].sum()
df_clus = df_clus1.merge(df_clus2,how='inner',left_on='customer_id',right_on='customer_id')
df_clus = df_clus.set_index('customer_id')
df_clus


# In[42]:


scaler = MinMaxScaler()

# df_clus2
df_scaled = scaler.fit_transform(df_clus.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=['order','quantity','total_price'])

print("Scaled Dataset Using MinMaxScaler")
df_scaled
# df_clus2


# In[43]:


def kmeans_clus(df_clus,df_ori,n):
    # Let's get the label for k = 3

    # Initialize KMeans for 3 clusters
    cluster_model = KMeans(n_clusters = n, random_state = 42)

    # Fit the data into model
    cluster_model.fit(df_clus)

    # 
    df_clus_r = df_ori.copy()
    df_clus_r['cluster'] = cluster_model.labels_

    df_clus_r.groupby('cluster',as_index=False).agg(['mean','median']).T.to_excel(f'kmeans-{n}-cluster-pivot.xlsx')

    df_clus_r.to_excel(f'kmeans-{n}-cluster.xlsx')
    return df_clus_r


# In[44]:


# kmeans_clus(df_scaled,df_clus,3)
# kmeans_clus(df_scaled,df_clus,4)
# kmeans_clus(df_scaled,df_clus,5)
kmeans_clus(df_scaled,df_clus,6)


# In[45]:


cluster_list = []
for dirname, _, filenames in os.walk('http://localhost:8890/tree/Documents/belajar%20python/Capstone%20Project%20KM-RevoU%20DA#notebooks'):
    for filename in filenames:
        cluster_list.append(os.path.join(dirname, filename))

df_cluster = pd.read_excel('kmeans-6-cluster.xlsx')


# In[46]:


df_cluster.groupby('cluster',as_index=False)['order','quantity','total_price'].mean().sort_values(by='total_price',ascending=True)


# In[47]:


def clusname(x):
    if x == 0 :
        return 'Anak Juragan'
    elif x == 1 :
        return 'Warga'
    elif x == 2 :
        return 'Sultan'
    elif x == 3 :
        return 'Anak Sultan'
    elif x == 4 :
        return 'Juragan'
    elif x == 5 :
        return 'Bos'
    elif x == 10 :
        return 'Anak Baru'
    else : 
        return ''


# In[48]:


df_cluster['clus_name'] = df_cluster['cluster'].apply(clusname)
df_all = df.merge(df_cluster[['customer_id','cluster','clus_name']], how='inner', left_on='customer_id', right_on='customer_id')
df_all


# In[49]:


clus1 = df_all.groupby(['customer_id','cluster','clus_name'],as_index=False)['order_id'].nunique()
clus1.rename(columns={'order_id':'order'},inplace=True)
clus2 = df_all.groupby(['customer_id','age'],as_index=False)['quantity','total_price'].sum()
# clus2 = clus2.reset_index()
clus = clus1.merge(clus2, how='inner',left_on='customer_id', right_on='customer_id')
clus1 = clus.groupby('clus_name',as_index=False)['age','order','quantity','total_price'].mean()
clus2 = clus.groupby('clus_name',as_index=False)['customer_id'].nunique()
clusr = clus1.merge(clus2, how='inner',left_on='clus_name', right_on='clus_name')
clusr.rename(columns={'customer_id':'customer'},inplace=True)
clusr.sort_values(by='quantity',ascending=True)


# In[50]:


clus.groupby('clus_name',as_index=False)['age','quantity','total_price'].agg(pd.Series.median).sort_values(by='quantity',ascending=True)


# In[51]:


clus.groupby('clus_name',as_index=False)['order','quantity','total_price'].agg(pd.Series.sum).sort_values(by='quantity',ascending=True)


# In[52]:


df_all.to_excel('Clean dataset shopping cart with cluster.xlsx')


# In[53]:


df_all.groupby(['customer_id','cluster','clus_name'])['customer_id'].nunique().to_excel('customer_cluster.xlsx')


# **Using KPrototype**

# In[54]:


# Format scientific notation from Pandas
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[55]:


# Creating copies of our datasets
df_clus_kmodes = df_clus.merge(customer[['customer_id','age','gender','state']],how='inner',left_on='customer_id',right_on='customer_id')

# drop if all row of columns is null
df_clus_kmodes.dropna(axis=1, how='all', inplace=True)

df_clus_kmodest = pd.get_dummies(df_clus_kmodes, dummy_na=True)
df_clus_kmodest.set_index('customer_id',inplace=True)
df_clus_kmodest


# In[56]:


import plotly.io as pio
pio.renderers.default = "notebook"

sns.histplot(data=df_clus_kmodest, x="quantity")


# In[57]:


from sklearn.preprocessing import StandardScaler

X = df_clus_kmodest.copy()
scaled_X = StandardScaler().fit_transform(X[['order','total_price', 'quantity','age']])
X[['order','total_price', 'quantity','age']] = scaled_X


# In[58]:


X.info()


# In[59]:


from kmodes.kprototypes import KPrototypes
#dataframe to an array

smart_array = X.values

#index of categorical columns
categorical_index = list(range(4,22))
categorical_index


# In[60]:


# Function for plotting elbow curve
def plot_elbow_curve(start, end, data,categorical_index):
    no_of_clusters = list(range(start, end+1))
    cost_values = []
    
    for k in no_of_clusters:
        test_model = KPrototypes(n_clusters=k, init='Huang', random_state=42)
        test_model.fit_predict(data, categorical=categorical_index)
        cost_values.append(test_model.cost_)
        
    sns.set_theme(style="whitegrid", palette="bright", font_scale=1.2)
    
    plt.figure(figsize=(15, 7))
    ax = sns.lineplot(x=no_of_clusters, y=cost_values, marker="o", dashes=False)
    ax.set_title('Elbow curve', fontsize=18)
    ax.set_xlabel('No of clusters', fontsize=14)
    ax.set_ylabel('Cost', fontsize=14)
    ax.set(xlim=(start-0.1, end+0.1))
    plt.plot()


# In[61]:


# Plotting elbow curve for k=2 to k=10
# plot_elbow_curve(2,10,smart_array,categorical_index)


# In[62]:


def kproto_clustering(df_clus_kmodes,smart_array,categorical_index, n):
    model_3 = KPrototypes(n_clusters=n, init='Huang', random_state=42, n_jobs=-1)
    model_3.fit_predict(smart_array, categorical=categorical_index)
    print(model_3.cost_)
    #new column for cluster labels associated with each subject
    df_clus_kmodes_r = df_clus_kmodes.copy()
    df_clus_kmodes_r['cluster'] = model_3.labels_

    df_clus_kmodes_fin = df_clus_kmodes_r.merge(df_clus_kmodest.iloc[:, 4:],how='inner', left_on='customer_id', right_on='customer_id')
    df_clus_kmodes_fin.drop(columns=['gender_nan','state_nan'],axis=1,inplace=True)

    df_clus_kmodes_fin.groupby('cluster',as_index=False).agg(['mean','median']).T.to_excel(f'kproto-{n}-cluster-pivot.xlsx')
    df_clus_kmodes_fin.to_excel(f'kproto-{n}-cluster.xlsx')
    return df_clus_kmodes_fin


# In[63]:


kproto_clustering(df_clus_kmodes,smart_array,categorical_index,6)
# kproto_clustering(df_clus_kmodes,smart_array,categorical_index,5)
# kproto_clustering(df_clus_kmodes,smart_array,categorical_index,4)
# kproto_clustering(df_clus_kmodes,smart_array,categorical_index,3)


# In[64]:


Z = df.copy()
N = customer[~(customer['customer_id'].isin(Z['customer_id']))]
N


# In[65]:


N['age'].agg([pd.Series.mean,pd.Series.median])


# In[66]:


N['gender'].unique()


# # **Cohort Analysis**

# In[67]:


n_orders = df.groupby(['customer_id'])['order_id'].nunique()
mult_orders_perc = np.sum(n_orders > 1) / df['customer_id'].nunique()
print(f'{100 * mult_orders_perc:.2f}% of customers ordered more than once.')


# In[68]:


ax = sns.distplot(n_orders, kde=False, hist=True)
ax.set(title='Distribution of number of orders per customer',
       xlabel='# of orders', 
       ylabel='# of customers');


# In[69]:


dfca = df.groupby(['customer_id', 'order_id', 'order_date', 'order_month'],as_index=False)['total_price','quantity'].sum()
dfca


# In[70]:


df['cohort'] = df.groupby('customer_id')['order_date'].transform('min').dt.to_period('M')
df_cohort = df.groupby(['cohort', 'order_month']) \
              .agg(n_customers=('customer_id', 'nunique')) \
              .reset_index(drop=False)

df_cohort['period_number'] = (df_cohort.order_month - df_cohort.cohort).apply(attrgetter('n'))


# In[71]:


cohort_pivot = df_cohort.pivot_table(index = 'cohort',
                                     columns = 'period_number',
                                     values = 'n_customers')


# In[72]:


cohort_size = cohort_pivot.iloc[:,0]
retention_matrix = cohort_pivot.divide(cohort_size, axis = 0)
cohort_size_df = pd.DataFrame(cohort_size).rename(columns={0: 'cohort_size'})


# In[73]:


with sns.axes_style("white"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True, gridspec_kw={'width_ratios': [1, 11]})
    
    # retention matrix
    sns.heatmap(retention_matrix, 
                mask=retention_matrix.isnull(), 
                annot=True, 
                fmt='.0%', 
                cmap='RdYlGn', 
                ax=ax[1])
    ax[1].set_title('Monthly Cohorts: User Retention', fontsize=16)
    ax[1].set(xlabel='# of periods',
              ylabel='')

    # cohort size
    white_cmap = mcolors.ListedColormap(['white'])
    sns.heatmap(cohort_size_df, 
                annot=True, 
                cbar=False, 
                fmt='g', 
                cmap=white_cmap, 
                ax=ax[0])
    print("sans")
    fig.tight_layout()


# In[74]:


cohort_size_df2 = cohort_size_df.reset_index()
retention_matrix2 = retention_matrix.reset_index()
cust_ret = cohort_size_df2.merge(retention_matrix2,how='inner',left_on='cohort',right_on='cohort')
cust_ret.to_excel('customer_retention.xlsx')


# # **Market Basket Analysis**

# In[75]:


# df
my_basket = []
order_id_list = []

list_orderid = df.order_id.unique().tolist()
list_orderid.sort()
list_orderid
for order_id in list_orderid:
  new_df = df[df['order_id']==order_id]['product_name']
  new_basket = []
  for item in new_df:
    new_basket.append(item)

  my_basket.append(new_basket)


# **Manual Checking**

# In[76]:


def frequency_items (x,y):
    fx_=sum([x in i for i in my_basket])
    fy_=sum([y in i for i in my_basket])
    
    fxy_=sum([all(z in i for z in [x,y]) for i in my_basket])
    
    support=fxy_/len(my_basket)
    confidence = support/(fx_/len(my_basket))
    lift =confidence /(fy_/len(my_basket))
    if confidence ==1:
        conviction = 0
    else:
        conviction=(1-(fy_/len(my_basket)))/(1-confidence)
    
    print("Support = {}".format(round(support,2)))
    print("Confidence = {}".format(round(confidence,2)))
    print("Lift= {}".format(round(lift,2)))
    print("Conviction={}".format(round(conviction,2)))


# In[77]:


pd.pivot_table(df,index='product_name',values='order_id', aggfunc='count')


# In[78]:


frequency_items('Bomber','Cardigan')


# **Using Apriori Algorithm**

# In[79]:


import mlxtend.frequent_patterns 
import mlxtend.preprocessing

encode_=mlxtend.preprocessing.TransactionEncoder()
encode_arr=encode_.fit_transform(my_basket)

print(encode_arr)


# In[80]:


encode_df=pd.DataFrame(encode_arr, columns=encode_.columns_)
encode_df


# In[81]:


len(df['order_id'].unique())


# **Calculating Support**

# In[82]:


md=mlxtend.frequent_patterns.apriori(encode_df)
md_minsup=mlxtend.frequent_patterns.apriori(encode_df, min_support=0.01, use_colnames=True)
md_minsup.head(20)


# **Creating rules (Metric: Confidence) Antecedents ⇒ Consequents**

# In[83]:


rules = mlxtend.frequent_patterns.association_rules(md_minsup, metric="confidence",min_threshold=0.125000,support_only=False)

rules.head(20)


# **Creating rules (Metric: Lift) Antecedents ⇒ Consequents**

# In[84]:


rules2=mlxtend.frequent_patterns.association_rules(md_minsup, metric="lift",min_threshold=0.06,support_only=False)

rules2.head(20)


# **Export Output**

# In[85]:


associate_items = rules2
associate_items.to_excel('associate_items2.xlsx')


# **Scatter plots help us to evaluate general tendencies of rules between antecedents and consequents**

# In[86]:


# Generate scatterplot using support and confidence
plt.figure(figsize=(10,6))
sns.scatterplot(x = "support", y = "confidence", 
                size = "lift", data = rules)
plt.margins(0.01,0.01)
plt.show()


# In[87]:


# Generate scatterplot using support and confidence
plt.figure(figsize=(10,6))
sns.scatterplot(x = "support", y = "lift", 
                size = "confidence", data = rules2)
plt.margins(0.01,0.01)
plt.show()


# # **Forecasting**

# In[88]:


df_fc = df_all.groupby(['order_date'],as_index=False)['order_id'].nunique()
df_fc2 = df_all.groupby(['order_date'],as_index=False)['quantity','total_price'].sum()
df_fc = df_fc.merge(df_fc2,how='inner',left_on='order_date',right_on='order_date')
df_fc.rename(columns={'order_id':'order'},inplace=True)
# df_fc.info()
df_fc['dayname'] = df_fc['order_date'].dt.day_name()
df_fc2 = pd.get_dummies(df_fc,drop_first=True)
df_fc2.set_index('order_date',inplace=True)
df_fc2


# In[89]:


import statsmodels.api as sm
sm.graphics.tsa.plot_pacf(df_fc['order'], lags = 30)
plt.show()


# In[90]:


import statsmodels.api as sm
sm.graphics.tsa.plot_pacf(df_fc['quantity'], lags = 30)
plt.show()


# In[91]:


import statsmodels.api as sm
sm.graphics.tsa.plot_pacf(df_fc['total_price'], lags = 30)
plt.show()


# In[92]:


df_fc2['lag_order'] = df_fc2['order'].shift(1)
df_fc2['lag2_order'] = df_fc2['order'].shift(2)
df_fc2['lag3_order'] = df_fc2['order'].shift(3)
df_fc2['lag4_order'] = df_fc2['order'].shift(4)
df_fc2['lag5_order'] = df_fc2['order'].shift(5)


# In[93]:


# Data for forecasting
df_forecast = df_fc2.copy()

# Remove NA
df_forecast = df_forecast[~df_forecast.isna().any(axis=1)]

df_forecast.columns


# In[94]:


# Define predictor and 
X = df_forecast[['quantity', 'total_price',
        'dayname_Monday', 'dayname_Saturday', 'dayname_Sunday',
        'dayname_Thursday', 'dayname_Tuesday', 'dayname_Wednesday',
        'dayname_Monday', 'dayname_Saturday', 'dayname_Sunday',
        'dayname_Thursday', 'dayname_Tuesday', 'dayname_Wednesday',
        'lag_order', 'lag2_order', 'lag3_order',
        'lag4_order', 'lag5_order']]

y = df_forecast[['order']]

# To test the accuracy of our forecast, let's only train the model until June 2012 and see forecast from it forward
X_train = X[X.index < '2021-10-01']
y_train = y[y.index < '2021-10-01']



X_test = X[X.index >= '2021-10-01']
y_test = y[y.index >= '2021-10-01']


# In[95]:


from sklearn.linear_model import LinearRegression # model lin. reg

model = LinearRegression()
model.fit(X_train,y_train)


# In[96]:


model.score(X_train,y_train)


# In[97]:


# Create prediction for whole sample
df_prediction = y.copy()
df_prediction['Predicted'] = model.predict(X)

# Combine result to original data
df_fc_pred = df_fc2.merge(df_prediction, how='left', left_index = True, right_index = True)
df_fc_pred


# In[98]:


df_fc_pred[~df_fc_pred['Predicted'].isna()]['order_x']


# In[99]:


df_fc_pred['order_month'] = df_fc_pred.index.month
df_fig = df_fc_pred.groupby('order_month')['order_x','Predicted'].sum()
# df_fig
fig = df_fig[['order_x','Predicted']].plot(kind = 'line', figsize = (15,8))
fig.axvline(x = '2021-10-01',color = 'black', dashes = (3,2))
fig


# In[100]:


#how to check RMSE score(this usually used in kaggle)

from sklearn.metrics import mean_squared_error
import math


#we filter check the testing data (not used in training)
test = df_fc_pred[df_fc_pred.index >= '2021-10-01']
y_actual = test['order_x']
y_predicted = test['Predicted']
 
MSE = mean_squared_error(y_actual, y_predicted)
 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)
#your prediction off around 50k from actual data


# In[101]:


test


# **Forecasting using Arima**

# Note: this section is unfinished, will be continue

# In[102]:


from statsmodels.tsa.arima.model import ARIMA


# In[103]:


ARIMAmodel = ARIMA(y, order = (2, 1, 0))
ARIMAmodel = ARIMAmodel.fit()

# y_pred = ARIMAmodel.get_forecast(len(test.index))
y_pred = ARIMAmodel.get_forecast(90)
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df
# y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
# y_pred_df.index = test.index
# y_pred_out = y_pred_df["Predictions"] 
# plt.plot(y_actual, color='Blue', label = 'Order')
# plt.plot(y_pred_out, color='Orange', label = 'ARIMA Predictions')
# plt.legend()


# # **Data Preparation 1**

# In[104]:


import missingno as msno
msno.bar(df)


# In[105]:


df.duplicated().sum()


# **What is the overall distribution of customer ages in the dataset?**

# In[106]:


# Count of the each age value
df['age'].value_counts()


# In[107]:


# mean or average value of age 
df['age'].mean()


# In[108]:


# unique function shows the unique values of any columns
df['gender'].unique()


# In[109]:


# we are cutting the age into some category and storing in the different column
df['age_category'] = pd.cut(df['age'], bins= [0,15, 18 , 30 , 50 , 70] , labels= ['child' , 'teen' , 'Young Adults' ,'Middle-Aged Adults'
                                                                                             , 'old'] )


# In[112]:


# we use plotly library to use plots
fig = px.histogram(df , y = 'age' , x = 'age_category')
fig.show()


# **How does the average purchase amount vary across different product categories?**

# In[113]:


df['product_type'].unique()


# In[114]:


df.groupby('product_type')['quantity'].mean()


# **Which gender has the highest number of purchases?**

# In[115]:


# this is the seaborn plot
sns.barplot(df , x = 'gender' , y = 'quantity')


# In[116]:


df_group = df.groupby('gender')['quantity'].sum().reset_index()


# In[117]:


fig = px.bar(df_group , x = 'gender' , y = 'quantity')
fig.show()


# In[118]:


plt.figure(figsize=(20,25))
sns.countplot(data=df, x='product_type',hue='gender')
plt.grid(axis='y',linestyle='--', alpha=0.5)
plt.xlabel('product_type')
plt.ylabel('Frequency')
plt.title('Distribution Category by Gender')


# **What are the most commonly purchased items in each category?**

# In[119]:


# we are seeking Item purchased based on Category
df.groupby('product_type')['product_name'].value_counts()


# In[120]:


fig = px.histogram(df , x = 'product_name' , color = 'product_type')
fig.show()


# In[121]:


df['product_type'].value_counts().plot(kind='pie',autopct='%.1f%%',shadow=True, startangle=90)
plt.title('Distribution of Category')
plt.ylabel(' ')


# **How does the frequency of purchases vary across different age groups?**

# In[122]:


# we are cutting the age into some category and storing in the different column
# df['age_category'] = pd.cut(df['age'], bins= [0,15, 18 , 30 , 50 , 70] , labels= ['child' , 'teen' , 'Young Adults' ,'Middle-Aged Adults'
#                                                                                              , 'old'] )

df[['age' , 'age_category']]


# In[123]:


df['age_category'].unique()


# In[124]:


plt.figure(figsize=(20,10))
sns.countplot(data=df, x='product_type',hue='age_category')
plt.grid(axis='y',linestyle='--', alpha=0.5)
plt.xlabel('product_type')
plt.ylabel('Frequency')
plt.title('Distribution Product Type by age category')


# In[125]:


plt.figure(figsize=(40,20))
sns.countplot(data=df, x='product_name',hue='age_category')
plt.grid(axis='y',linestyle='--', alpha=0.5)
plt.xlabel('product_name')
plt.ylabel('Frequency')
plt.title('Distribution Products by age category')


# **Are there any correlations between the size of the product and the purchase amount?**

# In[126]:


df_group = df.groupby('size')['stock'].sum().reset_index()


# In[127]:


fig  = px.bar(df_group , x = 'size' , y ='stock'  )
fig.show()


# In[128]:


df['stock'].mean()


# **Are there any specific colors that are more popular among customers?**

# In[129]:


df['colour'].value_counts().nlargest(5)


# In[130]:


plt.figure(figsize=(20,10))
sns.countplot(data=df, x='colour',hue='product_type')
plt.grid(axis='y',linestyle='--', alpha=0.5)
plt.xlabel('colour')
plt.ylabel('Frequency')
plt.title('Distribution Product Type by colour')


# **Which age has the highest number of purchases?**

# In[131]:


df_group = df.groupby('product_type')['age'].mean().reset_index()


# In[132]:


fig = px.bar(df_group ,y = 'age' , x= 'product_type')
fig.show()


# In[133]:


df_group = df.groupby('product_name')['age'].mean().reset_index()


# In[134]:


fig = px.bar(df_group ,y = 'age' , x= 'product_name')
fig.show()


# **Which age has the largest number of purchases based on size?**

# In[135]:


plt.figure(figsize=(12,6))
sns.barplot(data=df, x='size', y='age')


# **Best Selling Products**

# In[136]:


plt.figure(figsize=(12,6))
df['product_name'].value_counts().head(10).plot(kind='bar', title='Top 10 Products Most Buy', color='green')
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()


# # **Data Preparation 2**

# In[137]:


msno.bar(cop)


# In[138]:


cop["sales"] = cop["price"] * cop["quantity"] # let's make a sales data
cop.head(2)


# In[139]:


# let's get the year data in order date column
cop['year_order'] = cop['order_date'].dt.year

# let's get the month data in order date column
cop['month_order'] = cop['order_date'].dt.month

# Let's get the day data in order date column
cop["day_order"] = cop["order_date"].dt.day


# In[140]:


cop.head(2)


# In[141]:


# let's get the year data in delivery date column
cop['year_delivery'] = cop['delivery_date'].dt.year

# let's get the month data in delivery date column
cop['month_delivery'] = cop['delivery_date'].dt.month

# Let's get the day data in delivery date column
cop["day_delivery"] = cop["delivery_date"].dt.day


# In[142]:


cop.head(2)


# # **Data Analysis**

# In[143]:


sns.set_style("whitegrid") # set the seaborn style
# let's make a correlation matrix for `cop_data`
fig = plt.figure(dpi=100, figsize=(24, 18)) # figure the size
sns.heatmap(cop.corr(), annot=True, cmap="Reds") # create a heatmap
plt.title("COP (Customer, Order, Product) Data Correlation", weight="bold", fontsize=30, fontname="fantasy", pad=75) # title
plt.xticks(weight="bold", fontsize=15) # x-ticks
plt.yticks(weight="bold", fontsize=15); # y-ticks


# In[144]:


# Let's see the correlation from `cop_data`
(cop.corr()["sales"] # transform it into data corr
         .sort_values(ascending=False) # sort values
         .to_frame() # change it into data frame
         .T) # transpose


# In[145]:


# let's make a correlation matrix for `sales`
fig = plt.figure(figsize=(24, 16)) # figure the size
sns.heatmap(sales_data.corr(), annot=True, cmap="Greens") # construct the heatmap
plt.title("Sales Data Correlation", weight="bold", fontsize=30, fontname="fantasy", pad=75) # title
plt.xticks(weight="bold", fontsize=15) # x-ticks
plt.yticks(weight="bold", fontsize=15); # y-ticks


# In[146]:


# Let's see the correlation
(sales_data.corr()["total_price"] # transform it into data corr
      .sort_values(ascending=False) # sort the values
      .to_frame() # change it into data frame
      .T) # transpose 


# **Statistical Measure**

# In[147]:


cop.describe(include=[np.number]) # Let's have a look to the discrete and continuous data first


# In[148]:


df.describe(include=[np.number]) # Let's have a look to the discrete and continuous data first


# In[149]:


cop.describe(exclude=[np.number]) # Let's have a look to categorical data


# In[150]:


df.describe(exclude=[np.number]) # Let's have a look to categorical data


# In[151]:


try:
    sales_data.describe(exclude=[np.number]) # Let's see on sales data
except ValueError as error:
    print(error)


# **Univariate Analysis**

# SALES

# In[152]:


# checking and visualizing the type of distribution of a feature column
def univariate_analysis(data, color, title1, title2):
    
    """
    Showing visualization of univariate
    analysis with displot and qqplot
    visualization from seaborn and statsmodel
    library.
    
    Parameters
    ----------
    data : DataFrame, array, or list of arrays, optional
        Dataset for plotting. If ``x`` and ``y`` are absent, this is
        interpreted as wide-form. Otherwise it is expected to be long-form. 
    title1: The title of the visualization, title1 for displot visualization
        And title2 for quantile plot from statsmodel.
    title2: The title of the visualization, title1 for displot visualization
        And title2 for quantile plot from statsmodel.
        
    Returns
    -------
    fig : matplotlib figure
        Returns the Figure object with the plot drawn onto it.
    """
    
    fig, (ax1, ax2) = plt.subplots( # subplots
        ncols=2, # num of cols
        nrows=1, # num of rows
        figsize=(20, 6) # set the width and high
    )

    sns.distplot( # create a distplot visualization
        data, # data
        ax=ax1, # axes 1
        kde=True, # kde
        color=color # color
    )
    
    ax1.set_title( # set the title 1
        title1, 
        weight="bold", # weight
        fontname="monospace", # font-name
        fontsize=25, # font-size
        pad=30 # padding
    )
    
    qqplot( # qqplot (quantile plot)
        data, # data
        ax=ax2, # axes 2
        line='s' # line 
    )
    
    ax2.set_title( # set the title 2
        title2, 
        weight="bold", # weight
        fontname="monospace", # font-name
        fontsize=25, # font-size
        pad=30 # padding
    )
    
    return fig # returning the figure


# In[153]:


# Sales Data
univariate_analysis( # call the function
    data=cop['sales'], # put the data
    color='deepskyblue', # pick the color
    title1='COP Data - Sales Data Distribution', # title1
    title2='Quantile Plot'); # title2


# In[154]:


round(cop.sales.mean())
round(np.std(cop.sales, ddof=1))


# AGE

# In[155]:


# Age Data
univariate_analysis( # call the function
    data=cop['age'], # put the data
    color='orange', # pick the color
    title1='COP Data - Age Data Distribution', # title1
    title2='Quantile Plot'); # title2


# In[156]:


round(cop.age.mean())
round(np.std(cop.age, ddof=1))


# PRICE

# In[157]:


# Price Data
univariate_analysis( # call the function
    data=cop['price'], # put the data
    color='darkviolet', # pick the color
    title1='COP Data - Price Data Distribution', # title1
    title2='Quantile Plot'); # title2


# In[158]:


round(cop.price.mean())
round(np.std(cop.price, ddof=1))


# QUANTITY

# In[159]:


# Quantity Data
univariate_analysis( # call the function
    data=cop['quantity'], # put the data
    color='slategrey', # pick the color
    title1='COP Data - Quantity Data Distribution', # title1
    title2='Quantile Plot'); # title2


# In[160]:


round(cop.quantity.mean())
round(np.std(cop.quantity, ddof=1))


# **checking skewness value**

# In[161]:


# checking skewness value
# if value lies between -0.5 to 0.5  then it is normal otherwise skewed
skew_value = cop.skew().sort_values(ascending=False).to_frame().head()
skew_value


# Total Price Data

# In[162]:


# Total Price Data
univariate_analysis( # call the function
    data=sales_data['price_per_unit'], # put the data
    color='chartreuse', # pick the color
    title1='SALES Data - Price Per-Unit Data Distribution', # title1
    title2='Quantile Plot'); # title2


# In[163]:


round(sales_data.price_per_unit.mean())
round(np.std(sales_data.price_per_unit, ddof=1))


# PRICE

# In[164]:


# Price Data
univariate_analysis( # call the function
    data=sales_data['total_price'], # put the data
    color='darkslategray', # pick the color
    title1='SALES Data - Total Price Data Distribution', # title1
    title2='Quantile Plot'); # title2


# In[165]:


round(sales_data.total_price.mean())
round(np.std(sales_data.total_price, ddof=1))


# Price per-unit

# In[166]:


# Price per-unit Data
univariate_analysis( # call the function
    data=sales_data['quantity'], # put the data
    color='aqua', # pick the color
    title1='SALES Data - Quantity Data Distribution', # title1
    title2='Quantile Plot'); # title2


# In[167]:


round(sales_data.quantity.mean())
round(np.std(sales_data.quantity, ddof=1))


# **checking skewness value**

# In[168]:


# checking skewness value
# if value lies between -0.5 to 0.5  then it is normal otherwise skewed
skew_value = sales_data.skew().sort_values(ascending=False).to_frame().head()
skew_value


# # **Which products were sold the most in the last month?**

# In[169]:


(cop.groupby(["month_order", "product_type", "product_name"])["sales"] # groupping
        .sum() # sum
        .astype("int") # change the type 
        .sort_values(ascending=False) # sort the values
        .to_frame() # change it into data frame
        .head(17) # look the first 17 rows
        .T) # Transpose


# In[170]:


cop.groupby(["month_order"]).sum().astype("int")


# In[171]:


# set-up
color_map = ["#5FCDF5" for _ in range(20)]
color_map[0] = "#E3866F"
color_map[2] = "#E3866F"
sns.set_palette(sns.color_palette(color_map))
span_range = [-0.5, 2.5]

# group the Month cols
sum_month_order = cop.groupby(["month_order"]).sum().astype("int")

# let's plot it
fig, ax = plt.subplots(
    1, 1, 
    figsize=(24, 10), 
    facecolor="mintcream")

# makes bar plot 
sns.barplot(
    x=sum_month_order.index,
    y=sum_month_order["sales"], 
    data=sum_month_order,
    zorder=2,
    palette=color_map,
    saturation=.9,
    alpha=.7,
    ax=ax) 

# title 
ax.set_title(
    "Months with the highest number of sales", 
    fontname="fantasy", 
    weight="bold", 
    fontsize=35, 
    pad=120)

plt.suptitle(
    "How have sales and revenue changed over the past few quarters?", 
    fontname="fantasy",
    weight="bold",
    fontsize=20)

# labels
ax.set_xlabel( 
    "Months", 
    weight="bold", 
    color="black",
    family="fantasy",
    fontsize=25, 
    loc="center",
    labelpad=25)
ax.set_ylabel(
    "Sales in Dollar Australia ($)", 
    weight="bold", 
    family="fantasy",
    fontsize=20,
    labelpad=25)

# ticklabels
ax.set_xticklabels( 
    labels=["Jan", "Feb", "Mar", "Apr", "May", 
            "Jun", "Jul", "Aug", "Sep", "Oct"],
    weight="bold", 
    family="fantasy",
    fontsize=15)
ax.set_yticklabels( 
    labels=["$0.0", "$200.000", "$400.000", 
            "$600.000", "$800.000", "$1.000.000"],
    weight="bold",
    family="fantasy",
    fontsize=15)

# y-limit
ax.set_ylim(0, 1000000)

# face-color
ax.set_facecolor("mintcream")

# text 
ax.text(0, 765000-66000, " $727.160 ", va="center", ha="center", 
        family="fantasy", weight="bold", fontsize=15)

ax.text(1, 648500-66000, " $611.133 ", va="center", ha="center", 
        family="fantasy", weight="bold", fontsize=15)

ax.text(2, 797200-66000, " $759.620 ", va="center", ha="center", 
        family="fantasy", weight="bold", fontsize=15)

ax.text(3, 693000-66000, " $653.023 ", va="center", ha="center", 
        family="fantasy", weight="bold", fontsize=15)

ax.text(4, 589000, " $552.995 ", va="center", ha="center", 
        family="fantasy", weight="bold", fontsize=15)

ax.text(5, 696000-66000, " $658.699 ", va="center", ha="center", 
        family="fantasy", weight="bold", fontsize=15)

ax.text(6, 743000-66000, " $658.699 ", va="center", ha="center", 
        family="fantasy", weight="bold", fontsize=15)

ax.text(7, 719900, " $688.716 ", va="center", ha="center", 
        family="fantasy", weight="bold", fontsize=15)

ax.text(8, 688000-66000, " $651.023 ", va="center", ha="center", 
        family="fantasy", weight="bold", fontsize=15)

ax.text(9, 562000-66000, " $524.515 ", va="center", ha="center", 
        family="fantasy", weight="bold", fontsize=15)

# annotate
ax.annotate("(μ mean)", xy=(9, sum_month_order["sales"].mean()), 
             xytext=(9.5, sum_month_order["sales"].mean() + 9000),
             size=13, ha='right', va="center", color="black", 
             weight="semibold", family="fantasy",
             bbox=dict(boxstyle="round", pad=.5, color="#FFE699"),
             arrowprops=dict(arrowstyle="->", color="#FFE699"));

# axv-span
ax.axvspan(
    span_range[0], 
    span_range[1], 
    color="peachpuff", 
    alpha=0.3)

# axh-line
ax.axhline(
    y=sum_month_order["sales"].mean(), 
    color="black", 
    ls="--", 
    lw=1.5);


# In[172]:


### set-up
labels = ["0-30yo", "31-40yo", "41-50yo", "51-60yo", "61-70yo", "71-80yo"]
bins= [30, 31, 41, 51, 61, 71, 80]
cop["age_group"] = pd.cut(cop["age"], bins=bins, labels=labels, right=False)
sum_total_age = [660,  5652,  7382,  9209, 10890, 10883]
sum_age_group = pd.DataFrame({"age": labels, "sum_total_age": sum_total_age})
color_map = ["#5FCDF5" for _ in range(6)]
color_map[5] = "#E3866F"
color_map[4] = "#E3866F"
sns.set_palette(sns.color_palette(color_map))
span_range = [3.5, 5.5]
span_range2 = [[-0.15, 0.14], 
               [0.85, 1.14], 
               [1.85, 2.14], 
               [2.85, 3.14], 
               [3.85, 4.14], 
               [4.85, 5.14]]

# subplots
fig, ax = plt.subplots(
    1, 1, 
    figsize=(24, 12), 
    facecolor=("mintcream"))

# countplot
ax.scatter(
    sum_age_group["age"], 
    sum_age_group["sum_total_age"], 
    color=color_map,
    s=3500,
    zorder=1)

# title
ax.set_title(
    "Understanding Customer demographics and their preferences", 
    fontname="fantasy",
    weight="bold",
    fontsize=35,
    pad=75)

# axv-span
ax.axvspan(span_range[0], span_range[1], color="peachpuff", alpha=0.2)
ax.axvspan(span_range2[0][0], span_range2[0][1], color="gray", alpha=0.2)
ax.axvspan(span_range2[1][0], span_range2[1][1], color="gray", alpha=0.2)
ax.axvspan(span_range2[2][0], span_range2[2][1], color="gray", alpha=0.2)
ax.axvspan(span_range2[3][0], span_range2[3][1], color="gray", alpha=0.2)
ax.axvspan(span_range2[4][0], span_range2[4][1], color="gray", alpha=0.2)
ax.axvspan(span_range2[5][0], span_range2[5][1], color="gray", alpha=0.2)

# labels
ax.set_xlabel(
    "Age Group", 
    weight="bold", 
    family="fantasy", 
    fontsize=25,
    labelpad=25)
ax.set_ylabel(
    "Quantity", 
    weight="bold", 
    family="fantasy", 
    fontsize=25,
    labelpad=25)

# ticklabels
ax.set_xticklabels(
    labels=labels, 
    weight="bold", 
    fontsize=15,
    family="fantasy")
ax.set_yticklabels(
    labels=list(np.arange(0, 16000, 2000)), 
    weight="bold", 
    fontsize=15,
    family="fantasy")

# text
ax.text(0, 660, " 1.47% ", va="center", ha="center", 
         fontsize=15, c="white", family="fantasy", weight="semibold")

ax.text(1, 5652, " 12.65% ", va="center", ha="center", 
         fontsize=15, c="white", family="fantasy", weight="semibold")

ax.text(2, 7382, " 16.52% ", va="center", ha="center", 
         fontsize=15, c="white", family="fantasy", weight="semibold")

ax.text(3, 9209, " 20.61% ", va="center", ha="center", 
         fontsize=15, c="white", family="fantasy", weight="semibold")

ax.text(4, 10890, " 24.37% ", va="center", ha="center", 
         fontsize=15, c="white", family="fantasy", weight="semibold")

ax.text(5, 10883, " 24.35% ", va="center", ha="center", 
         fontsize=15, c="white", family="fantasy", weight="semibold")

# adjust ticks
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")

# y-limit
ax.set_ylim(0, 15000)

ax.grid(False, which="major", axis="x")

# face-color
ax.set_facecolor("mintcream");


# In[173]:


# set-up
color_map = ["#5FCDF5" for _ in range(20)]
color_map[0] = "#E3866F"
sns.set_palette(sns.color_palette(color_map))
span_range = [-0.5, .5]

# Subplots
fig, (ax1, ax2) = plt.subplots(
    ncols=2, 
    nrows=1, 
    facecolor=("mintcream"),
    figsize=(24, 12) 
)

# barplot
sns.barplot( 
    x=cop["gender"].value_counts().values, 
    y=cop["gender"].value_counts().index,
    saturation=.9,
    alpha=.7,
    ax=ax1)

# prepare data for Pie Plots
cop_pie = {"gender": ["Male", "Non-binary", "Polygender", "Genderqueer", "Genderfluid", "Bigender", "Female", "Agender"], 
           "count": [143, 131, 128, 127, 122, 120, 115, 114]} 

# convert into dataframe
cop_pie = pd.DataFrame(cop_pie)

# make a pie plot
cop_pie.plot(  
    kind="pie", 
    y="count",
    labels=cop_pie["gender"], 
    autopct='%1.1f%%',
    startangle=90, 
    legend=False, 
    wedgeprops=dict(width=0.12),
    pctdistance=0.75,
    fontsize=20,
    textprops=dict(color="black", weight="bold", family="fantasy"), 
    ax=ax2)

# labels
ax1.set_xlabel("Count", 
               weight="bold", 
               family="fantasy", 
               fontsize=25, 
               labelpad=25)
ax1.set_ylabel("Genders", 
               weight="bold", 
               family="fantasy", 
               fontsize=25, 
               labelpad=25)

# ticks
ax1.set_xticklabels(labels=list(np.arange(0, 225, 25)), 
                    weight="bold", 
                    fontsize=15,
                    family="fantasy")
ax1.set_yticklabels(labels=cop["gender"].value_counts().index, 
                    weight="bold",
                    fontsize=15,
                    family="fantasy")

# text
ax1.text(153-20, 0, " 143 ", va="center", ha="center", 
         fontsize=20, family="fantasy", weight="semibold")

ax1.text(141-20, 1, " 131 ", va="center", ha="center", 
         fontsize=20, family="fantasy", weight="semibold")

ax1.text(138-20, 2, " 128 ", va="center", ha="center", 
         fontsize=20, family="fantasy", weight="semibold")

ax1.text(137-20, 3, " 127 ", va="center", ha="center", 
         fontsize=20, family="fantasy", weight="semibold")

ax1.text(132-20, 4, " 122 ", va="center", ha="center", 
         fontsize=20, family="fantasy", weight="semibold")

ax1.text(130-20, 5, " 120 ", va="center", ha="center", 
         fontsize=20, family="fantasy", weight="semibold")

ax1.text(125-20, 6, " 115 ", va="center", ha="center", 
         fontsize=20, family="fantasy", weight="semibold")

ax1.text(124-20, 7, " 114 ", va="center", ha="center", 
         fontsize=20, family="fantasy", weight="semibold")

# x-limit
ax1.set_xlim(0, 200)

# axh-span
ax1.axhspan(span_range[0], 
            span_range[1], 
            color="peachpuff", 
            alpha=0.3)

# face-color
ax1.set_facecolor("mintcream")

# ax2 y-label
ax2.set_ylabel(None);


# In[174]:


# set-up
color_map = ["#5FCDF5" for _ in range(8)]
color_map[4] = color_map[3] = "#E3866F"
sns.set_palette(sns.color_palette(color_map))
state = ['Australian Capital Territory', 'New South Wales', 'Northern Territory', 'Queensland',
         'South Australia', 'Tasmania', 'Victoria', 'Western Australia']
values = [799094, 844465, 842360, 862965, 907400, 674646, 782525, 819482]
state_sales = pd.DataFrame({"state": state, "values": values})
span_range = [2.8, 4.2]

# let's plot it
fig, ax = plt.subplots(
    1, 1,
    figsize=(24, 10),
    facecolor="mintcream")

# makes bar plot 
sns.lineplot(
    x=state_sales["state"],
    y=state_sales["values"],
    data=state_sales,
    marker="d",
    markersize=15,
    markerfacecolor="#E3866F",
    lw=4,
    color="#5FCDF5",
    ax=ax)

# title
ax.set_title( # title
    "State with the highest number of Sales", 
    fontname="fantasy",
    weight="bold",
    fontsize=35,
    pad=120)
plt.suptitle(
    "Which state has the highest sales?", 
    fontname="fantasy",
    weight="bold",
    fontsize=20)

# labels
ax.set_xlabel(
    "State", 
    family="fantasy",
    weight="bold",
    fontsize=25,
    loc="center",
    labelpad=25)
ax.set_ylabel( # y-label
    "Sales in Dollar Australia ($)", 
    family="fantasy",
    weight="bold",
    fontsize=20, 
    loc="center",
    labelpad=25)

# ticklabels
ax.set_xticklabels(
    labels=state_sales["state"],
    weight="bold", 
    fontsize=14, 
    family="fantasy",
    rotation=0)
ax.set_yticklabels(
    labels=["$0.0",     "$200.000",
            "$400.000", "$600.000",
            "$800.000", "$1.000.000",
            "$1.200.000", "$1.400.000"],
    weight="bold",
    family="fantasy",
    fontsize=15)

# annotate
ax.annotate("$799.094", xy=(0, 799094), xytext=(0.5, 666666),
             size=13, ha='right', va="center", color="black", 
             weight="semibold", family="fantasy",
             bbox=dict(boxstyle="round", pad=.5, color="#FFE699"),
             arrowprops=dict(arrowstyle="->", color="#FFE699"));

ax.annotate("$844.465", xy=(1, 844465), xytext=(1.5, 644465),
             size=13, ha='right', va="center", color="black",
             weight="semibold", family="fantasy",
             bbox=dict(boxstyle="round", pad=.5, color="#FFE699"),
             arrowprops=dict(arrowstyle="->", color="#FFE699"));

ax.annotate("$842.360", xy=(2, 842360), xytext=(1.5, 1042360),
             size=13, ha='right', va="center", color="black",
             weight="semibold", family="fantasy",
             bbox=dict(boxstyle="round", pad=.5, color="#FFE699"),
             arrowprops=dict(arrowstyle="->", color="#FFE699"));

ax.annotate("$862.965", xy=(3, 862965), xytext=(3.5, 662965),
             size=13, ha='right', va="center", color="black",
             weight="semibold", family="fantasy",
             bbox=dict(boxstyle="round", pad=.5, color="#FFE699"),
             arrowprops=dict(arrowstyle="->", color="#FFE699"));

ax.annotate("$862.965", xy=(3, 862965), xytext=(3.5, 662965),
             size=13, ha='right', va="center", color="black",
             weight="semibold", family="fantasy",
             bbox=dict(boxstyle="round", pad=.5, color="#FFE699"),
             arrowprops=dict(arrowstyle="->", color="#FFE699"));

ax.annotate("$907.400", xy=(4, 907400), xytext=(3.5, 1107400),
             size=13, ha='right', va="center", color="black",
             weight="semibold", family="fantasy",
             bbox=dict(boxstyle="round", pad=.5, color="#FFE699"),
             arrowprops=dict(arrowstyle="->", color="#FFE699"));

ax.annotate("$674.646", xy=(5, 674646), xytext=(5.5, 604646),
             size=13, ha='right', va="center", color="black",
             weight="semibold", family="fantasy",
             bbox=dict(boxstyle="round", pad=.5, color="#FFE699"),
             arrowprops=dict(arrowstyle="->", color="#FFE699"));

ax.annotate("$782.525", xy=(6, 782525), xytext=(6.5, 982525),
             size=13, ha='right', va="center", color="black",
             weight="semibold", family="fantasy",
             bbox=dict(boxstyle="round", pad=.5, color="#FFE699"),
             arrowprops=dict(arrowstyle="->", color="#FFE699"));

ax.annotate("$819.482", xy=(7, 819482), xytext=(6.8, 619482),
             size=13, ha='right', va="center", color="black",
             weight="semibold", family="fantasy",
             bbox=dict(boxstyle="round", pad=.5, color="#FFE699"),
             arrowprops=dict(arrowstyle="->", color="#FFE699"));

# axv-span
ax.axvspan(span_range[0], 
           span_range[1], 
           color="peachpuff", 
           alpha=0.3)

# facecolor
ax.set_facecolor("mintcream")

# y-limits
ax.set_ylim(0, 1400000);


# In[175]:


# set-up
color_map = ["#5FCDF5" for _ in range(6)]
color_map[0] = "#E3866F"
sns.set_palette(sns.color_palette(color_map))
city_list = ['East Aidan', 'East Sophia', 'West Sebastianfort', 
             'East Max', 'Port Hunter', 'South Georgia']
sales = [20247, 19628, 18240, 18127, 16128, 15945]
top_6_city = pd.DataFrame({"city_list": city_list, "sales": sales})
span_range = [-0.2, 0.2]

# let's plot it
fig, ax = plt.subplots(
    1, 1,
    figsize=(24, 10),
    facecolor="mintcream")

# line-plot 
sns.lineplot(
    x=top_6_city["city_list"],
    y=top_6_city["sales"],
    data=state_sales,
    marker="^",
    markersize=15,
    markerfacecolor="#E3866F",
    lw=1,
    color="#5FCDF5",
    ax=ax)

# title
ax.set_title(
    "Cities with the highest number of Sales", 
    fontname="fantasy",
    weight="bold",
    fontsize=35,
    pad=120)
plt.suptitle(
    "Which city has the highest sales?", 
    fontname="fantasy",
    weight="bold",
    fontsize=20)

# labels
ax.set_xlabel(
    "Cities", 
    weight="bold",
    family="fantasy",
    fontsize=25,
    loc="center",
    labelpad=25)
ax.set_ylabel( # y-label
    "Sales in Dollar Australia ($)", 
    weight="bold",
    family="fantasy",
    fontsize=20, 
    loc="center",
    labelpad=25)

# ticklabels
ax.set_xticklabels(
    labels=top_6_city["city_list"],
    weight="bold", 
    fontsize=15, 
    family="fantasy",
    rotation=0)
ax.set_yticklabels(
    labels=["$0.0",     "$5.000",
            "$10.000", "$15.000",
            "$20.000", "$25.000",
            "$30.000", "$35.000",
            "$40.000"],
    weight="bold",
    family="fantasy",
    fontsize=15)

# texts
ax.text(0, 22000, " $20.247 ", va="center", ha="center",
         fontsize=15, family="fantasy", weight="semibold")

ax.text(1, 21000, " $19.628 ", va="center", ha="center", rotation=-5,
         fontsize=15, family="fantasy", weight="semibold")

ax.text(2, 20000, " $18.240 ", va="center", ha="center",
         fontsize=15, family="fantasy", weight="semibold")

ax.text(3, 19500, " $18.127 ", va="center", ha="center",
         fontsize=15, family="fantasy", weight="semibold")

ax.text(4, 17500, " $16.128 ", va="center", ha="center", rotation=-5,
         fontsize=15, family="fantasy", weight="semibold")

ax.text(5, 17000, " $15.945 ", va="center", ha="center",
         fontsize=15, family="fantasy", weight="semibold")

# axv-span
ax.axvspan(span_range[0], 
           span_range[1], 
           color="peachpuff", 
           alpha=0.3)

# grid
ax.grid(which="major", 
        axis="x", 
        color="gray")

# facecolor
ax.set_facecolor("mintcream")

# y-limits


# In[176]:


# set-up
span_range = [[24, 30], [23, 29]]

# let's plot it
fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1,
    figsize=(24, 10),
    facecolor="mintcream")

sns.lineplot(
    x="day_order", 
    y="sales", 
    data=cop, 
    lw=1.5,
    color="#E3866F",
    ax=ax1)

sns.lineplot(
    x="day_delivery", 
    y="sales", 
    data=cop,
    lw=1.5,
    color="#E3866F",
    ax=ax2)

# labels
ax1.set_xlabel("Day Order", 
               weight="bold", 
               family="fantasy", 
               fontsize=25, 
               labelpad=50)
ax1.set_ylabel("Sales", 
               weight="bold", 
               family="fantasy", 
               fontsize=25, 
               labelpad=50)
ax2.set_xlabel("Day Delivery", 
               weight="bold", 
               family="fantasy", 
               fontsize=25, 
               labelpad=50)
ax2.set_ylabel("Sales", 
               weight="bold", 
               family="fantasy", 
               fontsize=25, 
               labelpad=50)

# axis
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position("top")

# ticklabels
ax1.set_xticklabels(labels=list(np.arange(0, 35, 5)),
                    weight="bold",
                    family="fantasy",
                    fontsize=15)
ax2.set_xticklabels(labels=list(np.arange(0, 35, 5)),
                    weight="bold",
                    family="fantasy",
                    fontsize=15)
ax1.set_yticklabels(labels=list(np.arange(5000, 8500, 500)),
                    weight="bold",
                    family="fantasy",
                    fontsize=15)
ax2.set_yticklabels(labels=list(np.arange(5000, 8500, 500)),
                    weight="bold",
                    family="fantasy",
                    fontsize=15)

# limits
ax1.set_xlim(0, 31)
ax2.set_xlim(0, 31)
ax1.set_ylim(5000, 8000)
ax2.set_ylim(5000, 8000)

# axv-line
ax1.axvline(x=8, color="red", ls="--", lw=1)
ax1.axvline(x=27, color="red", ls="--", lw=1)
ax1.axvline(x=18, color="blue", ls="--", lw=1)
ax2.axvline(x=25, color="red", ls="--", lw=1)
ax2.axvline(x=2, color="blue", ls="--", lw=1)
ax2.axvline(x=13, color="blue", ls="--", lw=1)

# axv-span
ax1.axvspan(span_range[0][0], 
            span_range[0][1], 
            color="peachpuff", 
            alpha=0.3)
ax2.axvspan(span_range[1][0], 
            span_range[1][1], 
            color="peachpuff", 
            alpha=0.3)

# grids
ax1.grid(which="major", axis="x")
ax2.grid(which="major", axis="x")

# face-color
ax1.set_facecolor("mintcream")
ax2.set_facecolor("mintcream");

