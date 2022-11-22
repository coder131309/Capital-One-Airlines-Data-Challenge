import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.io as io



# Function to load csv files and convert any column with _ID to string type
def load_data_file(file_name):
    if file_name == "Airport_Codes.csv":
        df = (pd.read_csv(f"data/{file_name}",  low_memory=False)).query("TYPE in ('medium_airport','large_airport')")
    else:
        df = pd.read_csv(f"data/{file_name}",  low_memory=False)
    
    for col in df.columns:
        if "_ID" in col:
            df[f"{col}"] = df[f"{col}"].astype(str)
    return df

def boxplot(dataset,field_names):
    dataset.boxplot(column = [field_names])
    plt.grid(False)
    plt.show()

# Function to check datatype and range for numeric fields, this is run to better understand the underlying data
   
def summary_statistics(dataset):
    return (dataset.count(),dataset.dtypes, dataset.describe(), dataset.shape)

# Converts char fields to numbers

def to_number(dataset,field_names):
    for field in field_names:
        dataset[field] = pd.to_numeric(dataset[field], errors='coerce')
    return dataset[field]
    
# Used to create histogram for numeric fields, helps to understand the outliers
def hist(dataset,file_name):
    ds = dataset.select_dtypes(include=np.number)
    with open(f"{file_name}.html",'w') as f:
        for i in ds.columns:
            fig = px.histogram(ds, x=[i])
            f.write(io.to_html(fig, full_html = True))



def remove_outliers(dataset, fields, low_range, upper_range):
    for field in fields:
        median = dataset[field].median()
        min_threshold, max_threshold = dataset[field].quantile([low_range,upper_range])
        print (min_threshold)
        print (max_threshold)
        dataset.loc[dataset[field] > max_threshold,field] = np.nan
        dataset.loc[dataset[field] < min_threshold,field] = np.nan
        dataset.fillna(median,inplace=True)
    return dataset



# Used to convert arr_delay and dep_Delays into costs
def delay_costs(delay):
    if delay > 15:
        delay_costs = (delay - 15) * 75
    else:
        delay_costs = 0
    return delay_costs

# Use this to check for cost for airports
def airport_costs(type):
    if type == 'medium_airport':
        airport_costs = 5000
    elif type == 'large_airport':
        airport_costs = 10000
    else:
        airport_costs = 0
    return airport_costs


# Converts miles into dollars
def costs_per_mile(miles):
    return miles*(1.18+8)

# Converts occupancy into number of passengers for one-way flight
def number_of_passengers(occ_rate):
    return (200 * occ_rate)

# Used to calculate median - median will be used instead of 'mean' if observations have skewed distribution
def q50(x):
    return x.quantile(0.5)

# Following function creates graphs for various variables to understand the relationship between different datapoints

def graphs(gr_type,ds,x,y):
    fig = gr_type(ds,x=ds[x],y=ds[y])
    f.write(io.to_html(fig, full_html=True))
    #return fig.show()
    
                   