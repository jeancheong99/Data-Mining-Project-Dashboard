from sklearn import preprocessing, model_selection, linear_model, metrics, neighbors, cluster, decomposition, ensemble, naive_bayes, tree
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import yfinance as yf  
import os
from tqdm import tqdm
import altair as alt
alt.data_transformers.disable_max_rows()
#from owid.grapher import Chart # pip install git+https://github.com/owid/owid-grapher-py
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import math

# Custom matplotlib style
custom_style = {
    'figure.autolayout': True,
    'figure.titlesize': 20,
    'figure.figsize': (10, 5),
    'figure.dpi': 100,
    'axes.spines.top': False,
    'axes.spines.left': False,
    'axes.titlesize': 10,
    'axes.labelsize': 14,
    'axes.grid': True,
    'axes.titlelocation': 'left',
    'xtick.direction': 'inout',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'ytick.direction': 'inout',
    'ytick.right': True,
    'ytick.left': False,
    'ytick.labelright': True,
    'ytick.labelleft': False,
    'xaxis.labellocation': 'right',
    'yaxis.labellocation': 'top',
    'grid.color': '#969696',
    'font.family': 'monospace',
    'legend.fontsize': 10,
    'legend.loc': 'best',
}

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# Curated countries for this study
selected = ['Malaysia', 'Indonesia', 'Thailand', 'Singapore', 'Vietnam']

######################################################
def get_data():
    # OWID comprehensive COVID-19 datasets update daily or weekly
    world = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/latest/owid-covid-latest.csv')
    
    # Summarize mobility report from various sources: Google, Appple, Waze, TomTom encompassing few metrics on individual 
    indices = pd.read_csv('https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/summary_reports/summary_report_countries.csv')
    
    return world, indices

world, indices = get_data()

######################################################
def preprocessing_data(df1, df2, selected):
    
    # Prevent copying slice of dataframe
    world = df1.copy(); indices = df2.copy();
    
    # Filter selected columns
    world = world[['continent', 'location', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'new_tests', 'total_tests', 'total_vaccinations', 'new_vaccinations', 'population', 'population_density', 'median_age', 'aged_65_older', 'gdp_per_capita']]
    world.columns = ['Continent', 'Country', 'Total Cases', 'New Cases', 'Total Deaths', 'New Deaths', 'New Tests', 'Total Tests', 'Total Vaccinations', 'New Vaccinations', 'Population', 'Population Density', 'Median Age', 'Age 65 Older', 'GDP Per Capita']

    world = world[world['Country'].isin(selected)] # Filtered selected countries

    # Cases per hundred = Total number of Cases / Population at the start of the time interval
    world['Cases Per Hundred'] = (world['Total Cases']/world['Population'])*100
    world['Cases Per Million'] = (world['Total Cases']/world['Population'])*1000000

    # Incident Proportion = New Cases at the current time interval / Population at start of time interval
    world['Incident Proportion'] = world.apply(lambda x: x['New Cases']/x['Population'], axis=1)

    # prevalence_rate =  Total number of Cases include previous and existing / Population at current time interval (replace with Total Population at the start of time interval)
    world['Prevalence Rate'] = world.apply(lambda x: x['Total Cases']/x['Population'], axis=1)

    # Mortality rate = Total number of deaths during a given time interval / Mid-interval population (replace with Total Population at the start of time interval)
    world['Mortality Rate Per Hundred'] = (world['Total Deaths']/world['Population'])*100
    world['Mortality Rate Per Million'] = (world['Total Deaths']/world['Population'])*1000000

    world = world[~world['Continent'].isnull()] # Drop continent record

    # Min Max Scale the Indices
    for c in indices.columns[2:]:
        indices[c] = round((indices[c] - indices[c].min())/(indices[c].max() - indices[c].min()), 4)

    # Rename columns
    indices.columns = ['Country', 'Date', 'Retail', 'Grocery', 'Parks', 'Transit Station', 'Workplaces', 'Residential', 'Driving', 'Transit', 'Walking'] 

    indices = indices[indices['Country'].isin(selected)] # Filtered selected countries

    # Compute the primitive score across six metrics
    indices['Regularity'] = round(indices.loc[:, 'Retail':'Residential'].sum(axis=1).div(6), 4)

    indices['Date'] = pd.to_datetime(indices['Date']).copy() # Convert to datetime data type
    # indices.loc[:, 'Date'] = pd.to_datetime(indices['Date'])

    # Get latest date for each country
    fcindices = indices.loc[indices.groupby('Country').Date.idxmin()+70] # Get the starting indices
    lcindices = indices.loc[indices.groupby('Country').Date.idxmax()-14] # Get the ending indices

    # Creating dummy columns for wide to long transformation (Long Format)
    fcindices['Status'], lcindices['Status'] = 'Inception', 'Current'

    cindices_concated = pd.concat([fcindices, lcindices])

    # (Wide Form) 
    cindices = pd.merge(fcindices, lcindices, on='Country', suffixes=[' Inception', ' Current'])
    cindices['Change In Regularity'] = cindices['Regularity Current'] - cindices['Regularity Inception']

    # (Long Form)
    fcindices = pd.melt(fcindices, id_vars=['Country'], value_vars=fcindices.columns[2:8]).rename(columns={'variable': 'Index', 'value': 'Metric Score'})
    lcindices = pd.melt(lcindices, id_vars=['Country'], value_vars=lcindices.columns[2:8]).rename(columns={'variable': 'Index', 'value': 'Metric Score'})
    
    return world, indices, fcindices, lcindices, cindices_concated, cindices 

world, indices, fcindices, lcindices, cindices_concated, cindices, = preprocessing_data(world, indices, selected)

##########################################################################
# OWID COVID-19 Dataset
def get_data_long(selected):
    al = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')
    al = al[al['location'].isin(selected)] # Filtered selected countries
    al = al[['location', 'date', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_tests', 'new_tests', 'total_vaccinations', 'new_vaccinations', 'population']] # Filtered columns
    al.columns = ['Country', 'Date', 'Total Cases', 'New Cases', 'Total Deaths', 'New Deaths', 'Total Tests', 'New Tests', 'Total Vaccinations', 'New Vaccinations', 'Population'] # Rename columns
    al = al.loc[(al['Date'] > '2020-04-01') & (al['Date'] < '2021-10-01')] # Slice Interval date
    al.loc[:, 'Date'] = pd.to_datetime(al['Date']) # Change to datetime dtype
    
    alc = [al for _, al in al.groupby('Country')] # Group data by country
    storage = pd.DataFrame()

    # Iterate through get country group computed several indices
    for i in range(len(alc)): 
        alc[i]['Total Cases Moving Average'] = alc[i]['Total Cases'].rolling(window=7).mean()
        alc[i]['Total Deaths Moving Average'] = alc[i]['Total Deaths'].rolling(window=7).mean()
        alc[i]['Test Positivity'] = (alc[i]['New Cases']/alc[i]['New Tests']).mul(100)
        alc[i].fillna(0, inplace=True)
        alc[i]['Test Positivity'] = np.where(alc[i]['Test Positivity'] == float(0), np.NaN, alc[i]['Test Positivity'])
        storage = pd.concat([storage, alc[i]])
        
    return storage

df_long = get_data_long(selected)

########################################################################
# Get Stock Market Index data from Yahoo Finance
# Local data retrieve from investing.com 
# Data date interval 2020-04-01 ~ 2021-10-01
def get_index_data():
    # Dict of countries and correspond stock index
    # Priority on selecting index
    # 1. Comprehensive Coverage in that specific market
    # 2. Higher in Price per unit
    index_countries = {'Malaysia': 'KLSE',
                       'Singapore': 'STI',
                       'Indonesia': 'JKSE',
                       }

    index_data = pd.DataFrame() # Create empty frame for index data

    # Iterative get data from api
    for cn, ci in zip(list(index_countries.keys()), list(index_countries.values())):
        print(cn)
        current_index = yf.download('^{}'.format(ci),'2020-04-01','2021-10-01')
        current_index['Country'] = cn
        index_data = pd.concat([index_data, current_index], axis=0)

    index_data.reset_index(inplace=True) # Reset Date columns

    index_data = index_data[['Date', 'Open', 'Close', 'High', 'Low', 'Country']] # Filtered columns

    # Read data from local
    index_path = './datasets/stock_index'

    for folder in os.listdir(index_path):
        temp_i = pd.read_csv(index_path+'/'+folder)
        temp_i['Country'] = str(folder)[:-4]
        print(str(folder)[:-4])
        print('[*********************100%***********************]  1 of 1 completed')
        index_data = pd.concat([index_data, temp_i], axis=0)

    index_data.loc[:, 'Date'] = pd.to_datetime(index_data['Date']) # Convert to datetime dtype
    return index_data

index_data = get_index_data()

# Compiled datasets
def compiled_data(selected, df_long, indices, index_data):
    df = pd.merge(df_long, indices, how='left', on=['Country', 'Date'])
    df = pd.merge(df, index_data, how='left', on=['Country', 'Date'])
    
    storage = pd.DataFrame()
    grouped_df = [df for _, df in df.groupby('Country')]
    
    for c in range(len(grouped_df)):
        for i in grouped_df[c].columns[14:]:
            grouped_df[c][i] = grouped_df[c][i].fillna(method='ffill').fillna(0)
            
    storage = pd.concat(grouped_df)    
    storage.replace(',', '', regex=True, inplace=True)
    
    storage[['Open', 'Close', 'High', 'Low']] = storage[['Open', 'Close', 'High', 'Low']].astype('float64')    
    
    return storage

df = compiled_data(selected, df_long, indices, index_data)

######################################################################################################################
st.header('Exploratory Data Analysis')
dots = alt.Chart(world).transform_calculate(
        url='https://www.google.com/search?q=' + alt.datum.Country + ' Covid-19',
).mark_circle().encode(
    href='url:N',
    x=alt.X('Population Density', scale=alt.Scale(type='log')),
    y=alt.Y('Cases Per Hundred', axis=alt.Axis(title='Cases Per Hundered'), scale=alt.Scale(padding=10)),
    size=alt.Size('Population', scale=alt.Scale(range=[1, 5000])),
    color='Country',
    tooltip=['Continent', 'Country', 'Total Cases', 'Total Deaths']
).properties(title={
        'text': ['Country Cases Per Hundred vs Population Density'],
        'subtitle': ['Curated 30 Countries with respective population and population density compared with current COVID-19 cases per 100 of person']})

dots.configure_axis(
    labelFontSize=10, 
    titleFontSize=14
).properties(
    width=1400, 
    height=500
)

st.altair_chart(dots, use_container_width=True)

######################################################################################################################
st.header('Regularity Index')
dots = alt.Chart(cindices).mark_circle().encode(
    x=alt.X('Regularity Inception', axis=alt.Axis(title='Inception Regularity'), scale=alt.Scale(domain=(0.18, 0.3))),
    y=alt.Y('Regularity Current', axis=alt.Axis(title='Current Regularity'), scale=alt.Scale(zero=False)),
    tooltip=['Regularity Inception', 'Regularity Current']
)

text = dots.mark_text(
    align='right',
    baseline='middle',
    dx=-7,
    fontWeight='bold',
    fontSize=12
).encode(
    text='Country',
)

bars = alt.Chart(cindices).mark_bar().encode(
    x=alt.X('Change In Regularity'),
    y=alt.Y('Country'),
    tooltip=['Country', 'Change In Regularity']
)

alt.hconcat(dots+text.properties(width=1000, height=300), bars).configure_axis(
    labelFontSize=10, 
    titleFontSize=14
).properties(
    title={
        'text': ['Regularity Index'],
        'subtitle': ['Regularity index is a measurement for the degree of mobility in series of daily routine encompassing following activies:', 
                     'Retail, Grocery, Parks, Transit Station, Workplaces and Residential']},
)

st.altair_chart(dots, use_container_width=True)
st.altair_chart(bars, use_container_width=True)


# ######################################################################################################################
st.header('Different Metric in Comparison')
bar1 = alt.Chart(fcindices).mark_bar().encode(
    x='Metric Score',
    y='Country',
    color=alt.Color('Index')
).properties(title='Beginning of the Pandemic')

bar2 = alt.Chart(lcindices).mark_bar().encode(
    x='Metric Score',
    y='Country',
    color=alt.Color('Index')
).properties(title='Current Status')

sbc = alt.hconcat(bar1, bar2).properties(
    title={
        'text': ['Different Metrics In Comparision'],
        'subtitle': ['Comparison between beginning of the pandemic and current status on six different metrics:', 
                     'Retail, Grocery, Parks, Transit Station, Workplaces and Residential']},
)

def different_metrics(df, metric):
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X(metric, stack='zero'),
        y=alt.Y('Country:N'),
        color=alt.Color('Status'),
    )
    text = alt.Chart(df).mark_text(dx=-12, dy=3, color='white').encode(
        x=alt.X(metric, stack='zero'),
        y=alt.Y('Country:N'),
        detail='Status:N',
        text=alt.Text(metric, format='.2f')
    )
    return (bars+text)

sbs1 = alt.hconcat(different_metrics(cindices_concated, 'Retail'), different_metrics(cindices_concated, 'Grocery'))
sbs2 = alt.hconcat(different_metrics(cindices_concated, 'Parks'), different_metrics(cindices_concated, 'Transit Station'), )
sbs3 = alt.hconcat(different_metrics(cindices_concated, 'Workplaces'), different_metrics(cindices_concated, 'Residential'))
sbs = alt.vconcat(sbs1, sbs2, sbs3).properties(
    title={
        'text': ['Different Metrics In Comparision'],
        'subtitle': ['Comparison between beginning of the pandemic and current status on six different metrics:', 
                     'Retail, Grocery, Parks, Transit Station, Workplaces and Residential']},)

alt.vconcat(sbs, sbc).configure_axis(
    labelFontSize=10, 
    titleFontSize=14)

st.altair_chart(sbs, use_container_width=True)
st.altair_chart(sbc, use_container_width=True)


# ######################################################################################################################

def get_outliers(df):    
    ndf, ori = df.copy(), df.copy()

    # A list of dataframe correspond to each country
    nds_countries = [x for _, x in ndf.groupby('Country')]
    
    # Using LOF to identify local density for outlier detection
    lof = neighbors.LocalOutlierFactor()
    sdf, temp_df = pd.DataFrame(), pd.DataFrame()
    
    # Iterate through each country, scaled numerical columns respect to each country, identify and drop outliers
    
    for i in range(len(nds_countries)):
        temp_df = nds_countries[i]
        numerical_columns = temp_df.columns[2:14].tolist()
        numerical_columns.extend(temp_df.columns[-4:].tolist())
        numerical_columns.remove('Test Positivity') # Columns test positivity contains null hence drop it
        scaler = preprocessing.MinMaxScaler() 
        temp_df[numerical_columns] = scaler.fit_transform(temp_df[numerical_columns])
        country_string = str(temp_df.iloc[0,0]) # Get country name string
        datapoint = temp_df.iloc[:, 2:] # Slicing the dataframe to avoid getting datetime type and categorical string columns
        temp_df['outliers_'+country_string] = lof.fit_predict(datapoint.drop(columns=['Test Positivity']))
        sdf = pd.concat([sdf, temp_df], axis=0)
        sdf.drop(sdf[sdf.iloc[:, -1] == -1].index, axis=0, inplace=True) # From the outliers columns, drop outliers
        sdf.drop(sdf.columns[-1], axis=1, inplace=True) # Drop outliers columns
        
    ori = ori[ori.index.isin(sdf.index)]    
    return ori

# return both scaled and unscaled dataframe
df = get_outliers(df)

# ######################################################################################################################
def clustering(df):
    ndf = df.copy()
    ndf.drop(columns=['Date', 'Test Positivity'], inplace=True) # Drop date type columns and column with np.NaN
    ndf['Country'] = preprocessing.LabelEncoder().fit_transform(ndf['Country']) # Label Encode string to int
    
    # Scaled numerical columns 
    float_columns = list(ndf.select_dtypes(include=['float64']).columns)
    scaler = preprocessing.MinMaxScaler()
    ndf[float_columns] = scaler.fit_transform(ndf[float_columns])
    
    clustering = cluster.AgglomerativeClustering(n_clusters=5).fit(ndf.drop(columns=['Country']))
    
    return clustering.labels_

df['Cluster'] = clustering(df)

def plot_cluster(df):
    
    ndf = df.copy()
    ndf['Cluster'] = ndf.apply(lambda x: 'Cluster '+str(x['Cluster']), axis=1)
    
    cluster_chart = []
    for c in ndf['Country'].unique():
        chart = alt.Chart(ndf[ndf['Country']==c]).mark_line().encode(
            x='Date:T',
            y='Total Cases',
            color='Cluster:N',
            tooltip=['Date', 'Total Cases', 'Cluster']
        ).properties(width=200, height=200, title='Cluster for {}'.format(c))
        cluster_chart.append(chart)
        
    count_cluster = ndf['Cluster'].value_counts().to_frame().reset_index().rename(columns={'index': 'Cluster', 'Cluster': 'Counts'})

    cc = alt.Chart(count_cluster).mark_bar().encode(
        y=alt.Y('Cluster', axis=alt.Axis(title='Cluster')),
        x=alt.X('Counts', axis=alt.Axis(title='Counts')),
        color='Cluster:N',
        tooltip=['Cluster', 'Counts']
    ).properties(title='Count Per Cluster')
    
    text = cc.mark_text(align='left', dx=5).encode(text='Counts')
    
    return alt.vconcat(cc+text, alt.hconcat(*cluster_chart[0:5])).configure_axis(
        labelFontSize=10, 
        titleFontSize=14).properties(title={
        'text': ['Identify Cluster Trend'],
        'subtitle': ['Purpose of clustering is to identify potential growth trend for a given time frame across 5 different countries']},)

st.header('Identify Cluster Trend')
st.altair_chart(plot_cluster(df), use_container_width=True)

# ######################################################################################################################
def mas_state_covid():
    cases = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_state.csv')
    cases = cases.iloc[:, :6]
    cases['cases_total'] = cases['cases_new'].cumsum()
    deaths = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/deaths_state.csv')
    deaths = deaths.iloc[:, :4]
    deaths['deaths_total'] = deaths['deaths_new'].cumsum()
    hospital = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/hospital.csv')
    hospital.drop(columns=['beds', 'beds_noncrit'], inplace=True)
    icu = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/icu.csv')
    icu.drop(columns=['beds_icu', 'beds_icu_rep', 'beds_icu_total', 'vent', 'vent_port'], inplace=True)
    tests = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/tests_state.csv')
    tests['tests_new'] = tests.apply(lambda x: x['rtk-ag'] + x['pcr'], axis=1)
    tests['tests_total'] = tests['tests_new'].cumsum()
    tests.drop(columns=['rtk-ag', 'pcr'], inplace=True)
    state = pd.merge(cases, deaths, how='left', on=['date', 'state'])
    state = pd.merge(state, hospital, how='left', on=['date', 'state'])
    state = pd.merge(state, icu, how='left', on=['date', 'state'])
    state = pd.merge(state, tests, how='left', on=['date', 'state'])
    state['date'] = pd.to_datetime(state['date'])
    state.fillna(0, inplace=True)
    
    return state
# ######################################################################################################################
# st.header('Animated State Cases')
# state = mas_state_covid()
# Chart(state).mark_line().encode(x='date', y='cases_new', c='state').label('Malaysia State Cases New Breakdown').interact(scale_control=True)
# ######################################################################################################################
def vaccinations_data():
    vax = pd.read_csv('https://raw.githubusercontent.com/CITF-Malaysia/citf-public/main/vaccination/vax_state.csv')
    vax = vax[['date', 'state', 'daily', 'cumul']]
    cases = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_state.csv')
    cases = cases[cases['date'] > '2021-02-23']
    cases = cases[['date', 'state', 'cases_new']]
    combine = pd.merge(vax, cases, how='left', on=['date', 'state'])
    combine['date'] = pd.to_datetime(combine['date'])
    combine.rename(columns={'date': 'Date', 'state': 'State', 'daily': 'Daily Vaccinations', 'cumul': 'Cumulative Vaccinated', 'cases_new': 'Daily New Cases'}, inplace=True)
    combine.set_index('Date', inplace=True)
    
    for c in combine.columns[1:]:
        combine[c] = round((combine[c] - combine[c].min())/(combine[c].max() - combine[c].min()), 4)

#     combine = pd.melt(combine, id_vars='State', value_vars=['Daily Vaccinations', 'Cumulative Vaccinated', 'Daily New Cases'], value_name='Scaled Values', ignore_index=False)
#     combine.rename(columns={'variable': 'Variable'}, inplace=True)

    display(combine.plot(kind='line', figsize=(20,10), title='Does Vaccine Program Helps ?', ylabel='Scaled Values', fontsize=14))

st.header('Does Vaccine Program Helps?')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(vaccinations_data())
# ######################################################################################################################
def vaccinations_rate():
    vax = pd.read_csv('https://raw.githubusercontent.com/CITF-Malaysia/citf-public/main/vaccination/vax_state.csv')
    pop = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/static/population.csv')
    vax = vax[['date', 'state', 'cumul_full']]
    vax['Population'] = vax['state'].copy()
    population = dict(zip(pop['state'], pop['pop']))
    vax.replace({'Population': population}, inplace=True)
    vax['left'] = vax['Population']-vax['cumul_full']
    vax['rate'] = (vax['cumul_full']/vax['Population'])*100
    vax['date'] = pd.to_datetime(vax['date'])
    
    whole_malaysia_rate = list()
    for d in vax['date'].unique():
        temp = vax[vax['date']==d]
        whole_malaysia_rate.append(temp['rate'].sum()/len(temp))
        
    whole_malaysia_rate = pd.DataFrame(data={'Vaccination Rate': whole_malaysia_rate}, index=vax['date'].unique())
    fig, ax = plt.subplots(figsize=(10,5))
    plt.plot(whole_malaysia_rate.index, whole_malaysia_rate['Vaccination Rate'], label='Vaccination Rate')
    ax.hlines(y=80, xmin=whole_malaysia_rate.index.min(), xmax=whole_malaysia_rate.index.max(), linewidth=2, color='g', label='Herd Immunity Index')
    plt.legend(); plt.xlabel('Date'); plt.ylabel('Vaccination Rate')
    plt.suptitle('Malaysia Overview', x=.25, y=1, fontsize=18, fontweight='ultralight')
    plt.ylim([0,100])
    
    display(whole_malaysia_rate.index.max())
    vax.columns = ['Date', 'State', 'Cumulative Fully Vaccinated', 'Population', 'Leftover Individual', 'Vaccination Rate']
    
    return alt.Chart(vax).mark_line().encode(
        x='Date',
        y='Vaccination Rate',
        color='State',
        tooltip=['Date', 'Vaccination Rate', 'State', 'Cumulative Fully Vaccinated', 'Leftover Individual']).configure_axis(
        labelFontSize=10, 
        titleFontSize=14).properties(width=500, title={
        'text': ['Vaccinations Rate State Break In Malaysia'],
        'subtitle': ['Overflow cause by non naive identity or vaccine assigned to cross border innoculator']},)
    
    # display(vax, lastest_frame)
st.header('When We Able Hit Herd Immunity?')
st.altair_chart(vaccinations_rate())
# ######################################################################################################################
def prepare_and_preprocessing():
    ts_data = pd.read_csv('./datasets/ts_data.csv')
    ts_data['date'] = pd.to_datetime(ts_data['date'], dayfirst=True, format='%d/%m/%Y')
    ts_data.set_index('date', inplace=True)
    data = pd.read_csv('./datasets/data.csv')
    data.drop(columns='date', inplace=True)
    
    return ts_data, data

ts_data, data = prepare_and_preprocessing()
# ######################################################################################################################
# Commensurate lstm.py  
def lstm_model(df, cn, window, forecast, epochs, bs, style, n_samples=1, seed=True, plot=True):
    """
    Train and forecast LSTM. 
    Parameters:
    -----------
        df: pandas.DataFrame
            The Dataframe with datetime.date as index with x,y 
        cn: str
            The forecast columns name
        window: int
            The window size for time series
        forecast: int
            The number of days to forecast
        epochs: int
            The number of model training epochs
        bs: int
            The number of model training batch_size
        style: dict
            Custom matplotlib style
        n_samples: int, default=1
            The number of model created for computing confidence interval
        plot: boolean, default=True, optional
            Visualize plot
    ----------
    Returns: 
        - 
    """
    # Preprocessing                                                       
    data = df.filter([cn]).values
    nmin, nmax = data.min(), data.max()
    scaled_data = ((df[cn]-nmin)/(nmax-nmin)).values.reshape(-1,1)
    split = round(0.8*len(data)) 
    train = scaled_data[:split]
    trainX, trainy = [], []
    for i in range(window, len(train)):
        trainX.append(train[i-window:i, 0])
        trainy.append(train[i, 0])
    trainX, trainy = np.array(trainX), np.array(trainy)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    
    tf.random.set_seed(7) # set seed for reproducibility
        
    # Training
    lstm_input = tf.keras.layers.Input(shape=(trainX.shape[1], 1))
    lstm_1 = tf.keras.layers.LSTM(units=50, return_sequences=True)(lstm_input)
    lstm_2 = tf.keras.layers.LSTM(units=150)(lstm_1)
    lstm_output = tf.keras.layers.Dense(units=1)(lstm_2)
    model = tf.keras.models.Model(inputs=lstm_input, outputs=lstm_output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(trainX, trainy, epochs=epochs, batch_size=bs, verbose=0)
       
    # Testing
    test = scaled_data[split-window:]
    testX, testy = [], data[split:]
    for i in range(window, len(test)):
        testX.append(test[i-window:i, 0])
    testX = np.array(testX)
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    prediction = model.predict(testX)
    prediction = (nmax-nmin)*prediction+nmin
    prediction_series = pd.Series(prediction.flatten(), index=df.index[split:])
        
    # Forecast
    results = pd.DataFrame()
    for n in range(n_samples):
        forecast_list = scaled_data[-window:]
        lstm_input = tf.keras.layers.Input(shape=(trainX.shape[1], 1))
        lstm_1 =  tf.keras.layers.LSTM(units=50, return_sequences=True)(lstm_input)
        lstm_2 =  tf.keras.layers.LSTM(units=150)(lstm_1)
        lstm_output = tf.keras.layers.Dense(units=1)(lstm_2)
        model = tf.keras.models.Model(inputs=lstm_input, outputs=lstm_output)
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(trainX, trainy, epochs=epochs, batch_size=bs, verbose=0)
        for _ in range(forecast):
            x = forecast_list[-window:]
            x = x.reshape((1, window, 1))
            out = model.predict(x)[0][0]
            forecast_list = np.append(forecast_list, out)

        forecast_list = forecast_list[window-1:]
        forecast_list = (nmax-nmin)*forecast_list+nmin
        results['Forecast{}'.format(n+1)] = forecast_list.copy()
        
    last_date = df.index.values[-1]
    forecast_dates = pd.date_range(last_date, periods=forecast+1).tolist()               
    results['Date'] = forecast_dates 
    results.set_index('Date', inplace=True)
    results['Mean'] = results.mean(axis=1)
    results['_CI'], results['+CI'] = results['Mean']-results.std(axis=1)/np.sqrt(n_samples)*1.96, results['Mean']+results.std(axis=1)/np.sqrt(n_samples)*1.96

    # Output
    print('Window: {} | Forecast: {} | Epochs: {} | N_samples: {}'.format(window, forecast, epochs, n_samples))
    print('RMSE:', np.sqrt(np.mean(testy-prediction)**2))
    if plot:
        with plt.style.context(style):
            fig, ax = plt.subplots()
            plt.plot(df[cn][:split])
            plt.plot(df[cn][split:])
            plt.plot(prediction_series)
            plt.plot(results['Mean'])
            plt.fill_between(results.index, results['_CI'], results['+CI'], color='#ababab')
            plt.legend(['Train', 'Test', 'Prediction', 'Forecast', 'CI'], ncol=5, loc='upper left')
            plt.title('Forward forecast n coming day values', color='grey')
            plt.suptitle('LSTM Model Forecast', ha='left', x=.015, y=.95)
            fig.autofmt_xdate(); ax.set_xlabel('Date'); ax.set_ylabel('Values'); ax.yaxis.set_label_position('right')

st.header('Forecasting on COVID-19 Malaysia New Cases Moving Average with LSTM MODEL')

with st.form(key='my_form'):
    window = int(st.number_input('Insert number of window', value=45))
    forecast = int(st.number_input('Insert number of forecast', value=30))
    epochs = int(st.number_input('Insert number of epochs', value=1))
    bs = int(st.number_input('Insert number of bs', value=8))
    n_samples = int(st.number_input('Insert number of samples', value=1))

    submit = st.form_submit_button(label='Submit')

lstm = lstm_model(df=ts_data, cn='cases', window=window, forecast=forecast, epochs=epochs, bs=bs, style=custom_style, n_samples=n_samples, plot=True)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(lstm) 
######################################################################################################################
def get_mas_covid2():
    cases = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_malaysia.csv')
    cases.drop(cases.columns[2:], axis=1, inplace=True)
    cases['cases_total'] = cases['cases_new'].cumsum()
    deaths = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/deaths_malaysia.csv')
    deaths.drop(deaths.columns[2:], axis=1, inplace=True)
    deaths['deahts_total'] = deaths['deaths_new'].cumsum()
    tests = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/tests_malaysia.csv')
    tests['tests_new'] = tests.apply(lambda x: x['rtk-ag'] + x['pcr'], axis=1)
    tests['tests_total'] = tests['tests_new'].cumsum()
    tests.drop(columns=['rtk-ag', 'pcr'], inplace=True)
    vaccinations = pd.read_csv('https://raw.githubusercontent.com/CITF-Malaysia/citf-public/main/vaccination/vax_malaysia.csv')
    vaccinations = vaccinations[['date', 'daily', 'cumul']]
    vaccinations.rename(columns={'daily': 'vaccinations_new', 'cumul': 'vaccinations_total'}, inplace=True)
    
    mas = pd.merge(tests, cases, how='left', on='date')
    mas = pd.merge(mas, deaths, how='left', on='date')
    mas = pd.merge(mas, vaccinations, how='left', on='date')
    mas.fillna(0, inplace=True)
    
    mas['date'] = pd.to_datetime(mas['date'])
    mas = mas.loc[(mas['date'] > '2020-02-01') & (mas['date'] < '2021-10-10')]
    mas.drop(columns=['date'], inplace=True)
    
    return mas
######################################################################################################################
# Commensurate lasso.py
def lasso_model(df, cn, ntf, lag, forecast, style, plot):
    """
    Timeseries to supervised and forecast Lasso
    Parameters:
    -----------
        df: pandas.DataFrame
            The dataframe with x,y
        cn: str
            The forecast columns name 
        ntf: int
            The number of top feature
        lag: int
            The number time series lag
        forecast: integer
            The number days to forecast 
        style: dict
            Custom matplotlib style
        plot: boolean, optional
            Visualize plot
    -----------
    Returns:
        - 
    """
    # Preprocessing    
    trainX = df.sample(frac=.8, random_state=7); testX = df.drop(trainX.index)
    trainy, testy = trainX.pop(cn), testX.pop(cn)
    
    l = linear_model.Lasso(max_iter=3000, random_state=7, normalize=True) 
    l.fit(trainX, trainy)
    pred = l.predict(testX)
    feature_df = pd.DataFrame({'Feature': df.drop(columns=cn).columns, 'Coefficient': l.coef_})
    tf = feature_df.sort_values(by='Coefficient', ascending=False).iloc[:ntf, 0].to_list()

    # Feature selection
    lasso = linear_model.Lasso(max_iter=3000, random_state=7, normalize=True)
    lasso.fit(trainX[tf], trainy)
    apred = lasso.predict(testX[tf])
    
    # Training
    s_feature_model = list() 
    for x in range(len(tf)):
        data = df[tf[x]].tolist()
        n_vars = 1 if type(data) is list else data.shape[1]
        temp = pd.DataFrame(data)
        cols, names = list(), list()

        for i in range(lag, 0, -1): # Input sequence (t-n, ... t-1)
            cols.append(temp.shift(i))
            names += [('%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        cols.append(temp.shift(-1)) # Current forecast t
        names += [('%d(t)' % (j+1)) for j in range(n_vars)]
        
        new_df = pd.concat(cols, axis=1)
        new_df.columns = names
        new_df.dropna(inplace=True) 
        new_df = new_df.add_prefix(str(tf[x])+' ')
        
        XX, yy = new_df.iloc[:, :lag], new_df.iloc[:, lag:lag+1]
        trainXX = XX.sample(frac=.8, random_state=7); testXX = XX.drop(trainXX.index)
        trainyy = yy.sample(frac=.8, random_state=7); testyy = yy.drop(trainyy.index)
        model = linear_model.Lasso(max_iter=3000, random_state=7, normalize=True)
        model.fit(trainXX, trainyy)
        s_feature_model.append(model)
        # print(np.sqrt(np.mean(testyy.values.flatten()-model.predict(testXX))**2))
             
    # Forecast
    result = df[tf].iloc[-lag:, :].copy()
    # display(result)
    result['Forecast'] = 0
    for _ in range(forecast):
        temp_result = list()
        for x in range(len(s_feature_model)):
            temp_result.append(int(s_feature_model[x].predict(result.iloc[-lag:, x].values.reshape(1, -1))))
        temp_result.append(int(lasso.predict(np.array(temp_result).reshape(1, -1))))
        result = result.append(pd.Series(temp_result, index=result.columns), ignore_index=True)
    
    moving_average_window = 7
    result['Forecast Moving Average'] = result['Forecast'].rolling(window=moving_average_window).mean()
    
    # Output
    # feature_df.plot.bar(x='Feature', y='Coefficient')
    # print('Without feature Selection: {}'.format(np.sqrt(np.mean(testy-pred)**2)))
    # print('With feature selection: {}'.format(np.sqrt(np.mean(testy-apred)**2)))
    print('Forecast: {} | Lag: {}'.format(forecast, lag))
    if plot:
        with plt.style.context(style):
            fig, ax = plt.subplots()
            plt.plot(df[cn])
            plt.plot(pd.Series(result['Forecast'][-forecast:].tolist(), index=np.arange(int(df.index[-1:].values[0]), int(df.index[-1:].values[0])+forecast)))
            plt.plot(pd.Series(result['Forecast Moving Average'][-forecast+moving_average_window:].tolist(), index=np.arange(int(df.index[-1:].values[0])+moving_average_window, int(df.index[-1:].values[0])+forecast)))
            plt.legend(['Actual', 'Forecast', 'Forecast Moving Average'], ncol=5, loc='upper left')
            plt.title('Interpret lasso coefficient for feature selection and forecast target with previous values', color='grey')
            plt.suptitle('Lasso Model Forecast', ha='left', x=.015, y=.95, fontsize=20)
            ax.set_xlabel('Range'); ax.set_ylabel('Values'); ax.yaxis.set_label_position('right')
######################################################################################################################
st.header('Forecasting on COVID-19 Malaysia New Cases Moving Average with LASSO MODEL')

with st.form(key='my_form_2'):
    ntf = int(st.number_input('Insert number of ntf', value=3))
    lag = int(st.number_input('Insert number of lag', value=21))
    forecast_2 = int(st.number_input('Insert number of forecast', value=60))

    submit_2 = st.form_submit_button(label='Submit')

lasso = lasso_model(df=data, cn='cases', ntf=ntf, lag=lag, forecast=forecast_2, style=custom_style, plot=True)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(lasso)
######################################################################################################################
individual = pd.read_csv('./datasets/survived.csv')

def abstract_and_portray(df, custom_style):

    ndf = df.copy()
    # print('Innoculated first jab')
    # print('Survived: {} | Not Survived: {}'.format(ndf[ndf['days_dose1'] > 0]['survived'].value_counts().values[0], ndf[ndf['days_dose1'] > 0]['survived'].value_counts().values[1]))
    # print('Probability of Survived: {}'.format(round(100-(ndf[ndf['days_dose1'] > 0]['survived'].value_counts().values[1]/ndf[ndf['days_dose1'] > 0]['survived'].value_counts().values[0])*100), 4))
    # print(' ')

    # print('Innoculated first jab more than 14 days')
    # print('Survived: {} | Not Survived: {}'.format(ndf[ndf['days_dose1'] > 14]['survived'].value_counts().values[0], ndf[ndf['days_dose1'] > 14]['survived'].value_counts().values[1]))
    # print('Probability of Survived: {}'.format(round(100-(ndf[ndf['days_dose1'] > 14]['survived'].value_counts().values[1]/ndf[ndf['days_dose1'] > 14]['survived'].value_counts().values[0])*100), 4))
    # print(' ')

    # print('Fully vaccinated')
    # print('Survived: {} | Not Survived: {}'.format(ndf[ndf['days_dose2'] > 0]['survived'].value_counts().values[0], ndf[ndf['days_dose2'] > 0]['survived'].value_counts().values[1]))
    # print('Probability of Survived: {}'.format(round(100-(ndf[ndf['days_dose2'] > 0]['survived'].value_counts().values[1]/ndf[ndf['days_dose2'] > 0]['survived'].value_counts().values[0])*100), 4))
    # print(' ')

    # print('Fully vaccinated after 2 weeks')
    # print('Survived: {} | Not Survived: {}'.format(ndf[ndf['days_dose2'] > 14]['survived'].value_counts().values[0], ndf[ndf['days_dose2'] > 14]['survived'].value_counts().values[1]))
    # print('Probability of Survived: {}'.format(round(100-(ndf[ndf['days_dose2'] > 14]['survived'].value_counts().values[1]/ndf[ndf['days_dose2'] > 14]['survived'].value_counts().values[0])*100), 4))
    # print(' ')

    ndf = ndf[ndf['age']>0] 

    rfr = ensemble.RandomForestClassifier()
    # nb = naive_bayes.GaussianNB()
    dtre = tree.DecisionTreeClassifier()

    ndf['age'] = pd.cut(x=ndf['age'], bins=[0, 20, 30, 40, 50, 60, 70, 80, 90, 150], labels=['0-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90+'])
    ndf['state'] = preprocessing.LabelEncoder().fit_transform(ndf['state'])
    ndf = pd.get_dummies(ndf, columns=['age', 'gender', 'malaysian'])

    X = ndf.drop(columns=['survived'])
    y = ndf['survived']
    
    os = SMOTE(random_state=10)
    osX, osy = os.fit_resample(X, y)
    trainX, testX, trainy, testy = model_selection.train_test_split(osX, osy, test_size=.2, random_state=7)
    
    rfr.fit(trainX, trainy)
    # nb.fit(trainX, trainy)
    dtre.fit(trainX, trainy)
    predy_rfr = rfr.predict(testX)
    # predy_nb = nb.predict(testX)
    predy_tre = dtre.predict(testX)
    
    # print('Random Forest Classifier')
    # print('F1-Score: {}'.format(metrics.f1_score(testy, predy_rfr)))
    # print('Precision-Score: {}'.format(metrics.precision_score(testy, predy_rfr)))
    # print('Recall-Score: {}'.format(metrics.recall_score(testy, predy_rfr)))
    # print(' ')
    
    # print('Gaussian Naive Bayes Classifier')
    # print('F1-Score: {}'.format(metrics.f1_score(testy, predy_nb)))
    # print('Precision-Score: {}'.format(metrics.precision_score(testy, predy_nb)))
    # print('Recall-Score: {}'.format(metrics.recall_score(testy, predy_nb)))
    # print(' ')
    
    # print('Decision Tree Classifier')
    # print('F1-Score: {}'.format(metrics.f1_score(testy, predy_tre)))
    # print('Precision-Score: {}'.format(metrics.precision_score(testy, predy_tre)))
    # print('Recall-Score: {}'.format(metrics.recall_score(testy, predy_tre)))
    
    result_list = list()
    for m in [rfr, dtre]:
        predicted = m.predict_proba(testX)
        predicted = predicted[:, 1] 
        result_list.append(predicted)
    
    fpr_rfr, tpr_rfr, thresholds_rfr = metrics.roc_curve(testy, result_list[0]) 
    # fpr_nb, tpr_nb, thresholds_nb = metrics.roc_curve(testy, result_list[1]) 
    fpr_dtre, tpr_dtre, thresholds_dtre = metrics.roc_curve(testy, result_list[1]) 
        
    st.header('Receiver Operating Characteristic (ROC) Curve')

    with plt.style.context(custom_style):
        fig, ax, = plt.subplots()
        plt.plot(fpr_rfr, tpr_rfr, color='orange', label='Random Forest') 
        # plt.plot(fpr_nb, tpr_nb, color='blue', label='Gaussian NB') 
        plt.plot(fpr_dtre, tpr_dtre, color='red', label='Decision Tree') 
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.suptitle('Receiver Operating Characteristic (ROC) Curve')
        ax.yaxis.set_label_position('right')
        plt.legend()

    col1, col2 = st.columns(2)

    precision_rfr = metrics.precision_score(testy, predy_rfr)
    precision_rfr = "{:.2f}%".format(precision_rfr*100)
    col1.metric(label='Random Forest Precision', value=precision_rfr)

    precision_tre = metrics.precision_score(testy, predy_tre)
    precision_tre = "{:.2f}%".format(precision_tre*100)
    col2.metric(label='Decision Tree Precision', value=precision_tre)


    
roc = abstract_and_portray(individual, custom_style)
st.pyplot(roc)
######################################################################################################################


