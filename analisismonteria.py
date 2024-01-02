import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import boto3
from smart_open import open
from glob import glob


st.title('Monteria Blue Physics Analysis')

s3 = boto3.client('s3')

response = s3.list_objects_v2(Bucket='monteria', Delimiter='/')

listofdirectories = []

for common_prefix in response.get('CommonPrefixes', []):
        # Extract the folder name from the CommonPrefixes
        folder_name = common_prefix['Prefix'].rstrip('/')
        listofdirectories.append(folder_name)

directory1 = st.selectbox('Select Directory', listofdirectories)

response2 = s3.list_objects_v2(Bucket='monteria', Prefix=directory1)

#list of files in director

listoffiles = [file['Key'] for file in response2.get('Contents', [])][1:]

@st.cache_data
def read_files():
    dates = []
    notes = []
    for key in listoffiles:
        with open (f's3://monteria/{key}') as filenow:
            datenow = filenow.readline().strip()[11:]
            dates.append(datenow)
            notenow = filenow.readline().strip()[7:]
            notes.append(notenow)
    #st.write(dates)
    df = pd.DataFrame({'file':listoffiles, 'date':dates, 'note':notes})
    df['datetime'] = pd.to_datetime(df.date)
    df.sort_values(by='datetime', inplace = True)
    df.reset_index(inplace = True, drop = True)
    df.drop('date', inplace = True, axis = 1)
    return df

df = read_files()
st.write('List of files collected')
st.write(df)

@st.cache_data
def read_dataframe(file):
    path = f's3://monteria/{file}'
    df0 = pd.read_csv(path, skiprows = 4)
    #df0 = pd.read_csv(file, skiprows = 4) 
    return df0

filenow = st.selectbox('Select File to draw', df.file)
df0 = read_dataframe(filenow)
df0s = df0.loc[:, ['time', 'ch0', 'ch1']]
csv = df0s.to_csv().encode('utf-8')
st.dataframe(df0s)
st.download_button(
        label = 'Download data as CSV',
        data = csv,
        file_name = filenow
        )
lasttime = df0s.iloc[-1,0]
zeros = df0.loc[(df0.time < 1) | (df0.time > lasttime - 1) , 'ch0':].mean()
dfzeros = df0.loc[:,'ch0':]-zeros
dfzeros.columns = ['ch0z', 'ch1z']
dfz = pd.concat([df0, dfzeros], axis = 1)
dfz['chunk'] = dfz.number // int(300000/750)
dfzs = dfz.loc[:, ['time', 'ch0z']]
dfzs.columns =  ['time', 'reading']
dfzs['ch'] = 'sensor'

dfzc = dfz.loc[:,['time', 'ch1z']]
dfzc.columns = ['time', 'reading']
dfzc['ch'] = 'cerenkov'

dfztp = pd.concat([dfzs, dfzc])

dfgb = dfz.groupby('chunk').agg({'time':np.median, 'ch0z':np.sum, 'ch1z':np.sum})
dfg = dfgb.iloc[:-1,:]

dfgs = dfg.loc[:,['time', 'ch0z']]
dfgs.columns = ['time', 'reading']
dfgs['ch'] = 'sensor'

dfgc = dfg.loc[:, ['time', 'ch1z']]
dfgc.columns = ['time', 'reading']
dfgc['ch'] = 'cerenkov'

dfg = pd.concat([dfgs, dfgc])

fig1 = px.line(dfg, x='time', y='reading', color = 'ch',  markers = True, title = 'Summary every 300 ms')

st.plotly_chart(fig1)

showp = st.checkbox('show rawdata in pulses')

if showp:
    fig2 = px.line(dfztp, x = 'time', y = 'reading', color = 'ch', markers = True, title = 'Raw Data')
    st.plotly_chart(fig2)
