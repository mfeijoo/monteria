import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import boto3
from smart_open import open
from glob import glob


st.title('Monteria Blue Physics Calculate OFs')

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
def read_dataframe(file):
    path = f's3://monteria/{file}'
    df0 = pd.read_csv(path, skiprows = 4)
    #df0 = pd.read_csv(file, skiprows = 4) 
    return df0

filenow = st.selectbox('Select File to draw', listoffiles)
df0 = read_dataframe(filenow)
df0s = df0.loc[:, ['time', 'ch0', 'ch1']]
df0s['chunk']
dfng = dfng.iloc[:-1, :]

dfng['ch0diff'] = dfng.ch0z.diff()
cutoff = st.slider('Chose cut-off to autodetect limits', min_value = 20, max_value = 200, value = 50)
starttimes = dfng.loc[dfng.ch0diff > cutoff, 'time'].to_list()
sts = [starttimes[0]] + [v for i, v in list(enumerate(starttimes))[1:] if abs(starttimes[i-1]-v)>1]
stsg = [t - 0.2 for t in sts]
finishtimes = dfng.loc[dfng.ch0diff < -cutoff, 'time'].to_list()
fts = [finishtimes[0]] + [v for i, v in list(enumerate(finishtimes))[1:] if abs(finishtimes[i-1]-v)>1]
ftsg = [t + 0.2 for t in fts]

nzeros = df0.loc[(df0.time < stsg[0]) | (df0.time > ftsg[-1]), ['ch0','ch1']].mean()
dfnzeros = dfz.loc[:, ['ch0','ch1']] - nzeros
dfnzeros.columns = ['chn0z', 'chn1z']
dfnz = pd.concat([dfz, dfnzeros], axis = 1)

#Find pulses

maxzeros = dfnz.loc[(dfnz.time < stsg[0]) | (dfnz.time > ftsg[-1]), 'chn0z'].max()
pulsethreshold = st.slider('Chose threshold for pulses', min_value = 1, max_value = 20, value = 5)
dfnz['pulse'] = (dfnz.chn0z > maxzeros * (1 + pulsethreshold/100))

#Find coincide pulses
dfnz['pulseafter'] = dfnz.pulse.shift(-1)
dfnz['pulsecoincide'] = dfnz.pulse + dfnz.pulseafter == 2
dfnz['singlepulse'] = dfnz.pulse
dfnz['pulsecoincideafter'] = dfnz.pulsecoincide.shift()
dfnz.dropna(inplace = True)
dfnz.loc[dfnz.pulsecoincideafter, 'singlepulse'] = False
numberofpulses = dfnz.pulse.sum()
numberofpulsescoincide = dfnz.pulsecoincide.sum()
st.write('Number of Pulses: %s' %numberofpulses)
st.write('Number of pulses coinciding: %s' %numberofpulsescoincide)
st.write('Number of singlepulses: %s' %dfnz.singlepulse.sum())

ACR = st.number_input('ACR', value = 1)

#Find complete dose of pule and pulse after

dfnz['dose'] = dfnz.chn0z - dfnz.chn1z * ACR
dfnz['doseafter'] = 0
dfnz.loc[dfnz.pulsecoincide, 'doseafter'] = dfnz.dose
dfnz['completedose'] = dfnz.dose + dfnz.doseafter.shift(-1)


dfn1z = dfnz.loc[:, ['number', 'time', 'chn0z']]
dfn1z.columns = ['number', 'time', 'reading']
dfn1z['ch'] = 'sensor'
dfn2z = dfnz.loc[:, ['number', 'time', 'chn1z']]
dfn2z.columns = ['number', 'time', 'reading']
dfn2z['ch'] = 'cerenkov'
dfn3z = dfnz.loc[:, ['number', 'time', 'dose']]
dfn3z.columns = ['number', 'time', 'reading']
dfn3z['ch'] = 'dose'
dfnpz = dfnz.loc[:, ['number', 'time', 'pulse']]
dfnpz.columns = ['number', 'time', 'reading']
dfnpz['ch'] = 'pulse'
dfncpz = dfnz.loc[:, ['number', 'time', 'completedose']]
dfncpz.columns = ['number', 'time', 'reading']
dfncpz['ch'] = 'completedose'
dfnztp = pd.concat([dfn1z, dfn2z, dfn3z, dfnpz, dfncpz]) 
dfnztp['readingC'] = dfnztp.reading * 0.03 


fig3 = px.scatter(dfnztp, x='time', y='readingC', color = 'ch', title = 'Data set to zero')
fig3.update_layout(
    xaxis_title = "time (s)",
    yaxis_title = "Charge accumulated every 750 %ss (nC)" %u"\u00B5"
    )
for n,(s, f) in enumerate(zip(stsg, ftsg)):
    fig3.add_vline(s, line_color = 'green', line_dash = 'dash')
    fig3.add_vline(f, line_color = 'red', line_dash = 'dash')
    dfnz.loc[(dfnz.time > s) & (dfnz.time < f), 'shot'] = n
    

st.plotly_chart(fig3)

dfnz['chargesensor'] = dfnz.ch0z * 0.03
dfnz['chargecerenkov'] = dfnz.ch1z * 0.03
gr = dfnz.groupby('shot')
dfi = gr.agg({'time':np.min, 'chargesensor':np.sum, 'chargecerenkov':np.sum, 'pulse':np.sum, 'singlepulse':np.sum, 'dose':np.sum, 'completedose':np.sum})
dfi.columns = ['start_time(s)', 'chargesensor(nC)',  'chargecerenkov(nC)', 'pulses', 'singlepulses', 'dose(cGy)', 'completedose(cGy)']
dfi = dfi[['chargesensor(nC)', 'chargecerenkov(nC)', 'pulses', 'singlepulses',  'start_time(s)', 'dose(cGy)', 'completedose(cGy)']]
dfi['end_time(s)'] = gr['time'].max()
dfi['duration(s)'] = dfi['end_time(s)'] - dfi['start_time(s)']
dfi['Avg_dose_per_pulse(cGy)'] = dfi['dose(cGy)'] / dfi.pulses


st.write('Result of integrals')
st.write(dfi.round(2))
   
