import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import boto3
from smart_open import open
from glob import glob


st.title('Monteria Blue Physics Calculate Integrals')

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
    return df0

filenow = st.selectbox('Select File to draw', listoffiles)
df0 = read_dataframe(filenow)
df0s = df0.loc[:, ['number', 'time', 'ch0', 'ch1']]
df0s['chunk'] = df0s.number // int(300000//750)
dfng = df0s.groupby('chunk').agg({'time':np.median, 'ch0':np.sum, 'ch1':np.sum})
dfng = dfng.iloc[1:-1, :]

dfng['ch0diff'] = dfng.ch0.diff()
cutoff = st.slider('Chose cut-off to autodetect limits', min_value = 10, max_value = 100, value = 50)
starttimes = dfng.loc[dfng.ch0diff > cutoff, 'time'].to_list()
sts = [starttimes[0]] + [v for i, v in list(enumerate(starttimes))[1:] if abs(starttimes[i-1]-v)>1]
stsg = [t - 0.2 for t in sts]
finishtimes = dfng.loc[dfng.ch0diff < -cutoff, 'time'].to_list()
fts = [finishtimes[0]] + [v for i, v in list(enumerate(finishtimes))[1:] if abs(finishtimes[i-1]-v)>1]
ftsg = [t + 0.2 for t in fts]

nzeros = df0.loc[(df0.time < stsg[0]) | (df0.time > ftsg[-1]), ['ch0','ch1']].mean()
ngzeros = dfng.loc[(dfng.time < stsg[0]) | (df0.time > ftsg[-1]), ['ch0','ch1']].mean()
dfnzeros = df0.loc[:, ['ch0','ch1']] - nzeros
dfngzeros = dfng.loc[:, ['ch0', 'ch1']] - ngzeros
dfnzeros.columns = ['chn0z', 'chn1z']
dfngzeros.columns = ['chn0z', 'chn1z']
dfnz = pd.concat([df0s, dfnzeros], axis = 1)
dfngz = pd.concat([dfng, dfngzeros], axis = 1)

#Calculate dose

ACR = st.number_input('ACR', value = 1)
calib = st.number_input('Calibration Factor cGy/nC /1000', min_value = 700, max_value = 1000, value = 1000) 
dfnz['dose'] = (dfnz.chn0z - dfnz.chn1z * ACR) * 0.03 * calib/1000

#Find pulses
pulsethreshold = st.slider('pulse thershold', min_value = 0, max_value = 20, value = 5)
maxnoradiation = dfnz.loc[(dfnz.time < stsg[0]) | (dfnz.time > ftsg[-1]), 'chn0z'].max() 
dfnz['pulse'] = dfnz.chn0z > maxnoradiation * (1 + pulsethreshold/100)

#Find coincide pulses
dfnz['pulseafter'] = dfnz.pulse.shift(-1)
dfnz['pulsecoincide'] = dfnz.pulse + dfnz.pulseafter == 2
dfnz['doseafter'] = dfnz.dose.shift(-1)
dfnz['completedose'] = dfnz.dose
dfnz.loc[dfnz.pulsecoincide, 'completedose'] = dfnz.dose + dfnz.doseafter
dfnz['singlepulse'] = dfnz.pulse & ~dfnz.pulsecoincide
dfnz['pulsecoincideafter'] = dfnz.pulsecoincide.shift()
dfnz.dropna(inplace = True)
dfnz.loc[dfnz.pulsecoincideafter, 'completedose'] = 0

showpulses = st.checkbox('show raw data in pulses')

if showpulses:
    fig1 = px.line(dfnz, x = 'time', y = 'completedose', markers = True)
    mytitle = 'Complete Dose accumulated every 750 %ss (cGy)' %u"\u00B5"
    pulsesmultiplicator = 1
else:
    fig1 = px.line(dfngz, x = 'time', y = 'chn0z', markers = True)
    mytitle = 'Charge accumuplated every 300 ms (nC)' 
    pulsesmultiplicator = 100

fig1.update_layout(
    xaxis_title = "time (s)",
    yaxis_title = mytitle
    )
for n,(s, f) in enumerate(zip(stsg, ftsg)):
    fig1.add_vline(s, line_color = 'green', line_dash = 'dash')
    fig1.add_vline(f, line_color = 'red', line_dash = 'dash')
    dfnz.loc[(dfnz.time > s) & (dfnz.time < f), 'shot'] = n
st.plotly_chart(fig1)
dfnz['chargesensor'] = dfnz.chn0z * 0.03
dfnz['chargecerenkov'] = dfnz.chn1z * 0.03
gr = dfnz.groupby('shot')
dfi = gr.agg({'time':np.min, 'chargesensor':np.sum, 'chargecerenkov':np.sum, 'pulse':np.sum, 'singlepulse':np.sum, 'dose':np.sum, 'completedose':np.sum})
dfi.columns = ['start_time(s)', 'chargesensor(nC)',  'chargecerenkov(nC)', 'pulses', 'singlepulses', 'dose(cGy)', 'completedose(cGy)']
dfi = dfi[['chargesensor(nC)', 'chargecerenkov(nC)', 'pulses', 'singlepulses',  'start_time(s)', 'dose(cGy)', 'completedose(cGy)']]
dfi['end_time(s)'] = gr['time'].max()
dfi['duration(s)'] = dfi['end_time(s)'] - dfi['start_time(s)']
dfi['Avg_dose_per_pulse(cGy)'] = dfi['completedose(cGy)'] / dfi.pulses


st.write('Result of integrals')
st.write(dfi.round(2))
   
