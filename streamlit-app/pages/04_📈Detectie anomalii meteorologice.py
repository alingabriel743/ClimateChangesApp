import streamlit as st
import pandas as pd
import numpy as np
import ast
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from sklearn.ensemble import IsolationForest
from io import BytesIO
import plotly.graph_objs as go
from autoencoder import *
from sklearn.cluster import KMeans

tf.random.set_seed(42)

@st.cache_data()
def load_data():
    data = pd.read_csv('data/set_tradus.csv') 
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['phenomena'] = data['phenomena'].apply(lambda x: ast.literal_eval(x))
    return data

def get_season(month):
    if month in [12, 1, 2]:
        return 'Iarnă'
    elif month in [3, 4, 5]:
        return 'Primăvară'
    elif month in [6, 7, 8]:
        return 'Vară'
    elif month in [9, 10, 11]:
        return 'Toamnă'

optimizer_dict = {
    'adam': Adam(),
    'sgd': SGD(),
    'rmsprop': RMSprop()
}

loss_dict = {
    'mean_squared_error': MeanSquaredError(),
    'binary_crossentropy': BinaryCrossentropy()
}

all_numerical_features = ['pm10_quality', 'temperature', 'air_pressure', 'humidity']
data = load_data()
data['season'] = data['month'].apply(get_season)
if 'data_anom' not in st.session_state:
    st.session_state.data_anom = None

def standardize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data[all_numerical_features])

data_standardized = standardize_data(data)
input_shape = data_standardized.shape[1]
autoencoder = Autoencoder(input_shape)
option = st.sidebar.selectbox("Selectați analiza dorită:", ["Analiza anomalii", "Antrenare modele", "Analiză comparativă modele"])

if option == "Antrenare modele":
    model_selection = st.radio(
        "Alegeți modelul pe care doriți să-l antrenați:",
        ('Autoencoder', 'Isolation Forest')
    )
   
    if model_selection == 'Autoencoder':
        st.title("Detecție anomalii cu Autoencoder")
        optimizer = st.selectbox("Alegeți metoda de optimizare gradient:", ['adam', 'sgd', 'rmsprop'], index=0)
        loss_function = st.selectbox("Alegeți o funcție de cost:", ['mean_squared_error', 'binary_crossentropy'], index=0)
        epochs = st.number_input("Numărul de iterații (epochs):", min_value=1, value=100, step=1)
        validation_split = st.number_input("Proporție set testare (validare):", min_value=0.0, value=0.2, max_value=1.0, step=0.1)
        shuffle = st.checkbox("Doriți să amestecați instanțele în timpul antrenării?", True)

        if 'loss_history' not in st.session_state:
            st.session_state.loss_history = LossHistory()
        if 'anomalous_data_original' not in st.session_state:
            st.session_state.anomalous_data_original = None
        if 'mse_threshold' not in st.session_state:
            st.session_state.mse_threshold = None
        if 'num_anomalies' not in st.session_state:
            st.session_state.num_anomalies = None
        if 'training_plot' not in st.session_state:
            st.session_state.training_plot = None

        if st.button('Antrenați Autoencoder'):
            st.session_state['stop_training'] = False

            if st.button('Opriți antrenarea'):
                st.session_state['stop_training'] = True
            progress_bar = st.progress(0)
            progress_callback = ProgressCallback(progress_bar)
            stop_callback = StopTrainingCallback()
            loss_history = LossHistory()

            with st.spinner('Construire model Autoencoder...'):
                autoencoder.compile(optimizer=optimizer, loss=loss_function)
                autoencoder.fit(data_standardized, data_standardized,
                                epochs=epochs,
                                shuffle=shuffle,
                                validation_data=(data_standardized, data_standardized),
                                verbose=0,
                                validation_split=validation_split,
                                callbacks=[progress_callback, stop_callback, loss_history])
            
            st.session_state.loss_history = loss_history

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=loss_history.train_losses, mode='lines', name='Cost antrenament'))
            fig.add_trace(go.Scatter(y=loss_history.val_losses, mode='lines', name='Cost testare (validare)'))
            fig.update_layout(
                title='Cost antrenament și cost testare',
                xaxis_title='Iterații',
                yaxis_title='Cost',
                template='plotly_white'
            )
            st.session_state.training_plot = fig

            if not st.session_state.get('stop_training', False):
                st.success("Autoencoder-ul a fost antrenat cu succes!")
            
            reconstructed_data = autoencoder.predict(data_standardized)
            mse = np.mean(np.power(data_standardized - reconstructed_data, 2), axis=1)
            
            mse_threshold = np.percentile(mse, 90)
            st.session_state.mse_threshold = mse_threshold
            st.write(f"Prag MSE pentru detecție anomalii: {mse_threshold}")
            
            data['anomaly_autoencoder'] = [-1 if e > mse_threshold else 1 for e in mse]
            num_anomalies = np.sum(data['anomaly_autoencoder'] == -1)
            st.session_state.num_anomalies = num_anomalies
            st.write(f"Numărul de puncte de tip anomalii: {num_anomalies}")
            
            st.session_state.anomalous_data_original = data[data['anomaly_autoencoder'] == -1]
            if(st.session_state.data_anom is None):
                st.session_state.data_anom = data 
            else:
                st.session_state.data_anom['anomaly_autoencoder'] = data['anomaly_autoencoder'].copy()
        
        if st.session_state.training_plot:
            st.plotly_chart(st.session_state.training_plot, use_container_width=True)
        
        if st.session_state.mse_threshold:
            st.write(f"Prag MSE pentru detecție anomalii: {st.session_state.mse_threshold}")
        
        if st.session_state.num_anomalies:
            st.write(f"Numărul de puncte de tip anomalii: {st.session_state.num_anomalies}")
        
        if st.session_state.anomalous_data_original is not None:
            st.write("Numărul de puncte anomalii detectate:", st.session_state.anomalous_data_original)
            
    elif model_selection == "Isolation Forest":
        st.title("Detecție anomalii cu Isolation Forest")
        
        n_estimators = st.number_input("Numărul de estimatori: ", min_value=100, max_value=2000, step=100, value=100)
        max_features = st.number_input("Numărul de variabile: ", min_value=1, max_value=4, step=1, value=4)
        contamination = st.number_input("Factor contaminare:", min_value=0.0, value=0.1, step=0.1)
        n_jobs = st.number_input("Număr nuclee pentru antrenare: ", min_value=-1, max_value=8, step=1, value=-1)
        clf = None
        
        if st.button("Antrenează model IF"):
            clf = IsolationForest(n_estimators=n_estimators, max_features=max_features, contamination=contamination, 
                                n_jobs=n_jobs, random_state=42)
            
            anomalies = clf.fit_predict(data_standardized)
            data['anomaly_if'] = anomalies
            if st.session_state.data_anom is None:
                st.session_state.data_anom = data
            else:
                st.session_state.data_anom['anomaly_if'] = anomalies
            
            st.session_state.data_anom_if = st.session_state.data_anom.copy()
            
            st.session_state.num_anomalies_if = np.sum(st.session_state.data_anom['anomaly_if'] == -1)
            st.session_state.anomalous_data_if = st.session_state.data_anom[st.session_state.data_anom['anomaly_if'] == -1].copy()
            
            if 'anomaly_autoencoder' in st.session_state.anomalous_data_if.columns:
                st.session_state.anomalous_data_if.drop(columns=['anomaly_autoencoder'], inplace=True)
        
        if 'num_anomalies_if' in st.session_state:
            st.write(f"Numărul de anomalii detectate de IF: {st.session_state.num_anomalies_if}")
        
        if 'anomalous_data_if' in st.session_state:
            st.write("Anomaliile detectate de IF:", st.session_state.anomalous_data_if)
        
        if st.session_state.data_anom is not None:
            pass
        else:
            st.warning("Antrenați unul dintre modele pentru a vedea rezultatele analizei de anomalii.")

elif option == 'Analiza anomalii':
    if st.session_state.data_anom is None:
        st.warning("Antrenați modelele pentru vizualizare")
    else:
        analysis_selection = st.radio(
        "Alegeți tipul de vizualizare dorit:",
        ('Analiză parametri meteorologici cu anomalii evidențiate', 
         'Analiză longitudinală',
         'Analiză regională')
    )
        if analysis_selection == 'Analiză parametri meteorologici cu anomalii evidențiate':
            anomaly_model = st.selectbox(
                "Selectați modelul pentru detectarea anomaliilor:",
                ('Autoencoder', 'Isolation Forest')
            )
            parameter = st.selectbox(
                "Selectați parametrul pentru vizualizare:",
                ('air_pressure', 'pm10_quality', 'temperature', 'humidity')
            )
            year_range = st.slider(
                "Selectați intervalul de ani pentru vizualizare:",
                min_value=2009, max_value=2024, value=(2009, 2024)
            )

            data_filtered = st.session_state.data_anom[(st.session_state.data_anom['year'] >= year_range[0]) & 
                                                       (st.session_state.data_anom['year'] <= year_range[1])]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data_filtered['date'], y=data_filtered[parameter], mode='lines', name='Data', line=dict(color='blue')))

            if anomaly_model == 'Autoencoder':
                anomalies = data_filtered[data_filtered['anomaly_autoencoder'] == -1]
            else:
                anomalies = data_filtered[data_filtered['anomaly_if'] == -1]

            fig.add_trace(go.Scatter(x=anomalies['date'], 
                                     y=anomalies[parameter], 
                                     mode='markers', 
                                     name='Anomaly', marker=dict(color='red', size=6)))

            fig.update_layout(
                title=f"Analiza {parameter} în timp cu anomaliile evidențiate",
                xaxis_title="Data",
                yaxis_title=parameter,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)
        
        if analysis_selection == 'Analiză longitudinală':
            year_range = st.slider(
                "Selectați intervalul de ani pentru vizualizare:",
                min_value=2009, max_value=2024, value=(2009, 2024)
            )

            data = st.session_state.data_anom
            data['date'] = pd.to_datetime(data['date'])
            data['month'] = data['date'].dt.to_period('M').dt.to_timestamp()

            data_filtered = data[(data['date'].dt.year >= year_range[0]) & (data['date'].dt.year <= year_range[1])]

            key_phenomena = ['Heatwave', 'Storm', 'Flood']
            phenomena_counts = data_filtered.explode('phenomena').groupby(['month', 'phenomena']).size().unstack(fill_value=0)

            if_anomalies = data_filtered.groupby('month')['anomaly_if'].apply(lambda x: (x == -1).sum())
            ae_anomalies = data_filtered.groupby('month')['anomaly_autoencoder'].apply(lambda x: (x == -1).sum())

            fig = go.Figure()

            for phenomenon in key_phenomena:
                if phenomenon in phenomena_counts.columns:
                    fig.add_trace(go.Scatter(
                        x=phenomena_counts.index.astype(str),
                        y=phenomena_counts[phenomenon],
                        mode='lines+markers',
                        name=phenomenon.capitalize(),
                        line=dict(dash='solid'),
                    ))

            fig.add_trace(go.Scatter(
                x=ae_anomalies.index.astype(str),
                y=ae_anomalies,
                mode='lines+markers',
                name='AE Anomalies',
                line=dict(color='green', dash='dash'),
                yaxis='y2'
            ))

            fig.add_trace(go.Scatter(
                x=if_anomalies.index.astype(str),
                y=if_anomalies,
                mode='lines+markers',
                name='IF Anomalies',
                line=dict(color='red', dash='dot'),
                yaxis='y2'
            ))

            fig.update_layout(
                width=1200,  
                title='Frequența fenomenelor meteo cheie extreme și anomaliile detectate în timp',
                xaxis=dict(title='Timp (Lunar)'),
                yaxis=dict(title='Frecvența fenomenelor meteo'),
                yaxis2=dict(title='Numărul de anomalii', overlaying='y', side='right'),
                template='plotly_white',
                legend=dict(x=0, y=1, bgcolor='rgba(255, 255, 255, 0)')
            )

            st.plotly_chart(fig, use_container_width=True)
        if analysis_selection == 'Analiză regională':
            parameter = st.selectbox(
                "Selectați parametrul pentru vizualizare:",
                ('air_pressure', 'pm10_quality', 'temperature', 'humidity')
            )

            regions = st.session_state.data_anom['region'].unique()
            region_selection = st.selectbox(
                "Selectați regiunea pentru vizualizare:",
                regions
            )

            data = st.session_state.data_anom
            data['date'] = pd.to_datetime(data['date'])

            for region in regions:
                if region_selection == region:
                    region_data = data[data['region'] == region]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=region_data['date'],
                        y=region_data[parameter],
                        mode='lines',
                        name=parameter.capitalize(),
                        line=dict(color='teal'),
                    ))
                    ae_anomalies = region_data[region_data['anomaly_autoencoder'] == -1]
                    fig.add_trace(go.Scatter(
                        x=ae_anomalies['date'],
                        y=ae_anomalies[parameter],
                        mode='markers',
                        name='Anomalies (Autoencoder)',
                        marker=dict(color='red', symbol='x', size=8),
                    ))

                    if_anomalies = region_data[region_data['anomaly_if'] == -1]
                    fig.add_trace(go.Scatter(
                        x=if_anomalies['date'],
                        y=if_anomalies[parameter],
                        mode='markers',
                        name='Anomalies (Isolation Forest)',
                        marker=dict(color='blue', symbol='x', size=8),
                    ))

                    fig.update_layout(
                        title=f"{parameter.capitalize()} în {region} în timp cu anomaliile evidențiate",
                        xaxis=dict(title='Date'),
                        yaxis=dict(title=parameter.capitalize()),
                        template='plotly_white',
                        legend=dict(x=0, y=1, bgcolor='rgba(255, 255, 255, 0)')
                    )

                    st.plotly_chart(fig, use_container_width=True)

elif option == 'Analiză comparativă modele':
    comparison_option = st.radio(
                "Selectați rezultatul dorit pentru afișare:",
                ('Statistici obținute în urma antrenării', 
                 'Analiză comparativă - clusterizare')
            )
    if comparison_option == 'Statistici obținute în urma antrenării':
      
        data = st.session_state.data_anom

        total_instances = len(data)
        agree_both = len(data[(data['anomaly_autoencoder'] == -1) & (data['anomaly_if'] == -1)]) + len(data[(data['anomaly_autoencoder'] == 1) & (data['anomaly_if'] == 1)])
        disagree = len(data[(data['anomaly_autoencoder'] == -1) & (data['anomaly_if'] == 1)]) + len(data[(data['anomaly_autoencoder'] == 1) & (data['anomaly_if'] == -1)])
        agree_anomalies = len(data[(data['anomaly_autoencoder'] == -1) & (data['anomaly_if'] == -1)])
        agree_normals = len(data[(data['anomaly_autoencoder'] == 1) & (data['anomaly_if'] == 1)])

        st.write("## Concordanța între Autoencoder și Isolation Forest")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total instanțe", total_instances)
        col2.metric("Instanțe unde ambele metode sunt de acord", agree_both)
        col3.metric("Instanțe unde metodele nu sunt de acord", disagree)
        col1, col2 = st.columns(2)
        col1.metric("Instanțe unde ambele metode sunt de acord asupra punctelor normale", agree_normals)
        col2.metric("Instanțe unde ambele metode sunt de acord asupra anomaliilor", agree_anomalies)

        total_anomalies_ae = len(data[data['anomaly_autoencoder'] == -1])
        total_anomalies_if = len(data[data['anomaly_if'] == -1])
        unique_anomalies_ae = len(data[(data['anomaly_autoencoder'] == -1) & (data['anomaly_if'] != -1)])
        unique_anomalies_if = len(data[(data['anomaly_if'] == -1) & (data['anomaly_autoencoder'] != -1)])
        common_anomalies = len(data[(data['anomaly_autoencoder'] == -1) & (data['anomaly_if'] == -1)])

        st.write("## Statisticile anomaliilor")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total anomalii detectate de Autoencoder", total_anomalies_ae)
        col2.metric("Total anomalii detectate de Isolation Forest", total_anomalies_if)
        col3.metric("Anomalii unice detectate de Autoencoder", unique_anomalies_ae)
        col1, col2 = st.columns(2)
        col1.metric("Anomalii unice detectate de Isolation Forest", unique_anomalies_if)

    elif comparison_option == 'Analiză comparativă - clusterizare':
        st.title("Analiză comparativă - clusterizare")

        n_clusters = st.slider("Selectați numărul de clustere:", min_value=2, max_value=10, value=3)
        init_method = st.selectbox("Metoda de inițializare:", options=['k-means++', 'random'])
        max_iter = st.number_input("Numărul maxim de iterații:", min_value=100, max_value=1000, step=50, value=300)
        random_state = st.number_input("Random state:", min_value=0, value=42)

        data = st.session_state.data_anom.copy()
        numerical_features = ['pm10_quality', 'temperature', 'air_pressure', 'humidity']


        if st.button('Efectuează clusterizarea'):
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data[numerical_features])

            kmeans = KMeans(n_clusters=n_clusters, init=init_method, max_iter=max_iter, random_state=random_state)
            data['cluster'] = kmeans.fit_predict(data_scaled)
            st.session_state.data_anom = data

            cluster_counts = data['cluster'].value_counts().sort_index()
            cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
            cluster_centers_df = pd.DataFrame(cluster_centers, columns=numerical_features)

            anomaly_distribution = []
            for cluster in range(n_clusters):
                cluster_data = data[data['cluster'] == cluster]
                ae_anomalies = len(cluster_data[cluster_data['anomaly_autoencoder'] == -1])
                if_anomalies = len(cluster_data[cluster_data['anomaly_if'] == -1])
                total_in_cluster = len(cluster_data)
                ae_anomaly_rate = f"{(ae_anomalies / total_in_cluster) * 100:.2f}%"
                if_anomaly_rate = f"{(if_anomalies / total_in_cluster) * 100:.2f}%"
                anomaly_distribution.append([cluster, ae_anomalies, if_anomalies, total_in_cluster, ae_anomaly_rate, if_anomaly_rate])

            anomaly_distribution_df = pd.DataFrame(anomaly_distribution, columns=['Cluster', 'Anomalii AE', 
                                                                                  'Anomalii IF', 'Total în cluster', 'Rată anomalii AE', 
                                                                                  'Rată anomalii IF'])

            st.session_state.cluster_counts = cluster_counts
            st.session_state.cluster_centers_df = cluster_centers_df
            st.session_state.anomaly_distribution_df = anomaly_distribution_df

        if 'cluster_counts' in st.session_state:
            st.subheader("Statistici clustere")
            st.write(st.session_state.cluster_counts)
            st.subheader("Centrii clusteri")
            st.write(st.session_state.cluster_centers_df)
            st.subheader("Distribuția anomaliilor detectate")
            st.write(st.session_state.anomaly_distribution_df)

        if 'cluster' not in data.columns:
            st.warning("Apăsați butonul 'Efectuează clusterizarea' pentru a începe analiza.")