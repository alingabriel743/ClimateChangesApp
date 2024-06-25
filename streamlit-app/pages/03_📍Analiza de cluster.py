import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from kmodes.kprototypes import KPrototypes
import ast
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
st.set_page_config(layout="centered")
# Load and prepare data
@st.cache_data()
def load_data():
    data = pd.read_csv('data/set_tradus.csv')  # Adjust path as needed
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['phenomena'] = data['phenomena'].apply(lambda x: ast.literal_eval(x))
    data = data.explode('phenomena')
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


all_numerical_features = ['pm10_quality', 'temperature', 'air_pressure', 'humidity']
data = load_data()
data['season'] = data['month'].apply(get_season)
method = st.sidebar.radio("Selectați metoda de clusterizare / tipul de clusteri de analizat grafic:", ["K-Prototypes", "K-Means", 'Clusteri K-Prototypes', 'Clusteri K-Means'])

def preprocess_data(data, features, method):
        if method == "Standardizează":
            scaler = StandardScaler()
            data[features] = scaler.fit_transform(data[features])
        elif method == "Normalizează":
            scaler = MinMaxScaler()
            data[features] = scaler.fit_transform(data[features])
        return data

if method == 'K-Prototypes':
    st.title('Analiza de cluster - metoda K-Prototypes')
    st.header("Alegerea variabilelor")
    all_numerical_features = ['pm10_quality', 'temperature', 'air_pressure', 'humidity']
    categorical_features = st.multiselect('Selectați variabilele categoriale:', options=data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist(), default = 'phenomena')
    numerical_features = st.multiselect('Selectați variabilele numerice:', options=all_numerical_features, default=all_numerical_features)

    st.session_state.preprocessing_method = None
 
    if 'data' not in st.session_state or st.session_state.preprocessing_method == 'None':
        st.session_state.data = data.reset_index(drop=True).copy()
        st.session_state.preprocessing_method = 'None'

    st.header("Preprocesarea datelor")
    preprocessing_option = st.selectbox("Selectați metoda de preprocesare:", ["None", "Standardizează", "Normalizează"], key="preprocessing_select")

    if preprocessing_option != "None":
        numerical_features = st.multiselect('Selectați variabilele numerice:', options=st.session_state.data.select_dtypes(include=['number']).columns.tolist(), key=f"num_features_{preprocessing_option}")
        if numerical_features:
            st.session_state.data = preprocess_data(st.session_state.data.copy(), numerical_features, preprocessing_option)
    else:
        st.write("Nu s-a selectat nicio preprocesare.")


    if preprocessing_option != st.session_state.preprocessing_method and numerical_features:
        st.session_state.data = preprocess_data(st.session_state.data.copy(), numerical_features, preprocessing_option)
        st.session_state.preprocessing_method = preprocessing_option


    if categorical_features and numerical_features:
        features_to_process = categorical_features + numerical_features + ['month']
        kproto_data = st.session_state.data[features_to_process].dropna()
        kproto_data[categorical_features] = kproto_data[categorical_features].apply(lambda x: x.astype('category').cat.codes)

        st.header("Configurare metoda Elbow")
        k_min = st.number_input('Numărul minim de clusteri (Min K) pentru metoda Elbow:', min_value=2, max_value=10, value=2)
        k_max = st.number_input('Numărul maxim de clusteri (Max K) pentru metoda Elbow:', min_value=3, max_value=20, value=10)
        n_init_elbow = st.number_input('Numărul de inițializări (n_init) pentru metoda Elbow:', min_value=1, max_value=10, value=1)
        max_iter_elbow = st.number_input('Numărul maxim de iterații (max_iter) pentru metoda Elbow:', min_value=5, max_value=100, value=10)
        init_method_elbow = st.selectbox('Metoda de inițializare (init)', ['Huang', 'Cao'])
        if st.button('Generază grafic Elbow'):
            distortions = []
            print(distortions)
            total_steps = k_max - k_min + 1
            progress_bar = st.progress(0)
            for i, k in enumerate(range(k_min, k_max + 1), 1):
                kproto = KPrototypes(n_clusters=k, init=init_method_elbow, n_init=n_init_elbow, max_iter=max_iter_elbow, verbose=0)
                kproto.fit_predict(kproto_data, categorical=[kproto_data.columns.get_loc(col) for col in categorical_features])
                distortions.append(kproto.cost_)
                progress_bar.progress(i / total_steps)

            fig = go.Figure(data=go.Scatter(x=list(range(k_min, k_max + 1)), y=distortions, mode='lines+markers'))
            fig.update_layout(title='Graficul Elbow pentru alegerea numărului optim K (K-Prototypes)', xaxis_title='Număr de clusteri', yaxis_title='Cost')
            st.plotly_chart(fig, use_container_width=True)
            progress_bar.empty()  

        st.header("Construire model K-Prototypes")
        chosen_k = st.number_input('Numărul de clusteri (K) pentru model:', min_value=2, max_value=20, value=5)
        n_init_model = st.number_input('Numărul de inițializări (n_init) pentru model:', min_value=1, max_value=10, value=1)
        max_iter_model = st.number_input('Numărul maxim de iterații (max_iter) pentru model:', min_value=5, max_value=100, value=10)
        init_method_model = st.selectbox('Metoda de inițializare (init) pentru model:', ['Huang', 'Cao'])

        # kproto_data['original_phenomena'] = st.session_state.data['phenomena'].reset_index(drop=True).copy()
        kproto_data['phenomena'] = kproto_data['phenomena'].astype('category').cat.codes
        # st.write(kproto_data.columns.get_loc(col) for col in categorical_features)
        st.session_state.data.reset_index(inplace=True, drop=True)
        # st.write(st.session_state.data)
        if st.button('Construiți modelul'):
            with st.spinner('Construire model K-Prototypes...'):
                kproto = KPrototypes(n_clusters=chosen_k, init=init_method_model, n_init=n_init_model, max_iter=max_iter_model, verbose=1, n_jobs=-1)
                clusters = kproto.fit_predict(kproto_data, categorical=[kproto_data.columns.get_loc(col) for col in categorical_features])
                kproto_data['Cluster'] = clusters
                kproto_data.drop('phenomena', axis = 1, inplace=True)
                kproto_data = kproto_data.reset_index(drop=True) 
                kproto_data['phenomena'] = st.session_state.data['phenomena'].copy()
                st.session_state['kproto_data'] = kproto_data.copy()
                st.write("Distribuție clusteri:", kproto_data['Cluster'].value_counts())
                st.session_state['show_pca_proto'] = False 
                st.session_state['pca_result_proto'] = PCA(n_components=3).fit_transform(kproto_data[numerical_features])
                st.session_state['clusters_proto'] = kproto_data['Cluster'].astype(str)
                st.session_state['show_pca_proto'] = True 
        if st.session_state.get('show_pca_proto', False):
            pca_type = st.radio("Alegeți dimensionalitatea componentelor principale:", ("2D PCA", "3D PCA"))
            if pca_type == "2D PCA":
                fig = px.scatter(x=st.session_state['pca_result_proto'][:, 0], y=st.session_state['pca_result_proto'][:, 1], color=st.session_state['clusters_proto'])
                fig.update_layout(title="Vizualizare 2D în componente principale a clusterilor")
                st.plotly_chart(fig, use_container_width=True)
            elif pca_type == "3D PCA":
                fig = px.scatter_3d(x=st.session_state['pca_result_proto'][:, 0], y=st.session_state['pca_result_proto'][:, 1], z=st.session_state['pca_result_proto'][:, 2], color=st.session_state['clusters_proto'])
                fig.update_layout(title="Vizualizare 3D în componente principale a clusterilor")
                st.plotly_chart(fig, use_container_width=True)

elif method == 'K-Means':
    st.header("Analiza de cluster - K Means")

    st.header("Alegerea variabilelor")
    all_numerical_features = ['pm10_quality', 'temperature', 'air_pressure', 'humidity']
    numerical_features = st.multiselect('Selectați variabilele numerice:', options=all_numerical_features,  default=all_numerical_features)


    if 'data' not in st.session_state or st.session_state.preprocessing_method == 'None':
        st.session_state.data = data.copy()
        st.session_state.preprocessing_method = 'None'

    st.header("Preprocesarea datelor")
    preprocessing_option = st.selectbox("Selectați metoda de preprocesare:", ["None", "Standardizează", "Normalizează"], key="preprocessing_select")
    if preprocessing_option != "None":
        numerical_features = st.multiselect('Selectați variabilele:', options=st.session_state.data.select_dtypes(include=['number']).columns.tolist(), key=f"num_features_{preprocessing_option}")
        if numerical_features:
            st.session_state.data = preprocess_data(st.session_state.data.copy(), numerical_features, preprocessing_option)
    else:
        st.write("Nu s-a selectat nicio preprocesare.")
    
    if preprocessing_option != st.session_state.preprocessing_method and numerical_features:
        st.session_state.data = preprocess_data(st.session_state.data.copy(), numerical_features, preprocessing_option)
        st.session_state.preprocessing_method = preprocessing_option

    if numerical_features:
        kmeans_data = st.session_state.data[numerical_features].dropna()

        st.header("Configurarea metodei Elbow")
        k_min = st.number_input('Numărul minim de clusteri (Min K) pentru metoda Elbow:', min_value=2, max_value=10, value=2)
        k_max = st.number_input('Numărul maxim de clusteri (Max K) pentru metoda Elbow:', min_value=3, max_value=20, value=10)
        init_method = st.selectbox("Metoda de inițializare: ", ['k-means++', 'random']) 
        n_init_elbow = st.number_input('Numărul de inițializări (n_init):', min_value=1, max_value=10, value=1)
        max_iter_elbow = st.number_input('Numărul maxim de iterații:', min_value=5, max_value=100, value=10)
        algorithm_elbow = st.selectbox("Algoritmul utilizat: ", ['lloyd', 'elkan'])
        if st.button('Generați grafic Elbow'):
            distortions = []
            total_steps = k_max - k_min + 1
            progress_bar = st.progress(0)
            for i, k in enumerate(range(k_min, k_max + 1), 1):
                kmeans = KMeans(n_clusters=k, init = init_method, n_init = n_init_elbow, max_iter=max_iter_elbow, algorithm=algorithm_elbow)
                kmeans.fit_predict(kmeans_data)
                distortions.append(kmeans.inertia_)
                progress_bar.progress(i / total_steps)

            fig = go.Figure(data=go.Scatter(x=list(range(k_min, k_max + 1)), y=distortions, mode='lines+markers'))
            fig.update_layout(title='Graficul Elbow pentru alegerea numărului optim K (K-Means)', xaxis_title='Numărul de clusteri', yaxis_title='Cost')
            st.plotly_chart(fig, use_container_width=True)
            progress_bar.empty() 

        st.header("Construiți model K-Means")
        chosen_k = st.number_input('Numărul de clusteri (K) pentru model:', min_value=2, max_value=20, value=5)
        init_method_model = st.selectbox("Metoda de inițializare: ", ['k-means++', 'random'], key="b") 
        n_init_model = st.number_input('Numărul de inițializări (n_init) pentru:', min_value=1, max_value=10, value=1)
        max_iter_model = st.number_input('Numărul maxim de iterații (max_iter) pentru model:', min_value=5, max_value=100, value=10)
        algorithm_model = st.selectbox("Algoritmul utilizat: ", ['lloyd', 'elkan'], key="a")

        if st.button('Construiți model K-Means'):
            with st.spinner('Construire model K-Means...'):
                kmeans = KMeans(n_clusters=chosen_k, init = init_method_model, n_init=n_init_model, max_iter=max_iter_model, algorithm=algorithm_model)
                clusters = kmeans.fit_predict(kmeans_data)
                kmeans_data['Cluster'] = clusters
                st.session_state['kmeans_data'] = kmeans_data.copy()
                st.write("Repartiție clusteri:", kmeans_data['Cluster'].value_counts())
                st.session_state['show_pca_means'] = False  
                st.session_state['pca_result_means'] = PCA(n_components=3).fit_transform(kmeans_data[numerical_features])
                st.session_state['clusters_means'] = kmeans_data['Cluster'].astype(str)
                st.session_state['show_pca_means'] = True 


        if st.session_state.get('show_pca_means', False):
            pca_type = st.radio("Selectați dimensionalitatea:", ("2D PCA", "3D PCA"))
            if pca_type == "2D PCA":
                fig = px.scatter(x=st.session_state['pca_result_means'][:, 0], y=st.session_state['pca_result_means'][:, 1], color=st.session_state['clusters_means'])
                fig.update_layout(title="Vizualizare 2D PCA")
                st.plotly_chart(fig, use_container_width=True)
            elif pca_type == "3D PCA":
                fig = px.scatter_3d(x=st.session_state['pca_result_means'][:, 0], y=st.session_state['pca_result_means'][:, 1], z=st.session_state['pca_result_means'][:, 2], color=st.session_state['clusters_means'])
                fig.update_layout(title="Vizualizare 3D PCA")
                st.plotly_chart(fig, use_container_width=True)

elif method == 'Clusteri K-Prototypes':
    st.title("Analiza cluster K-Prototypes")

    graph_type = st.radio("Selectați tipul de analiză dorită:",
                          ['Distribuție clusteri', 'Distribuție parametri', 'Fenomene pe anotimpuri','Apariție fenomene extreme'])

    if graph_type == 'Distribuție clusteri':
        cluster_counts = st.session_state.kproto_data['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Frequency']

        fig = px.bar(cluster_counts, x='Cluster', y='Frequency',
                     title="Distribuția instanțelor pe fiecare cluster", text='Frequency')
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',
                          xaxis_title="Cluster",
                          yaxis_title="Frequency",
                          xaxis_tickmode='linear')
        st.plotly_chart(fig, use_container_width=True)
        descriptive_stats = st.session_state.kproto_data.groupby('Cluster')[all_numerical_features].agg(['mean', 'max', 'min']).reset_index()
        descriptive_stats.columns = [' '.join(col).strip() for col in descriptive_stats.columns.values]
        st.subheader("Statistici descriptive per cluster")
        st.dataframe(descriptive_stats.style.format("{:.2f}"))

    elif graph_type == 'Distribuție parametri':
        selected_parameters = st.multiselect('Selectați parametri pentru graficele de distribuție:',
                                             options=all_numerical_features, default=None)
        for parameter in selected_parameters:
            st.subheader(f"Distribuția parametrului {parameter} în clusteri")
            fig = px.strip(st.session_state.kproto_data, y='Cluster', x=parameter, orientation='h', 
                        title=f"Distribuția parametrului {parameter} în clusteri",
                        labels={"Cluster": "Cluster", parameter: f"{parameter}"})
            fig.update_traces(marker={'size': 5})
            fig.update_layout(
                xaxis_title=f"{parameter} Valoare",
                yaxis_title="Cluster",
                yaxis=dict(categoryorder='total ascending')  
            )
            st.plotly_chart(fig, use_container_width=True)

    elif graph_type == 'Fenomene pe anotimpuri':
        if 'Season' not in st.session_state.kproto_data.columns:
            st.session_state.kproto_data['Season'] = st.session_state.kproto_data['month'].apply(get_season)

        selected_season = st.selectbox('Selectați anotimpul', ['Iarnă', 'Primăvară', 'Vară', 'Toamnă'])
        season_data = st.session_state.kproto_data[st.session_state.kproto_data['Season'] == selected_season]

        grouped_data = season_data.groupby(['Cluster', 'phenomena']).size().reset_index(name='Count')
        grouped_data = grouped_data.sort_values(['Cluster', 'Count'], ascending=[True, False])

        for cluster in grouped_data['Cluster'].unique():
            cluster_data = grouped_data[grouped_data['Cluster'] == cluster]
            top_phenomena = cluster_data.nlargest(10, 'Count')

            fig = px.bar(top_phenomena, x='phenomena', y='Count', orientation='v', 
                        title=f'Top 10 cele mai frecvente fenomene în clusterul {cluster} în {selected_season}',
                        labels={'phenomena': 'Phenomena', 'Count': 'Frequency'})
            fig.update_layout(xaxis_title="Fenomene", yaxis_title="Frecvență", xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig, use_container_width=True)

    elif graph_type == 'Apariție fenomene extreme':
        extreme_phenomena = ['Hail', 'Whirlwind', 'Sleet', 'Heatwave', 'Storm', 'Flood', 'Blizzard', 'Drizzle', 'Freezing rain', 'Drought', 'Cyclone', 'Tornado', 'Fires']
        extreme_data = st.session_state.kproto_data[st.session_state.kproto_data['phenomena'].isin(extreme_phenomena)]
        grouped_extreme_data = extreme_data.groupby(['Cluster', 'phenomena']).size().reset_index(name='Count')

        fig = px.bar(grouped_extreme_data, x='phenomena', y='Count', color='Cluster', barmode='group',
                    title='Apariție fenomene extreme în fiecare cluster',
                    labels={'phenomena': 'Extreme Phenomena', 'Count': 'Frequency'})
        fig.update_layout(xaxis_title="Fenomene", yaxis_title="Frecvență", xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)

elif method == 'Clusteri K-Means':
    st.write(st.session_state['kmeans_data'])


        