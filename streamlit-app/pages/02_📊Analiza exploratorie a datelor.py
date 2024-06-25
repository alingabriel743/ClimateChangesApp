import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from io import BytesIO
from plotly.subplots import make_subplots
import ast
from pymongo import MongoClient
from scipy.stats import kendalltau, chi2_contingency, spearmanr



# client = MongoClient("mongodb://localhost:27017/")
# db = client['climate_changes']
# date = db['parameters'] 
# documents = date.find()
# data = pd.DataFrame(list(documents))
data = pd.read_csv('data/set_tradus.csv')

@st.cache_data
def load_data():
    data = pd.read_csv('data/set_tradus.csv')
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['phenomena'] = data['phenomena'].apply(lambda x: ast.literal_eval(x))
    data = data.explode('phenomena')
    return data

data = load_data()

# Determine the season based on the month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Iarnă'
    elif month in [3, 4, 5]:
        return 'Primăvară'
    elif month in [6, 7, 8]:
        return 'Vară'
    elif month in [9, 10, 11]:
        return 'Toamnă'

data['season'] = data['month'].apply(get_season)


with st.sidebar:
    st.title("Tipuri de analiză")
    analysis_type = st.radio(
        "Alege tipul de analiză:",
        ('Statistici descriptive', 'Vizualizare date', 'Analiza corelațiilor', 'Teste statistice')
    )

# Titlu principal
st.title('Analiza exploratorie a datelor')

# Logica pentru afișarea conținutului pe baza selecției din sidebar
if analysis_type == 'Statistici descriptive':
  
    year_options = st.multiselect(
        'Alege anii (selectează „Toți anii” pentru a include toți anii):', 
        ['Toți anii'] + sorted(data['year'].unique(), reverse=True)
    )
    month_options = st.multiselect(
        'Alege lunile (selectează „Toate lunile” pentru a include toate lunile):', 
        ['Toate lunile'] + sorted(data['month'].unique())
    )

    # Verificăm dacă utilizatorul a selectat opțiunea "Toți Anii" sau "Toate Lunile"
    if 'Toți anii' in year_options:
        year_options = sorted(data['year'].unique(), reverse=True)
    if 'Toate lunile' in month_options:
        month_options = sorted(data['month'].unique())

    # Filtrarea datelor pe baza selecției de ani și luni
    data_filtered = data[data['year'].isin(year_options) & data['month'].isin(month_options)]

    region_selection = st.radio(
        "Alege opțiunea pentru regiuni:",
        ('O singură regiune', 'Toate regiunile')
    )
    if region_selection == 'O singură regiune':
        option_region = st.selectbox('Alege regiunea:', data['region'].unique())
        data_filtered = data_filtered[data_filtered['region'] == option_region]

    param_options = ['pm10_quality', 'air_pressure', 'temperature', 'humidity']
    option_parameter = st.multiselect(
        'Alege parametrii (selectează „Toți Parametrii” pentru a include toți parametrii):',
        ['Toți parametrii'] + param_options,
        default=['Toți parametrii']
    )

    # Dacă "Toți Parametrii" este selectat, include toți parametrii
    if 'Toți parametrii' in option_parameter:
        option_parameter = param_options

    # Crearea unui DataFrame pentru afișarea statisticilor
    if not option_parameter:
        st.write("Selectează cel puțin un parametru pentru a vizualiza statisticile.")
    elif not data_filtered.empty:
        stats_df = pd.DataFrame()
        for param in option_parameter:
            stats = data_filtered[param].describe().rename(param)
            stats_df = pd.concat([stats_df, stats], axis=1)
        st.write("Statistici descriptive pentru selecția ta:")
        st.dataframe(stats_df)
        
         # Descărcarea datelor ca CSV
        csv = stats_df.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="Descarcă datele ca CSV",
            data=csv,
            file_name='raport_statistici.csv',
            mime='text/csv'
        )

        # Crearea buffer-ului pentru Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            stats_df.to_excel(writer, index=True)
        excel_data = output.getvalue()
        st.download_button(
            label="Descarcă datele ca Excel",
            data=excel_data,
            file_name='raport_statistici.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        st.write("Nu există date disponibile pentru selecția făcută.")

elif analysis_type == 'Vizualizare date':
    st.title('Vizualizare date')

    view_option = st.radio(
        "Selectează tipul de vizualizare:",
        ('Analiză fenomene meteorologice', 'Analiză parametrii standard', 'Box plot-uri regionale')
    )

    if view_option == 'Analiză parametrii standard':
        region = st.selectbox('Alege regiunea:', ['Toate regiunile'] + list(data['region'].unique()))
        parameter = st.selectbox('Alege parametrul:', ['pm10_quality', 'air_pressure', 'temperature', 'humidity'])
        graph_type = st.selectbox("Alege tipul de grafic:", ["Evoluție în timp", "Distribuție", "Box Plot"])

        if region == 'Toate regiunile':
            region_data = data
        else:
            region_data = data[data['region'] == region]

        if region_data.empty or not parameter:
            st.write("Selectați o regiune și un parametru pentru a vizualiza graficul.")
        else:
            region_data = region_data.sort_values(by='date')

            if graph_type == "Evoluție în timp":
                fig = px.line(region_data, x='date', y=parameter, title=f'Evoluția {parameter} în timp')
                st.plotly_chart(fig, use_container_width=True)
            elif graph_type == "Distribuție":
                fig = px.histogram(region_data, x=parameter, nbins=30, title=f'Distribuția {parameter}')
                st.plotly_chart(fig, use_container_width=True)
            elif graph_type == "Box Plot":
                fig = px.box(region_data, y=parameter, title=f'Distribuția {parameter} (Box Plot)')
                st.plotly_chart(fig, use_container_width=True)

    elif view_option == 'Analiză fenomene meteorologice':
        selected_season = st.selectbox("Selectați anotimpul pentru analiză:", ['Iarnă', 'Primăvară', 'Vară', 'Toamnă'])
        selected_region = st.selectbox('Alege regiunea:', ['Toate regiunile'] + list(data['region'].unique()))

        if not selected_season or not selected_region:
            st.write("Selectați un anotimp și o regiune pentru a vizualiza graficul.")
        else:
            def plot_phenomena_by_season_region(season, region):
                if region == 'Toate regiunile':
                    season_data = data[data['season'] == season]
                else:
                    season_data = data[(data['season'] == season) & (data['region'] == region)]

                if season_data.empty:
                    st.write("Nu există date disponibile pentru acest sezon și această regiune.")
                    return go.Figure()

                total_occurrences = season_data.groupby('year').size().reset_index(name='total_counts')

                phenomena_counts = season_data.groupby(['year', 'phenomena']).size().reset_index(name='counts')
                phenomena_counts = phenomena_counts.merge(total_occurrences, on='year')
                phenomena_counts['percentage'] = (phenomena_counts['counts'] / phenomena_counts['total_counts']) * 100

                top_phenomena = phenomena_counts.groupby('phenomena')['counts'].sum().nlargest(5).index
                fig = go.Figure()
                for phenomenon in top_phenomena:
                    phen_data = phenomena_counts[phenomena_counts['phenomena'] == phenomenon]
                    fig.add_trace(go.Scatter(x=phen_data['year'], y=phen_data['percentage'], mode='lines+markers', name=phenomenon))

                fig.update_layout(
                    title=f'Top 5 cele mai frecvente fenomene, anotimp {season}, pe ani ({region})',
                    xaxis_title='Anul',
                    yaxis_title='Procent',
                    legend_title='Fenomen'
                )
                return fig

            fig = plot_phenomena_by_season_region(selected_season, selected_region)
            st.plotly_chart(fig, use_container_width=True)

    elif view_option == 'Box plot-uri regionale':
        selected_regions = st.multiselect(
            'Alege regiunile:',
            options=['Toate regiunile'] + list(data['region'].unique()),
            default=['Toate regiunile']
        )

        if 'Toate regiunile' in selected_regions:
            selected_data = data
        else:
            selected_data = data[data['region'].isin(selected_regions)]

        if selected_data.empty:
            st.write("Selectați cel puțin o regiune pentru a vizualiza graficul.")
        else:
            def plot_regional_box_plots(regional_data):
                fig = make_subplots(rows=2, cols=2, subplot_titles=(
                    'Distribuția regională a particulelor PM 10',
                    'Distribuția regională a presiunii aerului',
                    'Distribuția regională a temperaturii aerului',
                    'Distribuția regională a umidității relative'
                ))

                fig.add_trace(px.box(regional_data, x='region', y='pm10_quality').data[0], row=1, col=1)
                fig.update_xaxes(tickangle=45, row=1, col=1)

                fig.add_trace(px.box(regional_data, x='region', y='air_pressure').data[0], row=1, col=2)
                fig.update_xaxes(tickangle=45, row=1, col=2)

                fig.add_trace(px.box(regional_data, x='region', y='temperature').data[0], row=2, col=1)
                fig.update_xaxes(tickangle=45, row=2, col=1)

                fig.add_trace(px.box(regional_data, x='region', y='humidity').data[0], row=2, col=2)
                fig.update_xaxes(tickangle=45, row=2, col=2)

                fig.update_layout(
                    title='Distribuția regională a parametrilor meteorologici',
                    height=800,
                    showlegend=False
                )
                return fig

            fig = plot_regional_box_plots(selected_data)
            st.plotly_chart(fig, use_container_width=True)


elif analysis_type == 'Analiza corelațiilor':
    st.subheader('Analiza corelațiilor')

    st.subheader("Explicație metode de corelație")
    st.write("""
        Metoda Pearson măsoară relația liniară dintre două variabile. Este sensibilă la valori extreme și presupune
        că datele sunt distribuite normal.

        Metoda Spearman măsoară relația monotonă dintre două variabile. Este mai robustă la valori extreme și nu
        presupune că datele sunt distribuite normal. Se bazează pe ranguri, deci este utilă atunci când datele
        nu respectă distribuția normală sau când relațiile dintre variabile nu sunt liniare.

        Alegeți metoda Pearson pentru a identifica relații liniare puternice și metoda Spearman pentru a detecta
        relații monotone și a reduce influența valorilor extreme.
    """)
    correlation_method = st.radio(
        "Alege metoda de corelație:",
        ('Pearson', 'Spearman')
    )
    parameters = ['pm10_quality', 'air_pressure', 'temperature', 'humidity']
    correlation_data = data[parameters]

    if correlation_data.empty:
        st.write("Nu există date disponibile pentru corelații.")
    else:
        
        if correlation_method == 'Pearson':
            corr_matrix = correlation_data.corr(method='pearson')
        else:
            corr_matrix = correlation_data.corr(method='spearman')

        fig = px.imshow(corr_matrix, 
                        text_auto=True, 
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        title=f'Matrice de corelație metoda {correlation_method}',
                        aspect="auto")
        fig.update_layout(
            width=800, 
            height=600, 
            title={
                'text': f'Matrice de corelație metoda {correlation_method}',
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis_title='Parametri',
            yaxis_title='Parametri'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Interpretarea coeficienților de corelație")
        for i in range(len(parameters)):
            for j in range(i + 1, len(parameters)):
                var1 = parameters[i]
                var2 = parameters[j]
                corr_value = corr_matrix.loc[var1, var2]
                if abs(corr_value) > 0.7:
                    nature = "puternică"
                elif abs(corr_value) > 0.3:
                    nature = "moderată"
                else:
                    nature = "slabă"
                st.write(f"Corelația între {var1} și {var2} este de natură {nature} ({corr_value:.2f}).")

    
elif analysis_type == 'Teste statistice':
    st.subheader('Teste statistice pentru fenomenele meteo extreme')

    phenomena_map = {
        'Caniculă': 'Heatwave',
        'Furtună': 'Storm',
        'Inundație': 'Flood',
        'Zăpadă': 'Snow'
    }


    test_type = st.selectbox(
        "Alege tipul de test statistic:",
        ['Testul Mann-Kendall', 'Testul Chi-Square pentru independență', 'Corelația rangurilor Spearman']
    )
    year_range = st.slider(
        'Selectează intervalul de ani:',
        int(data['year'].min()), int(data['year'].max()), 
        (int(data['year'].min()), int(data['year'].max()))
    )

    filtered_data = data[(data['year'] >= year_range[0]) & (data['year'] <= year_range[1])]
    st.write("Cele mai importante 4 fenomene meteo sunt: Caniculă, Furtună, Inundație, Zăpadă.")

    if test_type == 'Testul Mann-Kendall':
        st.write("""
            ### Testul Mann-Kendall
            Testul Mann-Kendall este utilizat pentru a detecta trendurile într-o serie temporală. 
            Un τ pozitiv indică un trend ascendent, în timp ce un τ negativ indică un trend descendent. 
            Un p-value mai mic decât 0.05 indică un trend semnificativ statistic.
        """)

        results_mk = []
        for phenomenon in phenomena_map.values():
            yearly_counts = filtered_data[filtered_data['phenomena'] == phenomenon].groupby('year').size()
            tau, p_value = kendalltau(yearly_counts.index, yearly_counts.values)
            results_mk.append((phenomenon, tau, p_value))
            st.write(f"Trendul fenomenului {phenomenon}: τ = {tau:.3f}, p-value = {p_value:.3f}")

        for phenomenon, tau, p_value in results_mk:
            if p_value < 0.05:
                trend = "semnificativ"
            else:
                trend = "nesemnificativ"
            st.write(f"Fenomenul {phenomenon} are un trend {trend} cu un τ de {tau:.3f} și un p-value de {p_value:.3f}.")

    elif test_type == 'Testul Chi-Square pentru independență':
        st.write("""
            ### Testul Chi-Square pentru independență
            Testul Chi-Square pentru Independență este utilizat pentru a determina dacă există o relație între două variabile categorice. 
            Un p-value mai mic decât 0.05 indică o relație semnificativă statistic între variabile.
        """)

        results_chi2 = []
        for phenomenon in phenomena_map.values():
            contingency_table = pd.crosstab(filtered_data[filtered_data['phenomena'] == phenomenon]['year'], filtered_data[filtered_data['phenomena'] == phenomenon]['season'])
            chi2, p, dof, ex = chi2_contingency(contingency_table)
            results_chi2.append((phenomenon, chi2, p))
            st.write(f"Relația dintre {phenomenon} și anotimp: Chi-Square = {chi2:.2f}, p-value = {p:.3f}")

        for phenomenon, chi2, p in results_chi2:
            if p < 0.05:
                significance = "semnificativă"
            else:
                significance = "nesemnificativă"
            st.write(f"Relația dintre {phenomenon} și anotimp este {significance} cu un Chi-Square de {chi2:.2f} și un p-value de {p:.3f}.")

    elif test_type == 'Corelația rangurilor Spearman':
        st.write("""
            ### Corelația rangurilor Spearman
            Corelația rangurilor Spearman este utilizată pentru a măsura forța și direcția asocierii monotone între două variabile. 
            Un ρ pozitiv indică o corelație pozitivă, în timp ce un ρ negativ indică o corelație negativă. 
            Un p-value mai mic decât 0.05 indică o corelație semnificativă statistic.
        """)

        # Allow user to select variables for Spearman correlation
        variable_x = st.selectbox('Selectează prima variabilă:', ['temperature', 'humidity', 'air_pressure'])
        variable_y = st.selectbox('Selectează a doua variabilă:', ['Caniculă', 'Furtună', 'Inundație', 'Zăpadă'])
        variable_y_mapped = phenomena_map[variable_y]

        # Perform Spearman Rank Correlation Test
        results_spearman = []
        occurrences = filtered_data[filtered_data['phenomena'] == variable_y_mapped].groupby('date').size().reindex(filtered_data['date'].unique(), fill_value=0)
        var_data = filtered_data[['date', variable_x]].drop_duplicates().set_index('date')
        merged_data = occurrences.to_frame(name='occurrences').join(var_data).dropna()
        rho, p_value = spearmanr(merged_data['occurrences'], merged_data[variable_x])
        results_spearman.append((variable_y, variable_x, rho, p_value))
        st.write(f"Corelația între {variable_y} și {variable_x}: ρ = {rho:.3f}, p-value = {p_value:.3f}")

        # Interpretation
        for phenomenon, var_x, rho, p_value in results_spearman:
            if p_value < 0.05:
                correlation = "semnificativă"
            else:
                correlation = "nesemnificativă"
            st.write(f"Corelația între {phenomenon} și {var_x} este {correlation} cu un ρ de {rho:.3f} și un p-value de {p_value:.3f}.")
