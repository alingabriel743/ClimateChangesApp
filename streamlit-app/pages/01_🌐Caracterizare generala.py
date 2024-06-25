import streamlit as st
from pymongo import MongoClient
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np


client = MongoClient("mongodb://localhost:27017/")
db = client['climate_changes']
date = db['parameters'] 


def display_map(df):
    map = folium.Map(location = [45.9442858, 25.0094303], zoom_start=6.4, tiles="CartoDB positron")
    map_regions_with_avg_temp = {}
    regions = list(df['region'].unique())
    group_regions = df.groupby('region').mean(numeric_only=True).reset_index()
    for region in regions:
        map_regions_with_avg_temp[region] = group_regions[group_regions['region'] == region]['temperature'].iloc[0]
    
    choropleth = folium.Choropleth(
        geo_data='data/regions_final.geojson',
        data = df,
        columns = ('region', 'temperature'),
        key_on='feature.properties.name',
        line_opacity=0.8,
        highlight=True
    )
    choropleth.geojson.add_to(map)

    for feature in choropleth.geojson.data['features']:
        region_name = feature['properties']['name']
        feature['properties']['mean_temp'] = 'Temperatura medie 2009-2024: ' + str(np.round(map_regions_with_avg_temp[region_name], 2)) + "°C"
        feature['properties']['general_air_pressure'] = 'Presiunea atmosferică medie 2009-2024: ' + str(np.round(group_regions[group_regions['region'] == region_name]['air_pressure'].iloc[0])) + " mbar"
        feature['properties']['general_air_quality'] = 'Calitatea aerului (medie) 2009-2024: ' + str(np.round(group_regions[group_regions['region'] == region_name]['pm10_quality'].iloc[0])) + " µg/m3"
        feature['properties']['general_humidity'] = 'Umiditate relativă (medie) 2009-2024: ' + str(np.round(group_regions[group_regions['region'] == region_name]['humidity'].iloc[0])) + "%"
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['name','mean_temp', 'general_air_pressure', 'general_air_quality', 'general_humidity'], labels=False)
    )
        
    st_map = st_folium(map, width = 700, height = 450)
    
def main():
    
    documents = date.find()
    df = pd.DataFrame(list(documents))
    st.title('Clima României: caracterizare generală')

    st.write("""
    România experimentează un climat temperat-continental, cu variații regionale specifice. Datele oficiale subliniază următoarele aspecte principale:

    ### Temperaturi
    - **Mediile de temperatură**: În ianuarie, temperaturile medii variază între -3°C în nordul țării și 0°C în sud. În iulie, acestea variază de la 19°C în zonele montane la aproximativ 24°C în regiunile sudice și estice.
    - **Extreme de temperatură**: România a înregistrat temperaturi de peste 40°C în timpul verilor caniculare, în timp ce iernile pot aduce minime de sub -20°C în Carpați.

    ### Precipitații
    - **Cantități anuale medii**: Cantitatea anuală de precipitații este de aproximativ 637 mm, dar variază considerabil de la regiune la regiune, cu mai multe precipitații în zonele montane (peste 1000 mm) și mai puține în Dobrogea (sub 400 mm).

    ### Fenomene meteorologice extreme
    - România este afectată ocazional de fenomene meteorologice extreme, inclusiv furtuni puternice, inundații și secete, fenomene ce tind să se intensifice în contextul schimbărilor climatice.

    ## Tendințe climatice (2009-2024)
    Analizele recente indică o tendință clară de încălzire:
    - **Încălzirea medie**: În ultimii 15 ani, temperaturile medii anuale au crescut cu aproximativ 0.5°C, o tendință care se aliniază cu încălzirea globală observată.
    - **Impactul asupra anotimpurilor**: Verile devin mai calde și mai lungi, în timp ce iernile sunt mai scurte și mai blânde, afectând biodiversitatea și agricultura.

    ## Implicații
    Schimbările climatice necesită adaptări strategice în gestionarea resurselor hidrice, agricultură și protecția mediului, subliniind importanța politicilor de mediu și a inițiativelor de sustenabilitate.
    """)

    display_map(df)

if __name__ == "__main__":
    main()
