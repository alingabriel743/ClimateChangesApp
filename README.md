<h1> Climate Changes Application for Master's Thesis </h1>
<p> This repository contains the source code for analyzing the climate changes in Romania between 2009-2024 timespan.</p>
<h2>Project Overview</h2>
<p> This project aims to explore the impact of climate change in Romania by analyzing meteorological and air quality data to detect trends and identify patterns in climate variation. </p>
<h2>Key Features</h3>
<ol>
  <li>Climate Data Analysis:</li>
  <ul>
    <li>Conducting Exploratory Data Analysis (EDA) to identify trends and patterns in meteorological data.</li>
    <li>Integrating structured meteorological data with unstructured data from weather news articles.</li>
  </ul>
  </br>
  <li>Machine Learning Models:</li>
  <ul>
    <li>Clustering using K-Means and K-Prototypes to group similar data points and identify patterns.</li>
    <li>Anomaly detection using Autoencoder and Isolation Forest models to identify extreme weather events and unusual patterns in the data.</li>
  </ul>
  </br>
  <li> Web Application: </li>
  <ul>
    <li>Developing a user-friendly web application using Streamlit to visualize data and analysis results.</li>
    <li>Providing interactive tools for data exploration, including statistical tests, correlation matrices, and various visualization options.</li>
  </ul>
</ol>
<h2>Data Sources</h3>
<ul>
  <li>Structured Data: Daily measurements of key meteorological parameters such as temperature, air pressure, humidity, and air quality (PM10).</li>
  <li>Unstructured Data: Weather-related news articles collected through web scraping techniques, processed to extract relevant meteorological phenomena.</li>
</ul>
<h2>Folder Structure</h3>
<p>
  <pre>
  MAIN/
  ├── climate/
  │   ├── data/
  │   │   ├── data_crawler/
  │   │   ├── data_tabular/
  │   │   └── datasets/
  │   ├── meteo_news_new/
  │   │   ├── meteo_news_updated.json
  │   │   ├── meteo_news.json
  │   │   └── scrapy.cfg
  │   ├── scripts_ds/
  │   │   ├── 1. data_preprocessing.ipynb
  │   │   ├── 2. data_concatenation.ipynb
  │   │   ├── 3. data_merge.ipynb
  │   │   ├── 4. data_transformation.ipynb
  │   │   └── 5. eda.ipynb
  │   ├── scripts_gpt/
  │   │   ├── 1. clean_data.ipynb
  │   │   └── 2. data_preprocessing.ipynb
  ├── streamlit-app/
  │   ├── data/
  │   │   ├── regions_final.geojson
  │   │   ├── romania-with-regions.geojson
  │   │   └── set_tradus.csv
  │   ├── pages/
  │   │   ├── 01_🌍Caracterizare_generala.py
  │   │   ├── 02_📊Analiza_exploratorie_a_datelor.py
  │   │   ├── 03_📍Analiza_de_cluster.py
  │   │   └── 04_📈Detectie_anomalii_meteorologice.py
  │   ├── autoencoder.py
  │   ├── Introducere.py
  │   ├── preprocess_geographic_regions.ipynb
  │   ├── utils.py
  │   └── validations.py
  </pre>
</p>
<h2>Files Description</h2>
<h3> a) Scripts for data gathering and preprocessing</h3>

<ol>
<li><code>scripts_ds</code>: Scripts related to data science tasks.</li>
  <ul>
    <li><code>1. data_preprocessing.ipynb</code>: Initial data cleaning and preprocessing steps.</li>
    <li><code>2. data_concatenation.ipynb</code>: Concatenate multiple data sources into a single cohesive dataset, allowing for comprehensive analysis.</li>
    <li><code>3. data_merge.ipynb</code>: Merge datasets from various origins, integrating structured meteorological data with unstructured weather report information.</li>
    <li><code>4. data_transformation.ipynb</code>: Data transformation into formats suitable for analysis, including normalization and scaling.</li>
    <li><code>5. eda.ipynb</code>: Conducts Exploratory Data Analysis (EDA) to identify patterns, trends, and insights within the data.</li>
  </ul>
<li><code>scripts_gpt</code>: Scripts utilizing GPT models for data processing.</li>
  <ul>
    <li><code>1. clean_data.ipynb</code>: Data cleaning using GPT models, including noise removal and format standardization.</li>
    <li><code>2. data_preprocessing.ipynb</code>: Further data preprocessing after initial cleaning, ensuring readiness for analysis and modeling.</li>
  </ul>
</ol>

<h3> Data </h3>

<ol>
<li><code>data</code>: Directories and files related to data storage and management.</li>
  <ul>
    <li><code>data_crawler/</code>: Contains scripts for web scraping to gather unstructured data from online sources such as weather reports and news articles.</li>
    <li><code>data_tabular/</code>: Stores processed tabular data, which includes cleaned and structured datasets ready for analysis.</li>
    <li><code>datasets/</code>: Consolidated datasets that have been merged and transformed, representing the final data used for comprehensive analysis.</li>
  </ul>
</ol>

<h3>b) Streamlit application </h3>

<h3>Utilized data</h3>

<ol>
<li><code>data</code>: Files used by the Streamlit application.</li>
  <ul>
    <li><code>regions_final.geojson</code>: GeoJSON file containing geographical data for the regions under study, used for spatial analysis and visualizations in the application.</li>
    <li><code>romania-with-regions.geojson</code>: Another GeoJSON file that includes additional regional details, enhancing the geographical context of the data.</li>
    <li><code>set_tradus.csv</code>: A CSV file with data used in the application, potentially including translated or preprocessed information relevant to the analysis.</li>
  </ul>
</ol>

<h3> Pages </h3>

<ol>
<li><code>pages</code>: Python scripts for different pages of the Streamlit application.</li>
  <ul>
    <li><code>01_🌍Caracterizare_generala.py</code>: General characterization of the data, including summary statistics and an introductory overview of the dataset.</li>
    <li><code>02_📊Analiza_exploratorie_a_datelor.py</code>: Conducts Exploratory Data Analysis (EDA), offering interactive visualizations and statistical summaries.</li>
    <li><code>03_📍Analiza_de_cluster.py</code>: Focuses on cluster analysis, implementing methods like K-Means and K-Prototypes to identify natural groupings within the data.</li>
    <li><code>04_⚠️Detectie_anomalii_meteorologice.py</code>: Detects meteorological anomalies using machine learning models such as Isolation Forest and Autoencoder.</li>
  </ul>
</ol>
</ol>

<h2>Getting started</h2>
<p>To get started with the project:</p>
<ol>
  <li>Clone the repository using Git: <code>git clone https://github.com/alingabriel743/climate-changes-app.git</code></li>
  <li>Install the required packages: <code>pip install -r requirements.txt</code></li>
  <li>Set up MongoDB, create a database and collections for storing meteorological data and user information.</li>
  <li>Start the Streamlit application: <code>streamlit run Introducere.py</code></li>
  <li>Access the application accessing the link: <code>http://localhost:8501</code></li>
</ol>
<h2>References</h2>
<p>The Streamlit application for studying the climate changes is based on two published articles regarding this theme:</p>
<ol>
    <li>
    <strong>Understanding Climate Change and Air Quality Over the Last Decade: Evidence From News and Weather Data Processing</strong> - We investigated the relationship between climate change and air quality, analyzing data over the past decade to uncover significant trends and correlations. Published in IEEE Access, 2023. [DOI: 10.1109/ACCESS.2020.3042571] (https://doi.org/10.1109/ACCESS.2020.3042571)
  </li>
  <li>
    <strong>Anomaly Detection in Weather Phenomena: News and Nuanced Data Processing</strong> - The main focus was detecting anomalies in weather phenomena by integrating data from news articles and detailed data processing techniques. Published in Springer International Journal of Computational Intelligence, 2024. [DOI: 10.1007/s00500-020-05241-3] (https://doi.org/10.1007/s00500-020-05241-3)
  </li>
</ol>
