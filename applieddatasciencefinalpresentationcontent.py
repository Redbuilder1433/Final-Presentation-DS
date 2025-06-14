import pandas as pd
import numpy as np
import requests
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv")
df.head(10)
import time
# Identify and calculate the percentage of the missing values in each attribute
# Check for the null data, compute the sum, return the dataframe multiplied by a hundred for percentage values
df.isnull().sum()/len(df)*100

# Identify which columns are numerical and categorical
url = 'https://api.spacexdata.com/v4/launches'
# Make the GET request
response = requests.get(url)

# Convert the response to JSON
data = response.json()


# Normalize and convert to DataFrame (flatten nested JSON)
launch_df = pd.json_normalize(data)
pd.options.mode.use_inf_as_na = True
if launch_df.isnull().values.any():
    launch_df.fillna(0, inplace=True) #Fill NaN values with 0
print(launch_df.columns)
print(launch_df.dtypes)
print(launch_df.shape)
print(df.dtypes)

# Apply value_counts() on column LaunchSite
print(df['LaunchSite'].value_counts())

# Apply value_counts on Orbit column
print(df['Orbit'].value_counts())

# landing_outcomes = values on Outcome column
# Count the number and occurrence of each mission outcome
landing_outcomes = df['Outcome'].value_counts()

# Print it out
print(landing_outcomes)

for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)

bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
print(bad_outcomes)

# landing_class = 0 if bad_outcome
# landing_class = 1 otherwise
# Define the set of bad outcomes
bad_outcomes = {'False ASDS', 'False Ocean', 'False RTLS', 'None ASDS', 'None None'}

# Create the landing_class list using list comprehension
landing_class = [0 if outcome in bad_outcomes else 1 for outcome in df['Outcome']]

# Optional: attach it to the DataFrame
df['Class'] = landing_class

# Preview the result
print(df[['Outcome', 'Class']].head())

df['Class']=landing_class
df[['Class']].head(8)
# Calculate the mean success rate percentage
e = df["Class"].mean()
print(e*100)

print("================ New Page ================")

#%load_ext sql
import csv
import sqlite3
import prettytable

prettytable.DEFAULT = 'DEFAULT'

con = sqlite3.connect("my_data1.db")
cur = con.cursor()
#%sql sqlite:///my_data1.db
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False,method="multi")

#DROP THE TABLE IF EXISTS

#%sql DROP TABLE IF EXISTS SPACEXTABLE;

#%sql create table SPACEXTABLE as select * from SPACEXTBL where Date is not null

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cur.fetchall())
print("1")
#time.sleep(30)

cur.execute("PRAGMA table_info(SPACEXTBL);")
for col in cur.fetchall():
    print(col[1])
print("2")
#time.sleep(30)
cur.execute("SELECT DISTINCT Launch_Site FROM SPACEXTBL;")
print(cur.fetchall())
print("3")
#time.sleep(30)
cur.execute("SELECT * FROM SPACEXTBL WHERE Launch_Site LIKE 'CCA%' LIMIT 5;")
for row in cur.fetchall():
    print(row)
print("4")
#time.sleep(30)
cur.execute("SELECT SUM(PAYLOAD_MASS__KG_) FROM SPACEXTBL WHERE Customer = 'NASA (CRS)';")
print(cur.fetchone()[0])
print("5")
#time.sleep(30)
cur.execute("SELECT AVG(PAYLOAD_MASS__KG_) FROM SPACEXTBL WHERE Booster_Version = 'F9 v1.1';")
print(cur.fetchone()[0])
print("6")
#time.sleep(30)
cur.execute("SELECT Date FROM SPACEXTBL WHERE Landing_Outcome = 'Success (ground pad)' ORDER BY Date ASC LIMIT 1;")
print(cur.fetchone()[0])
print("7")
#time.sleep(30)
cur.execute("""
    SELECT Booster_Version
    FROM SPACEXTBL
    WHERE Landing_Outcome = 'Success (drone ship)'
      AND PAYLOAD_MASS__KG_ > 4000
      AND PAYLOAD_MASS__KG_ < 6000;
""")
for row in cur.fetchall():
    print(row[0])
print("8")
#time.sleep(30)
cur.execute("""
    SELECT Mission_Outcome, COUNT(*) AS outcome_count
    FROM SPACEXTBL
    GROUP BY Mission_Outcome;
""")
for row in cur.fetchall():
    print(row)
print("9")
#time.sleep(30)


cur.execute("""
    SELECT Booster_Version
    FROM SPACEXTBL
    WHERE PAYLOAD_MASS__KG_ = (
        SELECT MAX(PAYLOAD_MASS__KG_) FROM SPACEXTBL
    );
""")
for row in cur.fetchall():
    print(row[0])
print("10")
#time.sleep(30)
cur.execute("""
    SELECT 
        substr(Date, 6, 2) AS month, 
        Landing_Outcome, 
        Booster_Version, 
        Launch_Site
    FROM SPACEXTBL
    WHERE 
        substr(Date, 1, 4) = '2015'
        AND Landing_Outcome = 'Failure (drone ship)';
""")
for row in cur.fetchall():
    print(row)
print("11")
#time.sleep(30)
cur.execute("""
    SELECT Landing_Outcome, COUNT(*) AS outcome_count
    FROM SPACEXTBL
    WHERE Date >= '2010-06-04' AND Date <= '2017-03-20'
    GROUP BY Landing_Outcome
    ORDER BY outcome_count DESC;
""")
for row in cur.fetchall():
    print(row)
print("12")
#time.sleep(30)

import matplotlib.pyplot as plt

print(df.columns)
print(df.shape)
print(df.info())
# Example 1: Count of launches per site
site_counts = df['Launch_Site'].value_counts()
site_counts.plot(kind='bar')
plt.title('Launches per Site')
plt.xlabel('Launch Site')
plt.ylabel('Number of Launches')
plt.show()

# Example 2: Distribution of Payload Mass
df['PAYLOAD_MASS__KG_'].plot(kind='hist', bins=20)
plt.title('Distribution of Payload Mass')
plt.xlabel('Payload Mass (kg)')
plt.ylabel('Frequency')
plt.show()

# Example 3: Count of each landing outcome
df['Landing_Outcome'].value_counts().plot(kind='bar')
plt.title('Landing Outcomes')
plt.xlabel('Landing Outcome')
plt.ylabel('Count')
plt.show()
print(launch_df.columns)
print(df.columns)
import time
#time.sleep(1)
# launch_df
# Index(['static_fire_date_utc', 'static_fire_date_unix', 'net', 'window',
#       'rocket', 'success', 'failures', 'details', 'crew', 'ships', 'capsules',
#       'payloads', 'launchpad', 'flight_number', 'name', 'date_utc',
#       'date_unix', 'date_local', 'date_precision', 'upcoming', 'cores',
#       'auto_update', 'tbd', 'launch_library_id', 'id', 'fairings.reused',
#       'fairings.recovery_attempt', 'fairings.recovered', 'fairings.ships',
#       'links.patch.small', 'links.patch.large', 'links.reddit.campaign',
#       'links.reddit.launch', 'links.reddit.media', 'links.reddit.recovery',
#       'links.flickr.small', 'links.flickr.original', 'links.presskit',
#       'links.webcast', 'links.youtube_id', 'links.article', 'links.wikipedia',
#       'fairings'],
#      dtype='object')
# df
#Index(['Date', 'Time (UTC)', 'Booster_Version', 'Launch_Site', 'Payload',
#       'PAYLOAD_MASS__KG_', 'Orbit', 'Customer', 'Mission_Outcome',
#       'Landing_Outcome'],
#      dtype='object')
# 1. Flight Number vs Launch Site (from df) 
if 'flight_number' in launch_df.columns and 'Launch_Site' in df.columns:
    # Get the minimum length
    min_len = min(len(launch_df['flight_number']), len(df['Launch_Site']))
    x = launch_df['flight_number'][:min_len]
    y = df['Launch_Site'][:min_len]
    plt.figure(figsize=(8,5))
    plt.scatter(x, y)
    plt.title('Flight Number vs Launch Site')
    plt.xlabel('Flight Number')
    plt.ylabel('Launch Site')
    plt.show()

# 2. Payload Mass vs Launch Site (from df) 
if 'Payload' in df.columns and 'Launch_Site' in df.columns:
    plt.figure(figsize=(8,5))
    plt.scatter(df['Payload'], df['Launch_Site'])
    plt.title('Payload vs Launch Site')
    plt.xlabel('Payload')
    plt.ylabel('Launch Site')
    plt.xticks([])
    plt.show()

# 3. Success Rate vs Orbit Type (from df) 
if 'Orbit' in df.columns and 'success' in launch_df.columns:
    #tempdf = pd.DataFrame()
    #tempdf['Orbit'] = df['Orbit']
    #tempdf['success'] = launch_df['success']
    min_len = min(len(df['Orbit']), len(launch_df['success']))
    tempdf = pd.DataFrame({
        'Orbit': df['Orbit'][:min_len],
        'success': launch_df['success'][:min_len]
    })
    success_rate = tempdf.groupby('Orbit')['success'].mean()
    success_rate.plot(kind='bar')
    plt.title('Success Rate vs Orbit Type')
    plt.xlabel('Orbit Type')
    plt.ylabel('Success Rate')
    plt.show()

# 4. Flight Number vs Orbit Type (from df) 
if 'flight_number' in launch_df.columns and 'Orbit' in df.columns:
    min_len = min(len(launch_df['flight_number']), len(df['Orbit']))
    x = launch_df['flight_number'][:min_len]
    y = df['Orbit'][:min_len]
    plt.figure(figsize=(8,5))
    plt.scatter(x, y)
    plt.title('Flight Number vs Orbit Type')
    plt.xlabel('Flight Number')
    plt.ylabel('Orbit Type')
    plt.show()

# 5. Payload Mass vs Orbit Type (from df) 
if 'Payload' in df.columns and 'Orbit' in df.columns:
    plt.figure(figsize=(8,5))
    plt.scatter(df['Payload'], df['Orbit'])
    plt.title('Payload vs Orbit Type')
    plt.xlabel('Payload')
    plt.ylabel('Orbit Type')
    plt.xticks([])
    plt.show()

# 6. Launch Success Yearly Trend (from df) 
if 'Date' in df.columns and 'success' in launch_df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    # Get minimum length
    min_len = min(len(df['Date']), len(launch_df['success']))
    # Create a temporary DataFrame
    tempdf2 = pd.DataFrame({
        'Date': df['Date'][:min_len],
        'success': launch_df['success'][:min_len]
    })
    #tempdf2=pd.DataFrame()
    #tempdf2['Date'] = df['Date']
    #tempdf2['success'] = launch_df['success']
    yearly_success = tempdf2.groupby(tempdf2['Date'].dt.year)['success'].mean()
    yearly_success.plot(marker='o')
    plt.title('Launch Success Yearly Trend')
    plt.xlabel('Year')
    plt.ylabel('Success Rate')
    plt.show()

import yfinance as yf
import pandas as pd
import logging
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

# Disclaimer: yfinance is outdated. 
# The ".info" and ".get_info" functions no longer work.
# When trying to run yfinance, 
# you will get this debug:

# DEBUG:urllib3.connectionpool:Starting new HTTPS connection (1): query2.finance.yahoo.com:443
# DEBUG:urllib3.connectionpool:https://query2.finance.yahoo.com:443 "GET /v8/finance/chart/TSLA?period1=1262322000&period2
# And if you attempt to visit "https://query2.finance.yahoo.com/" you will get an error from Yahoo.
# No data exists at this HTTPS connection. So it fails to work.
# I tried to resolve this by getting historical data (before I knew the problem)
# It does not work, the query is completely nonfunctional.

# "raise JSONDecodeError("Expecting value", s, err.value) from None"
# "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)"
# There is no data to grab.

# apple = yf.Ticker("AAPL") This was a test line
#logging.basicConfig(level=logging.DEBUG)

# try:
    # tesla_stock = yf.download("TSLA", start="2010-01-01")
    # gamestop_stock = yf.download("GME", start="2010-01-01")
# except Exception as e:
    # print("Error downloading")

# print(tesla_stock.head())
# print(gamestop_stock.head())

# Below code is using the newest yfinance version which is not the one in the assignment details


tesla = yf.Ticker("TSLA")
gme = yf.Ticker("GME")


# stock_data, title as placeholders for later specification
def make_graph(stock_data, title):
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data["Date"], stock_data["Close"])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.grid(True)
    plt.show()
    


tesla_data = tesla.history(period="max")
tesla_data.reset_index(inplace=True)
print(tesla_data.head())

gme = yf.Ticker("GME")
gme_data = gme.history(period="max")
gme_data.reset_index(inplace=True)
print(gme_data.head())

def scrape_revenue(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    html = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    df = pd.read_html(str(tables[1]))[0]
    df.columns = ["Date", "Revenue"]
    df["Revenue"] = df["Revenue"].str.replace(r"[$,]", "", regex=True)
    df.dropna(inplace=True)
    return df
tesla_url = "https://www.macrotrends.net/stocks/charts/TSLA/tesla/revenue"
gme_url = "https://www.macrotrends.net/stocks/charts/GME/gamestop/revenue"

tesla_revenue = scrape_revenue(tesla_url)
tesla_revenue.columns = ['Date','Revenue']
tesla_revenue["Revenue"] = tesla_revenue['Revenue'].str.replace(',|\$',"", regex=True)

gme_revenue = scrape_revenue(gme_url)
gme_revenue.columns = ['Date','Revenue']
gme_revenue["Revenue"] = gme_revenue['Revenue'].str.replace(',|\$',"", regex=True)


print(tesla_revenue.tail())
print(gme_revenue.tail())

make_graph(tesla_data, "Tesla Stock Price Over Time")
make_graph(gme_data, "GameStop Stock Price Over Time")
# Display graphs
plt.show()
time.sleep(20)


import folium

import folium

#launch_sites = [
    #{"site": "CCAFS SLC-40", "lat": 28.5618571, "lon": -80.577366},
    #{"site": "KSC LC 39A", "lat": 28.60839, "lon": -80.60433},
    #{"site": "VAFB SLC 4E", "lat": 34.632834, "lon": -120.610746},
    #{"site": "CCAFS LC-40", "lat": 28.5623, "lon": -80.5774},
#]
options=[{'label': 'All Sites', 'value': 'ALL'}] + [
    {'label': site, 'value': site} for site in df['Launch_Site'].unique()
]
map_all = folium.Map(location=[30, -100], zoom_start=3)
launch_sites = [
    {"site": "CCAFS LC-40", "lat": 28.5618571, "lon": -80.577366},
    {"site": "KSC LC 39A", "lat": 28.60839, "lon": -80.60433},
    {"site": "VAFB SLC 4E", "lat": 34.632834, "lon": -120.610746},
    {"site": "CCAFS SLC-40", "lat": 28.5623, "lon": -80.5774},
]

for site in launch_sites:
    folium.Marker([site['lat'], site['lon']], popup=site['site']).add_to(map_all)
map_all.save('folium_all_launch_sites.html')
print("All launch sites map saved as folium_all_launch_sites.html")


# Define the percent success per site (normalized to color)
site_outcomes = {
    "KSC LC 39A": 0.417,
    "CCAFS LC-40": 0.292,
    "VAFB SLC 4E": 0.167,
    "CCAFS SLC-40": 0.125,
}

def outcome_color(rate):
    if rate >= 0.4:
        return 'green'
    elif rate >= 0.25:
        return 'orange'
    else:
        return 'red'

map_outcomes = folium.Map(location=[30, -100], zoom_start=3)
for site in launch_sites:
    rate = site_outcomes.get(site['site'], 0.125) # Just so CCAFS SLC-40 has its correct rate since it wont see it for some reason
    folium.CircleMarker(
        location=[site['lat'], site['lon']],
        radius=14,
        color=outcome_color(rate),
        fill=True,
        fill_color=outcome_color(rate),
        fill_opacity=0.8,
        popup=f"{site['site']} ({(rate*100)}% success)"
    ).add_to(map_outcomes)
map_outcomes.save('folium_outcome_map.html')
print("Launch outcome map saved as folium_outcome_map.html")

import geopy
from geopy.distance import geodesic

ksc_coords = (28.60839, -80.60433)

coastline_coords = (ksc_coords[0], ksc_coords[1] + 0.048)  # about 4.8km

prox_map = folium.Map(location=[ksc_coords[0], ksc_coords[1]], zoom_start=12)
folium.Marker(ksc_coords, popup="KSC LC 39A Launch Pad").add_to(prox_map)
folium.Marker(coastline_coords, popup="Atlantic Coastline").add_to(prox_map)
#folium.Marker(popup='2.93 km from KSC LC 39A to Atlantic Coastline', location=coastline_coords).add_to(prox_map)
folium.PolyLine([ksc_coords, coastline_coords], color='blue', weight=3, popup='4.8 km distance to coast').add_to(prox_map)

# Calculate and display the geodesic distance in the console
distance_km = geodesic(ksc_coords, coastline_coords).km
print(f"Distance from KSC LC 39A to simulated coastline: {distance_km:.2f} km")

prox_map.save('folium_proximity_map.html')
print("Proximity to coastline map saved as folium_proximity_map.html")




import plotly.express as px

# Bar chart: Launches per site
fig = px.bar(df, x='Launch_Site', title='Launches per Site')
fig.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv')

# Use correct column names!
features = ['PAYLOAD_MASS__KG_', 'Booster_Version', 'Launch_Site']
X = pd.get_dummies(df[features])  # one-hot encode categorical features

# Make a binary target: success=1, anything else=0
y = df['Mission_Outcome'].apply(lambda x: 1 if x == 'Success' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Assuming y_test and y_pred are already defined
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix: Logistic Regression")
plt.show()
accuracy = accuracy_score(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.bar(['Logistic Regression'], [accuracy], color='skyblue')
plt.ylabel('Accuracy')
plt.title('Classification Model Accuracy')
plt.ylim(0, 1)
plt.show()






# The index for the launch dataframe is:
# Index(['static_fire_date_utc', 'static_fire_date_unix', 'net', 'window',
#       'rocket', 'success', 'failures', 'details', 'crew', 'ships', 'capsules',
#       'payloads', 'launchpad', 'flight_number', 'name', 'date_utc',
#      'date_unix', 'date_local', 'date_precision', 'upcoming', 'cores',
#       'auto_update', 'tbd', 'launch_library_id', 'id', 'fairings.reused',
#       'fairings.recovery_attempt', 'fairings.recovered', 'fairings.ships',
#       'links.patch.small', 'links.patch.large', 'links.reddit.campaign',
#       'links.reddit.launch', 'links.reddit.media', 'links.reddit.recovery',
#       'links.flickr.small', 'links.flickr.original', 'links.presskit',
#       'links.webcast', 'links.youtube_id', 'links.article', 'links.wikipedia',
#       'fairings'],
#      dtype='object')

print("Total launches:", len(df))
# Count the number of successful launches
successful_launches = launch_df['success'].sum()
print("Successful launches:", successful_launches)
# Count the number of failed launches
failed_launches = len(launch_df) - successful_launches
print("Failed launches:", failed_launches)



# List is unhashable

# KeyError: (9, 'Landing_Outcome', 'TEXT', 0, None, 0)
# Cannot hash list values, so we will iterate through the columns and print the unique values for each column
def safe_print_unique(df):
    """Print types and sample unique values for each column, handling unhashable types."""
    print("=== Column Types and Unique Value Samples ===")
    for col in df.columns:
        print(f"\nColumn: {col}")
        types = df[col].apply(type).unique()
        print("Types:", types)
        try:
            uniques = df[col].unique()
            print("Unique values (sample):", uniques[:5])
        except TypeError:
            print("Cannot get unique values: column contains unhashable types (like list or dict).")

def print_value_counts(df, column):
    """Print value counts for a given column."""
    if column in launch_df.columns:
        print(f"\nValue counts for '{column}':")
        print(launch_df[column].value_counts())
    else:
        print(f"\n1Column '{column}' does not exist in the DataFrame.")

def print_unique_list_values(df, column):
    """Print unique values for columns containing lists or unhashable types."""
    if column in launch_df.columns:
        seen = []
        print(f"\nUnique values for '{column}':")
        count = 0
        for value in launch_df[column]:
            if not any(repr(value) == repr(x) for x in seen):
                seen.append(value)
                count += 1
                if isinstance(value, list):
                    print(f"  List of length {len(value)}: {value[:5]}...")
                else:
                    print(f"  {value}")
        print(f"Total unique values: {count}")
    else:
        print(f"\n2Column '{column}' does not exist in the DataFrame.")

# --- Usage Example ---

safe_print_unique(df)

time.sleep(10)

# For columns known to contain lists/unhashable types, use:
list_columns = [
    'ships', 'crew', 'capsules', 'payloads', 'cores',
    'fairings.ships', #'fairings.flickr.small', #'fairings.flickr.original'
]
for col in list_columns:
    print_unique_list_values(df, col)

# For value counts on categorical columns:
print_value_counts(df, 'rocket')
print_value_counts(df, 'launchpad')

# TASK 1: Add a Launch Site Drop-down Input Component
# TASK 2: Add a callback function to render success-pie-chart based on selected site dropdown
# TASK 3: Add a Range Slider to Select Payload
# TASK 4: Add a callback function to render the success-payload-scatter-chart scatter plot

# wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv"
# Returns: 
# StatusCode        : 200
# StatusDescription : OK
# Content           : ,Flight Number,Launch Site,class,Payload Mass (kg),Booster Version,Booster Version Category
#                    0,1,CCAFS LC-40,0,0,F9 v1.0  B0003,v1.0
#                    1,2,CCAFS LC-40,0,0,F9 v1.0  B0004,v1.0
#                    2,3,CCAFS LC-40,0,525,F9 ...
# RawContent        : HTTP/1.1 200 OK
#                     X-Clv-Request-Id: ca604ead-04f9-456a-a051-d427478fcdae
#                     X-Clv-S3-Version: 2.5
#                     x-amz-request-id: ca604ead-04f9-456a-a051-d427478fcdae
#                     Accept-Ranges: bytes
#                   Content-Length: 2476
#                    Cach...
# Forms             : {}
# Headers           : {[X-Clv-Request-Id, ca604ead-04f9-456a-a051-d427478fcdae], [X-Clv-S3-Version, 2.5], [x-amz-request-id,
#                    ca604ead-04f9-456a-a051-d427478fcdae], [Accept-Ranges, bytes]...}
# Images            : {}
# InputFields       : {}
# Links             : {}
# ParsedHtml        : mshtml.HTMLDocumentClass
# RawContentLength  : 2476import urllib.request
import urllib.request
# Download the SpaceX launch data CSV file
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv"
output_path = "spacex_launch_dash.csv"
urllib.request.urlretrieve(url, output_path)
print("Download complete!")
# wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/t4-Vy4iOU19i8y6E3Px_ww/spacex-dash-app.py"
# This is what we will be working with
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/t4-Vy4iOU19i8y6E3Px_ww/spacex-dash-app.py"
output_path = "spacex_launch_dash_skeleton.csv"
urllib.request.urlretrieve(url, output_path)
print("Download complete!")

# Import required libraries
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()
# Create a dash application
app = dash.Dash(__name__)

# Fix code by converting min_payload and max_payload to integers before layout definition
#min_payload = int(spacex_df['Payload Mass (kg)'].min())
#max_payload = int(spacex_df['Payload Mass (kg)'].max())

# Create an app layout
app.layout = html.Div(children=[
    html.H1('SpaceX Launch Records Dashboard',
            style={'textAlign': 'center', 'color': '#503D36',
                   'font-size': 40}),
    # TASK 1: Add a dropdown list to enable Launch Site selection
    # The default select value is for ALL sites
    # dcc.Dropdown(id='site-dropdown',...)
    html.Br(),
    html.Div(dcc.Dropdown(id='site-dropdown',
                         options=[
                             {'label': 'All Sites', 'value': 'ALL'},
                             {'label': 'CCAFS LC-40', 'value': 'CCAFS LC-40'},
                             {'label': 'VAFB SLC-4E', 'value': 'VAFB SLC-4E'},
                             {'label': 'KSC LC-39A', 'value': 'KSC LC-39A'},
                             {'label': 'CCAFS SLC-40', 'value': 'CCAFS SLC-40'}
                         ],
                         value='ALL',
                         placeholder="Select a Launch Site",
                         searchable=True
                         )),
    # TASK 2: Add a pie chart to show the total successful launches count for all sites
    # If a specific launch site was selected, show the Success vs. Failed counts for the site
    html.Div(dcc.Graph(id='success-pie-chart')),
    html.Br(),
    # Show success vs failed counts for all sites via 'success-pie-chart'
    html.P("Payload range (Kg):"),
    # TASK 3: Add a slider to select payload range
    #dcc.RangeSlider(id='payload-slider',...)
    # Fix code by converting min_payload and max_payload to integers
    dcc.RangeSlider(id='payload-slider',
                    min=0,    #min=min_payload,
                    max=10000,    #max=max_payload,
                    step=1000,
                    marks={0: '0',
                           100: '100'},        #marks={i: str(i) for i in range(min_payload, max_payload + 1, 1000)},
                    value=[min_payload, max_payload], 
                    tooltip={"placement": "bottom", "always_visible": True}
                    ),
    # TASK 4: Add a scatter chart to show the correlation between payload and launch success
    html.Div(dcc.Graph(id='success-payload-scatter-chart')),
])
                                
# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output

@app.callback(
    Output('success-pie-chart', 'figure'),
    Input('site-dropdown', 'value')
)
def update_pie_chart(selected_site):
    if selected_site == 'ALL':
        # Pie chart: total successful launches by site
        fig = px.pie(
            spacex_df, 
            names='Launch Site', 
            values='class', 
            title='Total Successful Launches by Site'
        )
    else:
        # Filter for the selected site
        filtered_df = spacex_df[spacex_df['Launch Site'] == selected_site]
        # Count number of successes and failures
        outcome_counts = filtered_df['class'].value_counts().reset_index()
        outcome_counts.columns = ['class', 'count']
        outcome_counts['class'] = outcome_counts['class'].map({1: 'Success', 0: 'Failure'})
        fig = px.pie(
            outcome_counts,
            names='class',
            values='count',
            title=f'Success vs Failure for site {selected_site}',
            color='class',
            color_discrete_map={'Success': 'red', 'Failure': 'blue'}
        )
    return fig

# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output

@app.callback(
    Output('success-payload-scatter-chart', 'figure'),
    [Input('site-dropdown', 'value'),
     Input('payload-slider', 'value')]
)
def update_scatter(selected_site, payload_range):
    # Filter data by payload range
    filtered_df = spacex_df[
        (spacex_df['Payload Mass (kg)'] > 0) &
        (spacex_df['Payload Mass (kg)'] >= payload_range[0]) &
        (spacex_df['Payload Mass (kg)'] <= payload_range[1]) 
    ]
    # Further filter by site if not ALL
    if selected_site != 'ALL':
        filtered_df = filtered_df[filtered_df['Launch Site'] == selected_site]
    # Create scatter plot
    fig = px.scatter(
        filtered_df,
        x='Payload Mass (kg)',
        y='class',
        color='Booster Version Category',
        title='Correlation between Payload and Success for ' +
              (selected_site if selected_site != 'ALL' else 'All Sites'),
        labels={'class': 'Launch Success (1=Success, 0=Failure)'}
    )
    return fig

        
update_pie_chart('ALL')  # Initialize pie chart with all sites


# Run the app
if __name__ == '__main__':
    app.run(debug=False)