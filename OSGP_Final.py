import os, glob, requests
from datetime import datetime, date
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import ee


ee.Initialize(project='ee-jpthompson26-osgp-final')
print("Earth Engine initialized.")

shp_path = "/Users/jamesthompson/Desktop/OSGP_final/OSGP_data_JPT/bothHoss/bothHoss.shp"
gdf = gpd.read_file(shp_path)
print(f"Loaded shapefile from: {shp_path}")

features = []
for _, row in gdf.iterrows():
    geom = row['geometry']
    if geom.geom_type == 'Polygon':
        coords = [list(geom.exterior.coords)]
        features.append(ee.Feature(ee.Geometry.Polygon(coords)))
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            coords = [list(poly.exterior.coords)]
            features.append(ee.Feature(ee.Geometry.Polygon(coords)))

aoi = ee.FeatureCollection(features)
print("Converted shapefile to Earth Engine FeatureCollection.")

start_date = '2024-11-01'
end_date = date.today().isoformat()

sentinel_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'B10']

s2 = (ee.ImageCollection('COPERNICUS/S2')
      .filterBounds(aoi)
      .filterDate(start_date, end_date)
      .select(sentinel_bands))

print(f"Loaded Sentinel-2 collection with {s2.size().getInfo()} images.")

s2_list = s2.toList(s2.size())
count = s2.size().getInfo()

#model from model.py
model_path = "landcover_classifier.pkl"
pipe = joblib.load(model_path)
print(f"Loaded trained model with classes: {pipe.classes_}")


output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(output_folder, exist_ok=True)

#process images
for i in range(count):
    try:
        image = ee.Image(s2_list.get(i))

        timestamp = image.date().format('YYYYMMdd').getInfo()
        print(f"Processing image from {timestamp}...")

        url = image.getDownloadURL({
            'bands': sentinel_bands,
            'scale': 10,
            'region': aoi.geometry(),
            'format': 'GEO_TIFF'
        })

        temp_filename = os.path.join(output_folder, f's2_image_{timestamp}.tif')
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            with open(temp_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded image to {temp_filename}")
        else:
            print(f"Failed to download image {i}. Status code: {response.status_code}")
            continue
            
        #classification
        with rasterio.open(temp_filename) as src:
            meta = src.meta.copy()
            bands_data = []
            for b in src.indexes:
                bands_data.append(src.read(b))

        stacked = np.stack(bands_data)  
        stacked = np.nan_to_num(stacked, nan=0, posinf=0, neginf=0)
        height, width = meta['height'], meta['width']
        reshaped = stacked.reshape(len(bands_data), -1).T  
        band_cols = [f'band_{i+1}' for i in range(len(bands_data))]
        reshaped_df = pd.DataFrame(reshaped, columns=band_cols)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(reshaped_df)
        scaled_df = pd.DataFrame(scaled_data, columns=band_cols)
        
        print("Classifying image...")
        preds = pipe.predict(scaled_df)
        unique_classes, class_counts = np.unique(preds, return_counts=True)
        for cls, count in zip(unique_classes, class_counts):
            percent = (count / len(preds)) * 100
            print(f"  {cls}: {percent:.2f}%")
        
        label_map = {label: idx for idx, label in enumerate(pipe.classes_)}
        
        preds_int = np.array([label_map[label] for label in preds])
        predicted_image = preds_int.reshape(height, width)
        
        classified_filename = os.path.join(output_folder, f'classified_{timestamp}.tif')
        meta.update({
            'count': 1,
            'dtype': 'int32',
            'nodata': 255
        })
        
        with rasterio.open(classified_filename, 'w', **meta) as dst:
            dst.write(predicted_image.astype('int32'), 1)
        
        print(f"Saved classified image to: {classified_filename}")
        
        os.remove(temp_filename)

    except Exception as e:
        print(f"Error processing image {i}: {str(e)}")
        import traceback
        traceback.print_exc()

print("Processing complete!")

# check for empty files
classified_files = glob.glob(os.path.join(output_folder, "classified_*.tif"))
if classified_files:
    with rasterio.open(classified_files[0]) as src:
        data = src.read(1)
        unique_values = np.unique(data)
        print(f"Validation: Classification contains classes: {unique_values}")
        if len(unique_values) > 1:
            print("Classification successful!")
        else:
            print("xWarning: Classification contains only one class!")





shapefile_path = '/Users/jamesthompson/Desktop/OSGP_final/OSGP_data_JPT/bothHoss/bothHoss.shp'
shapefile = gpd.read_file(shapefile_path)

input_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
tif_files = sorted(glob.glob(os.path.join(input_folder, "classified_*.tif")))

print(f"Found {len(tif_files)} classified images in {input_folder}")

features_list = []

for tif_path in tif_files:
    try:
        with rasterio.open(tif_path) as src:
            if shapefile.crs != src.crs:
                shapefile = shapefile.to_crs(src.crs)

            geoms = [shape(feature["geometry"]) for feature in shapefile.__geo_interface__["features"]] 
            raster_bounds = src.bounds
            geoms_within_raster = [
                geom for geom in geoms if geom.intersects(
                    shape({
                        "type": "Polygon",
                        "coordinates": [[
                            [raster_bounds[0], raster_bounds[1]],
                            [raster_bounds[2], raster_bounds[1]],
                            [raster_bounds[2], raster_bounds[3]],
                            [raster_bounds[0], raster_bounds[3]],
                            [raster_bounds[0], raster_bounds[1]]
                        ]]
                    })
                )
            ]
            
            if not geoms_within_raster:
                print(f"No valid geometries to clip for {tif_path}")
                continue

            out_image, out_transform = mask(src, geoms_within_raster, crop=True)

            if isinstance(out_image, np.ndarray):
                out_image = out_image[0]  

           
            ice_class = 1
            water_class = 0

        
            ice_mask = out_image == ice_class  
            water_mask = out_image == water_class  
            final_mask = water_mask  
            results = shapes(out_image, mask=final_mask, transform=out_transform)

            base = os.path.basename(tif_path)
            date_str = base.split('_')[1].split('.')[0]  
            capture_date = pd.to_datetime(date_str, format='%Y%m%d')

            geometries = [shape(geom) for geom, val in results if val == water_class]

            if geometries:
                merged = unary_union(geometries)
                if merged:
                    features_list.append({'geometry': merged, 'date': capture_date})
                    print(f"Processed {base}: Found water coverage for {capture_date}")
            else:
                print(f"No water detected in {base}")
                
    except Exception as e:
        print(f"Error reading {tif_path}: {e}")

if not features_list:
    print("No water features found in any classified images!")
    features_list.append({
        'geometry': Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
        'date': datetime.now()
    })


#-----STATS-----

df = pd.DataFrame(features_list)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:32618")
gdf['geometry'] = gdf['geometry'].apply(
    lambda geom: gpd.GeoSeries([geom]).buffer(0)[0] if not geom.is_valid else geom
)

gdf['area'] = gdf['geometry'].area
gdf['percent_coverage'] = (gdf['area'] / 1278774.20352) 
gdf.loc[gdf['percent_coverage'] > 1, 'percent_coverage'] = 1

gdf['date'] = pd.to_datetime(gdf['date'], errors='coerce')
gdf_cleaned = gdf.dropna(subset=['date', 'percent_coverage'])
gdf_sorted = gdf_cleaned.sort_values(by='date')
most_recent_entry = gdf_sorted.iloc[0]

X = gdf_sorted[['date']]
y = gdf_sorted['percent_coverage']
X = X['date'].map(datetime.toordinal).values.reshape(-1, 1)

#-----REGRESSIONS-----

degree = 2
poly = PolynomialFeatures(degree)

X = gdf_sorted[['date']]
y = gdf_sorted['percent_coverage']
X_ordinal = X['date'].map(datetime.toordinal).values.reshape(-1, 1)

# Fit regression model
X_poly = poly.fit_transform(X_ordinal)
model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

# Compute predicted date from roots
coefficients = model.coef_
intercept = model.intercept_
poly_coeffs = np.array([coefficients[2], coefficients[1], intercept])
roots = np.roots(poly_coeffs)
real_roots = [root.real for root in roots if np.isreal(root) and root.real > 0]

predicted_date = None
if real_roots:
    predicted_date_ordinal = max(real_roots)
    if predicted_date_ordinal >= 1:
        predicted_date = datetime.fromordinal(int(predicted_date_ordinal))

# Override predicted date if last 3 entries have <5% coverage
last_three = gdf_sorted.tail(2)
low_coverage = (last_three['percent_coverage'] < 0.10).all()

if low_coverage:
    predicted_date = datetime.now() + pd.Timedelta(days=1)
    print("Low coverage detected in last 3 entries. Using tomorrow's date as predicted date.")

mse = mean_squared_error(y, y_pred)



#-----DASHAPP-----

app = dash.Dash(__name__)
mapbox_access_token = "pk.eyJ1IjoianB0aG9tcHNvbjI2IiwiYSI6ImNtOXN1cTVjNTAzbHIyanE0NXpiYzFodGYifQ.xOm5dGPXrUap1wY24uH_aw"
date_options = [{'label': str(date.date()), 'value': date} for date in gdf_sorted['date'].unique()]

app.layout = html.Div([
    html.H1(
        "Ice Coverage Tracker Using Remote Sensing",
        style={
            'textAlign': 'center',
            'padding': '20px',
            'fontFamily': 'Times New Roman',
            'backgroundColor': 'lightblue',
            'color': 'black',
            'borderBottom': '2px solid black',
            'marginBottom': '0'
        }
    ),
    html.Div([
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='date-dropdown',
                    options=date_options,
                    value=date_options[-1]['value'],
                    style={'width': '250px'}
                )
            ], style={'position': 'absolute', 'top': '100px', 'left': '80px', 'zIndex': '10'}),
            dcc.Graph(id='coverage-map', style={'width': '99%', 'height': '100vh', 'backgroundColor': 'white','border': '2px solid black'}),
        ], style={'width': '50%', 'height': '100vh', 'position': 'relative', 'backgroundColor': 'white'}),

        html.Div([
            dcc.Graph(id='regression-plot', style={'width': '100%', 'height': '50vh', 'backgroundColor': 'white','border': '2px solid black'}),
            html.Div(id='predicted-date', style={'padding': '20px', 'fontSize': '20px', 'textAlign': 'center'}),
            html.Div(
                "This is a dashboard displaying remotley sensed data on ice coverage on the Hosmer Ponds in Craftsbury, Vermont. " \
                "The primary data source is Sentinal-2 surface reflectance imagery, which was aquired from the Google Earth Engine API. The data " \
                "is then placed into a pipeline which conducts a PCA and a Supervised Classification via " \
                "random forests. The classification strategically oversamples ice to provide conservitive estimates. " \
                "The data is converted to shapley objects using raster.io, and these objects are used for analysis and visualizations. " \
                "A two degree polynomial regression is run on the ice coverage, which provides further visuals and and an estimated date of zero coverage." ,
                style={
                    'backgroundColor': 'white',
                    'border': '2px solid black',
                    'padding': '15px',
                    'textAlign': 'left',
                    'fontSize': '18px',
                    'marginTop': '20px',
                    'borderRadius': '8px',
                    'height': '34%',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'justifyContent': 'center',
                    'alignItems': 'center',
                    'boxSizing': 'border-box',
                    'width': '100%',
                    'color': 'black'
                }
            )
        ], style={'width': '50%', 'height': '100vh', 'position': 'relative', 'backgroundColor': 'white'})
    ], style={'display': 'flex', 'backgroundColor': 'white'})
], style={'backgroundColor': 'white'})

@app.callback(
    Output('coverage-map', 'figure'),
    [Input('date-dropdown', 'value')]
)
def update_map(selected_date):

    background_gdf = gpd.read_file(shapefile_path)
    background_gdf = background_gdf.to_crs(epsg=4326)

    if selected_date is None:
        selected_date = gdf_sorted['date'].max()
    else:
        selected_date = pd.to_datetime(selected_date)

    selected_gdf = gdf_sorted[gdf_sorted['date'] == selected_date]
    selected_gdf = selected_gdf.to_crs(epsg=4326)

    fig = go.Figure()
    min_lon, min_lat, max_lon, max_lat = None, None, None, None

    water_legend_added = False
    ice_legend_added = False

    # lake plot
    for idx, row in background_gdf.iterrows():
        geom = row['geometry']
        if geom is None:
            continue
        if geom.geom_type == 'MultiPolygon':
            for polygon in geom.geoms:
                x, y = polygon.exterior.xy
                fig.add_trace(go.Scattermapbox(
                    lon=list(x),
                    lat=list(y),
                    mode='lines',
                    fill='toself',
                    line=dict(width=0.5, color='lightblue'),
                    fillcolor='lightblue',
                    opacity=0.3,
                    name='Water',
                    showlegend=not water_legend_added
                ))
                water_legend_added = True
        elif geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            fig.add_trace(go.Scattermapbox(
                lon=list(x),
                lat=list(y),
                mode='lines',
                fill='toself',
                line=dict(width=0.5, color='lightblue'),
                fillcolor='lightblue',
                opacity=0.3,
                name='Water',
                showlegend=not water_legend_added
            ))
            water_legend_added = True
    for idx, row in selected_gdf.iterrows():
        geom = row['geometry']
        if geom is None:
            continue
        if geom.geom_type == 'MultiPolygon':
            for polygon in geom.geoms:
                x, y = polygon.exterior.xy
                lons, lats = list(x), list(y)

                fig.add_trace(go.Scattermapbox(
                    lon=lons,
                    lat=lats,
                    mode='lines',
                    fill='toself',
                    line=dict(width=1, color='blue'),
                    fillcolor='blue',
                    opacity=0.5,
                    name='Ice',
                    showlegend=not ice_legend_added
                ))
                ice_legend_added = True

                if min_lon is None:
                    min_lon, max_lon = min(lons), max(lons)
                    min_lat, max_lat = min(lats), max(lats)
                else:
                    min_lon, max_lon = min(min_lon, min(lons)), max(max_lon, max(lons))
                    min_lat, max_lat = min(min_lat, min(lats)), max(max_lat, max(lats))

        elif geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            lons, lats = list(x), list(y)

            fig.add_trace(go.Scattermapbox(
                lon=lons,
                lat=lats,
                mode='lines',
                fill='toself',
                line=dict(width=1, color='blue'),
                fillcolor='blue',
                opacity=0.5,
                name='Ice',
                showlegend=not ice_legend_added
            ))
            ice_legend_added = True

            if min_lon is None:
                min_lon, max_lon = min(lons), max(lons)
                min_lat, max_lat = min(lats), max(lats)
            else:
                min_lon, max_lon = min(min_lon, min(lons)), max(max_lon, max(lons))
                min_lat, max_lat = min(min_lat, min(lats)), max(max_lat, max(lats))
    if min_lon is not None and min_lat is not None:
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
    else:
        center_lat, center_lon = 44.5, -72.5

    formatted_date = selected_date.strftime('%m/%d/%y')

    fig.update_layout(
        title=dict(
            text=f"Ice on the Hosmer Ponds on {formatted_date}",
            x=0.5,
            xanchor='center',
            font=dict(family='Times New Roman', size=22, color='black')
        ),
        paper_bgcolor='white',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12,
            style="carto-positron"
        ),
        legend=dict(                
            title=None,
            orientation='v',
            x=0.01,                        
            y=0.01,
            xanchor='left',
            yanchor='bottom',
            bgcolor='rgba(255,255,255,0.7)',  
            bordercolor='black',
            borderwidth=1,
            font=dict(
                family='Times New Roman',
                size=14,
                color='black'
            )
        )
    )

    return fig


@app.callback(
    Output('regression-plot', 'figure'),
    [Input('date-dropdown', 'value')]
)
def update_regression_plot(selected_date):
    fig = px.scatter(
        x=gdf_sorted['date'],
        y=gdf_sorted['percent_coverage'],
        labels={'x': 'Date', 'y': 'Ice Coverage Percentage'},
        title=f'{degree}-Degree Polynomial Regression for Ice Coverage'
    )

    fig.update_traces(name='Ice Coverage Sample', showlegend=True, mode='markers')

    # Regression line
    fig.add_scatter(
        x=gdf_sorted['date'],
        y=y_pred,
        mode='lines',
        name=f'{degree}-Degree Polynomial Regression',
        line=dict(color='red')
    )

    fig.update_layout(
        title=dict(
            text=f'{degree}-Degree Polynomial Regression for Ice Coverage',
            x=0.5,
            xanchor='center',
            font=dict(family='Times New Roman', size=22, color='black')
        ),
        paper_bgcolor='white'
    )

    return fig

@app.callback(
    Output('predicted-date', 'children'),
    [Input('date-dropdown', 'value')]
)
def display_predicted_date(selected_date):
    if predicted_date:
        return html.Div(
            f"Estimated Date of No Ice Coverage: {predicted_date.strftime('%B %d, %Y')}",
            style={
                'backgroundColor': 'white',
                'border': '2px solid black',
                'padding': '15px',
                'textAlign': 'center',
                'fontSize': '20px',
                'marginTop': '20px',
                'borderRadius': '8px',
                'height': '100%',
                'display': 'flex',
                'flexDirection': 'column',
                'justifyContent': 'center',
                'alignItems': 'center',
                'boxSizing': 'border-box',
                'width': '100%',
            }
        )
    else:
        return html.Div(
            "No estimated date available",
            style={
                'backgroundColor': 'white',
                'border': '2px solid black',
                'padding': '15px',
                'textAlign': 'center',
                'fontSize': '20px',
                'marginTop': '20px',
                'borderRadius': '8px',
                'height': '100%',
                'display': 'flex',
                'flexDirection': 'column',
                'justifyContent': 'center',
                'alignItems': 'center',
                'boxSizing': 'border-box',
                'width': '100%',
            }
        )

if __name__ == '__main__':
    app.run(debug=True)
