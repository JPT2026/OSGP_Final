import os
import glob
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterstats import point_query
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.utils import resample
import joblib

band_folder = "/Users/jamesthompson/Desktop/OSGP_final/OSGP_Tiff_Combiner"
shapefile = "/Users/jamesthompson/Desktop/OSGP_final/training2/training.shp"

gdf = gpd.read_file(shapefile)
print(gdf.head())
band_files = sorted(glob.glob(os.path.join(band_folder, "*.tiff")))
print(f"Found {len(band_files)} bands.")

for i, band_path in enumerate(band_files):
    band_values = point_query(gdf.geometry, band_path)
    gdf[f'band_{i+1}'] = band_values
band_cols = [col for col in gdf.columns if col.startswith('band_')]
gdf_clean = gdf.dropna(subset=band_cols)

# Ice oversampling
df_ice = gdf_clean[gdf_clean['classname'] == 'ice']
df_other = gdf_clean[gdf_clean['classname'] != 'ice']

if not df_ice.empty and len(df_ice) < len(df_other):
    df_ice_upsampled = resample(df_ice, replace=True, n_samples=len(df_other), random_state=42)
    gdf_balanced = pd.concat([df_ice_upsampled, df_other])
else:
    gdf_balanced = gdf_clean.copy()

X = gdf_balanced[band_cols]
y = gdf_balanced['classname']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

pipe = Pipeline([
    ('pca', PCA(n_components=2)),  
    ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])
pipe.fit(X_train, y_train)

with rasterio.open(band_files[0]) as src0:
    meta = src0.meta.copy()
    height, width = src0.height, src0.width

all_bands = []
for path in band_files:
    with rasterio.open(path) as src:
        all_bands.append(src.read(1))

stacked = np.stack(all_bands)  
reshaped = stacked.reshape(len(all_bands), -1).T  

reshaped_df = pd.DataFrame(reshaped, columns=band_cols)

preds = pipe.predict(reshaped_df)

label_map = {
    'water': 1,
    'ice': 0
}
label_map['ice'] = len(pipe.classes_)  
preds_int = np.array([label_map[label] for label in preds])
predicted_image = preds_int.reshape(height, width)

meta.update({
    'count': 1,
    'dtype': 'int32'
})
output_path = "predicted_landcover.tif"
with rasterio.open(output_path, 'w', **meta) as dst:
    dst.write(predicted_image.astype('int32'), 1)

print(f"Saved classified raster to: {output_path}")

#save pipeline
model_path = "landcover_classifier.pkl"
joblib.dump(pipe, model_path)
print(f"Saved model to: {model_path}")
