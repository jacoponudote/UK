from qgis.gui import QgsMapCanvas
from qgis.core import QgsVectorLayer, QgsPointXY, QgsGeometry, QgsFeature, QgsProject
import pandas as pd

data = pd.read_csv('C:/Users/TE/Desktop/NEW_PORTS.csv')
# Create a new map canvas
canvas = QgsMapCanvas()
canvas.setWindowTitle("Points on QGIS")
size = 1

# Create a memory layer for the DataFrame points
points_layer = QgsVectorLayer('Point?crs=epsg:4326', '', 'memory')
pr = points_layer.dataProvider()

# Add features to the points layer
for index, row in data.iterrows():
    longitude = pd.to_numeric(row['LONGITUDE'], errors='coerce')
    latitude = pd.to_numeric(row['LATITUDE'], errors='coerce')
    if not pd.isnull(longitude) and not pd.isnull(latitude):
        point = QgsGeometry.fromPointXY(QgsPointXY(longitude, latitude))
        feature = QgsFeature()
        feature.setGeometry(point)
        pr.addFeature(feature)

# Add the points layer to the map
QgsProject.instance().addMapLayer(points_layer)

# Set symbol color for the points
renderer = points_layer.renderer()
symbol = renderer.symbol()
symbol.setSize(size)
points_layer.triggerRepaint()

# Refresh the map canvas
canvas.refresh()

# Show the canvas
canvas.show()
canvas.activateWindow()
