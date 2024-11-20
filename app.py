from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from pandas import DataFrame
import matplotlib.pyplot as plt
import io
import base64
from graphviz import Source
import os

app = Flask(__name__)

# Funciones auxiliares
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para cada acción individual
@app.route('/action/<action>', methods=['GET'])
def action(action):
    # Ruta del dataset ajustada
    df = pd.read_csv('/home/gabriel/Documentos/ProyectosZam/datasets/TotalFeatures-ISCXFlowMeter.csv')

    if action == 'load_data':
        data_head = df.head(10).to_html()
        return jsonify({"message": "Datos cargados", "data": data_head})

    elif action == 'length_features':
        data_length = len(df)
        num_features = len(df.columns)
        return jsonify({"message": "Longitud y Características", "length": data_length, "features": num_features})

    elif action == 'split_scale':
        train_set, val_set, test_set = train_val_test_split(df)
        X_train, y_train = remove_labels(train_set, 'calss')
        X_val, y_val = remove_labels(val_set, 'calss')
        X_test, y_test = remove_labels(test_set, 'calss')

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        data_scaled_head = X_train_scaled.head(10).to_html()

        return jsonify({"message": "Dataset dividido y escalado", "scaled_data": data_scaled_head})

# Nueva Ruta: Visualizar el árbol de decisión
@app.route('/train_tree_visualize', methods=['POST'])
def train_tree_visualize():
    # Leer el dataset
    df = pd.read_csv('/home/gabriel/Documentos/ProyectosZam/datasets/TotalFeatures-ISCXFlowMeter.csv')

    # Convertir la columna 'calss' a valores numéricos
    df['calss'], _ = pd.factorize(df['calss'])

    # División del dataset
    train_set, val_set, _ = train_val_test_split(df)
    X_train, y_train = remove_labels(train_set, 'calss')

    # Reducir las características si es necesario
    X_train_reduced = X_train.iloc[:, :10]

    # Entrenar el clasificador con el conjunto reducido
    clf_tree_reduced = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf_tree_reduced.fit(X_train_reduced, y_train)

    # Exportar el árbol a un archivo .dot
    dot_file = "android_malware.dot"
    export_graphviz(
        clf_tree_reduced,
        out_file=dot_file,
        feature_names=X_train_reduced.columns,
        class_names=["bening", "adware", "malware"],
        rounded=True,
        filled=True
    )

    # Leer el archivo .dot y convertirlo a imagen PNG
    graph = Source.from_file(dot_file)
    graph.render("android_malware", format="png", cleanup=True)  # Generar archivo PNG y limpiar .dot

    # Leer la imagen PNG generada y convertirla a base64
    with open("android_malware.png", "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    return jsonify({"message": "Árbol de decisión visualizado", "image": img_base64})

if __name__ == '__main__':
    app.run(debug=True)
