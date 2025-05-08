from dash import Dash, dcc, html, Input, Output, State, callback
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import plotly.express as px
import io
import os

import pandas as pd


app = Dash(__name__)

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            html.A('Upload File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'background-color': 'gray',
            'color': 'black',
            'text-decoration': 'none'
        },
        
        multiple=False
    ),
    html.Div(id='file-upload-output'),
html.H3("Select Target"),
dcc.Dropdown(id='target-dropdown', placeholder="Select target"),
html.Div(id='Graph-div', style = {'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center'} ,children=(
    html.Div(id='g1', children=(
        html.H3("Categorical Analysis"),
        dcc.RadioItems(id='categorical-radio', style = {'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'gap' : '10px'}),
        dcc.Graph(id='bar-chart-categorical', style={'display': 'inline-block'}),
    )),
    html.Div(id='g2', children=(
        html.H3("Correlation Strength of Numerical Variables with ..."),
        dcc.Graph(id='bar-chart-correlation' , style={'display': 'inline-block'}),
    )),
)),
html.Div(id="checkbox-div", children=(

    # Header
    html.H3("Select Features and Train Model", style={'textAlign': 'center'}),

    # Horizontal checklist
    html.Div([
        dcc.Checklist(
            id='feature-checklist',
            labelStyle={'display': 'inline-block', 'margin-right': '15px'},
            inline=True
        )
    ], id='feature-checkboxes', style={'textAlign': 'center', 'marginBottom': '15px'}),

    # Train Button
    html.Div([
        html.Button("Train Model", id='train-button', n_clicks=0)
    ], style={'textAlign': 'center', 'marginBottom': '10px'}),

    # R² output
    html.Div(id='train-output', style={'textAlign': 'center', 'marginBottom': '30px'}),

    # Prediction Section
    html.H3("Make a Prediction", style={'textAlign': 'center'}),

    html.Div([
        dcc.Input(id='prediction-input', type='text', placeholder='Enter feature values comma-separated', style={'width': '300px'}),
        html.Button('Predict', id='predict-button', n_clicks=0, style={'marginLeft': '10px'}),
        html.Div(id='prediction-output', style={'marginLeft': '15px', 'fontWeight': 'bold'})
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '10px'})

), style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
]),
#parses the CSV file and returns it as a data frame
def parse_contents(contents):
    import base64
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df

    

@callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def parse_uploaded_file(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@app.callback(
    Output('file-upload-output', 'children'),
    Output('target-dropdown', 'options'),
    Input('upload-data', 'contents'),
)
def update_output(content):
    global global_df
    if content is not None:
        global_df = parse_contents(content)
        numeric_cols = global_df.select_dtypes(include=np.number).columns
        return "", [{'label': col, 'value': col} for col in numeric_cols]
    return  "", []

@app.callback(
    Output('categorical-radio', 'options'),
    Output('bar-chart-correlation', 'figure'),
    Output('feature-checkboxes', 'children'),
    Input('target-dropdown', 'value')
)
def update_analysis(target):
    global global_df, target_variable
    if target is None or global_df.empty:
        return [], {}, {}, []

    target_variable = target
    cat_vars = global_df.select_dtypes(include='object').columns
    num_vars = global_df.select_dtypes(include=np.number).drop(columns=[target]).columns

    # First chart
    default_cat = cat_vars[0] if len(cat_vars) > 0 else None
    fig_cat = px.bar()
    if default_cat:
        group = global_df.groupby(default_cat)[target].mean().reset_index()
        fig_cat = px.bar(group, x=default_cat, y=target)

    # Second chart - correlations
    corrs = global_df[num_vars].corrwith(global_df[target]).abs().sort_values(ascending=False)
    fig_corr = px.bar(x=corrs.index, y=corrs.values, labels={'x': 'Feature', 'y': 'Absolute Correlation'})

    # Checkboxes for training
    checkboxes = [dcc.Checklist(
        id='feature-checklist',
        options=[{'label': col, 'value': col} for col in global_df.columns if col != target],
        value=[]
    )]

    return [{'label': col, 'value': col} for col in cat_vars], fig_corr, checkboxes

@app.callback(
    Output('bar-chart-categorical', 'figure'),
    Input('categorical-radio', 'value'),
    Input('target-dropdown', 'value')
)
def update_cat_bar(selected_cat, target):
    if global_df.empty or selected_cat is None or target is None:
        return {}
    group = global_df.groupby(selected_cat)[target].mean().reset_index()
    return px.bar(group, x=selected_cat, y=target)

@app.callback(
    Output('train-output', 'children'),
    Input('train-button', 'n_clicks'),
    State('feature-checklist', 'value')
)
def train_model(n_clicks, features):
    global trained_model, global_df, target_variable, selected_features

    if n_clicks > 0 and features and target_variable:
        selected_features = features
        X = global_df[features]
        y = global_df[target_variable]

        # Build pipeline
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include='object').columns.tolist()

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        model.fit(X, y)
        y_pred = model.predict(X)
        trained_model = model
        r2 = r2_score(y, y_pred)

        return f"Model trained successfully. R² score: {r2:.3f}"
    return ""

### Predict Callback
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('prediction-input', 'value')
)
def make_prediction(n_clicks, input_str):
    global trained_model, selected_features
    if n_clicks > 0:
        try:
            input_list = [x.strip() for x in input_str.split(',')]
            if len(input_list) != len(selected_features):
                return "Error: Incorrect number of input values."
            df = pd.DataFrame([input_list], columns=selected_features)
            prediction = trained_model.predict(df)[0]
            return f"Predicted value: {prediction:.2f}"
        except Exception as e:
            return f"Prediction failed: {str(e)}"
    return ""

server = app.server
# Run app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=True, host='0.0.0.0', port=port)
