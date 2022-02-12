from dash import Dash, dcc, html, dash_table, State, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import p7
import lightgbm_with_simple_features as lgbmsf
import json

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

with open('./config.json', 'r') as f:
    CONFIG = json.load(f)

PATH = CONFIG["PATH"]
NUM_ROWS = CONFIG["NUM_ROWS"]


# Reindexing is needed here otherwise an error is thrown when trying to select a customer
raw_df = pd.read_csv('../data/raw_df_test.csv')
raw_df.index = pd.Index(range(1, raw_df.shape[0]+1))
print('raw data loaded')

df = pd.read_csv('../data/df_test.csv')
df.index = pd.Index(range(1, df.shape[0]+1))
print('processed data loaded')

app.layout = html.Div([
    html.Div([
        html.H1('Home Credit Default Dashboard'),
        html.Div([
            html.H4('Choix du dataset'),
            dcc.RadioItems(
                ['raw', 'processed'],
                'raw',
                id='df-type',
            )
        ]),
        html.Hr(),
        html.H4('Choix du client'),
        html.Div([
            html.Div([
                dcc.Dropdown(   
                value=1,
                id='client-id',
                ),
                dash_table.DataTable(
                    style_table={'overflowX': 'auto', 'height': 400},
                    fixed_rows={'headers': True},
                    id='client-data'
                )
            ], style={'width':'45%'}),

            html.Div([
                html.Div([
                    html.Div([
                        html.H4('Select X axis'),
                        dcc.Dropdown(
                            value='CODE_GENDER',
                            id='xaxis-column',
                        )
                    ], style={'width':'30%'}),
                    html.Div([
                        html.H4('Select Y axis'),
                        dcc.Dropdown(
                            value='CNT_CHILDREN',
                            id='yaxis-column',
                        )
                    ], style={'width':'30%'})
                ], style={'display':'flex'}),
                html.Div([
                    html.H4(
                        'Visualisation', 
                        style={'textAlign':'center'}
                    ),
                    dcc.RadioItems(
                            ['box', 'scatter'],
                            'box',
                            id='viz-type'
                    ),
                    dcc.Graph(id='credit_default')
                ], )
            ], style={'width':'45%'})
        ], style={'width':'100%', 'display':'flex'})
    ])
])


@app.callback(
    Output('xaxis-column', 'options'),
    Input('df-type', 'value'))
def set_columns_options(selected_dataset):
    if selected_dataset == 'raw':
        return raw_df.columns
    else:
        return df.columns

@app.callback(
    Output('yaxis-column', 'options'),
    Input('xaxis-column', 'value'),
    Input('df-type', 'value')
    )
def set_columns_options(selected_var, df_type):
    if df_type == 'raw':
        data = raw_df
        cat_cols = [col for col in data.columns if data[col].dtype=='object']
        num_cols = [col for col in data.columns if data[col].dtype!='object']
    else:
        data = df
        cat_cols = [col for col in data.columns if data[col].nunique()<=2]
        num_cols = [col for col in data.columns if data[col].nunique()>2]

    if selected_var in cat_cols:    
        return num_cols
    else:
        return data.columns

@app.callback(
    Output('client-id', 'options'),
    Input('df-type', 'value'))
def set_client_ids(selected_dataset):
    if selected_dataset == 'raw':
        return list(raw_df.index)
    else:
        return list(df.index)

@app.callback(
    Output('client-data', 'data'),
    Output('client-data', 'columns'),
    Input('client-id', 'value'),
    Input('df-type', 'value'))
def display_client_data(selected_id, selected_dataset):
    
    if selected_dataset == 'raw':
        data = raw_df.iloc[selected_id].T.reset_index()
    else:
        data = df.iloc[selected_id].T.reset_index()

    data.columns = ['Feature', 'Value']

    return data.to_dict('records'), [{"name": i, "id": i} for i in data.columns]


@app.callback(
    Output('credit_default', 'figure'),
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('client-id', 'value'),
    Input('df-type', 'value'),
    Input('viz-type', 'value')
    )
def update_graph(xaxis_column_name,
yaxis_column_name,
selected_id,
df_type,
viz_type
):
    if df_type == 'raw':
        d = raw_df
    else:
        d = df
    
    client_data = d.iloc[selected_id].to_frame().transpose()
    # Add traces
    if viz_type == 'scatter':
        fig1 = go.Scatter(
            name='data',
            x=d[xaxis_column_name],
            y=d[yaxis_column_name],
            mode='markers'
        )
    
    elif viz_type == 'box':
        fig1 = go.Box(
            name='data',
            x=d[xaxis_column_name],
            y=d[yaxis_column_name],
        )

    fig2 = go.Scatter(
        name='Client {}'.format(selected_id),
        x=client_data[xaxis_column_name],
        y=client_data[yaxis_column_name],
        mode='markers', 
        marker_symbol = 'star',
        marker_size = 15,
        )
    
    fig = make_subplots()
    fig.add_trace(fig1)
    fig.add_trace(fig2)
    fig.update_xaxes(title_text=xaxis_column_name)
    fig.update_yaxes(title_text=yaxis_column_name)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)