from dash import Dash, dcc, html, dash_table, State, Input, Output
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import json
import joblib
import time
from contextlib import contextmanager

import shap
import requests 

with open('./config.json', 'r') as f:
    CONFIG = json.load(f)   

PATH = CONFIG["PATH"]
NUM_ROWS = CONFIG["NUM_ROWS"]

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Reindexing is needed here to prevent any error when selecting a customer
#raw_df = pd.read_csv('../data/raw_df_test.csv')
#raw_df.index = pd.Index(range(1, raw_df.shape[0]+1))
#print('raw data loaded')

# 3. Load data
with timer('Loading loan application...'):
    app_train = pd.read_csv(PATH+'application_train.csv', nrows=NUM_ROWS)
with timer('Loading previous credits (raw)...'):
    bureau = pd.read_csv(PATH+'bureau.csv', nrows=NUM_ROWS)
with timer('Loading previous credits monthly balance...'):
    bureau_balance = pd.read_csv(PATH+'bureau_balance.csv', nrows=NUM_ROWS)
with timer('Loading previous applications...'):
    previous_app = pd.read_csv(PATH+'previous_application.csv', nrows=NUM_ROWS)
with timer('Loading previous POS & card loans monthly balance...'):
    pos_cash = pd.read_csv(PATH+'POS_CASH_balance.csv', nrows=NUM_ROWS)
with timer('Loading repayment history...'):
    installment_payments = pd.read_csv(PATH+'installments_payments.csv', nrows=NUM_ROWS)
with timer('Loading previous credit card monthly balance...'):
    credit_card_balance = pd.read_csv(PATH+'credit_card_balance.csv', nrows=NUM_ROWS)
with timer('Loading processed data'):
    processed_data = pd.read_csv(PATH+'complete_test.csv', nrows=NUM_ROWS)

# Les index récupérables sont restreints aux clients sur lesquels on applique le modèle
client_ids = processed_data['SK_ID_CURR'].sort_values().to_list()[:NUM_ROWS]
#bb_id = bureau.loc[bureau.SK_ID_CURR.isin(client_ids), 'SK_ID_BUREAU'],

data_dict = {
    "loan application (raw)":app_train, 
    'previous credits (raw)':bureau,
    'previous credits monthly balance': bureau_balance,
    'previous applications': previous_app,
    'previous POS & card loans monthly balance': pos_cash,
    'repayment history': installment_payments,
    'previous credit card monthly balance': credit_card_balance,
    'processed data': processed_data
}

#df = pd.read_csv('../data/df_test.csv')
#df.index = pd.Index(range(1, df.shape[0]+1))
feats = data_dict['processed data'].drop(['SK_ID_CURR'], axis=1).columns
#print('processed data loaded')




# load prediction and explainer models
model = joblib.load(open(PATH+'fitted_lgbm.pickle', "rb"))
#tree_explainer = joblib.load(open('../../treeExplainer.pkl', 'rb'))
shap_values = joblib.load(open('../../shap_values.pkl', 'rb'))
base_value = joblib.load(open('../../base_value.pkl', 'rb'))
# getting shap value
#shap_values = tree_explainer(processed_data.drop(['SK_ID_CURR'], axis=1))
#base_value = tree_explainer.expected_value

# calculate the global feature importances and create the plot
global_feature_importance = pd.DataFrame(data=model.feature_importances_, index=feats, columns=['Feature_importance'])
f = global_feature_importance["Feature_importance"].sort_values(ascending=False)[:20][::-1]
fig = px.bar(data_frame=f, 
x='Feature_importance', 
labels={"index": "Features"},
title='Global feature importances',
width=600)
fig.update_layout(
    margin=dict(r=10),
    plot_bgcolor='rgba(0,0,0,0)'
)
fig.update_xaxes(
    showgrid=True, 
    gridwidth=1, 
    gridcolor='LightGrey'
    )

def predict_default(model, client_features):
    prediction = model.predict(client_features)[0]
    probability = model.predict_proba(client_features)[0].max()
    print(prediction, probability)
    return prediction, probability


def waterfall_plot(selected_id, prediction):
    df = data_dict['processed data']
    shap_id = df[df.SK_ID_CURR==selected_id].index
    shap_df = pd.DataFrame(data=shap_values.values[shap_id], columns=feats)
    high_importance_features = abs(shap_df.iloc[0].values).argsort()[::-1][:10]
    less_important_features = abs(shap_df.iloc[0].values).argsort()[::-1][10:]
    
    idx1 = pd.Index(['All other {} features'.format(len(feats)-10)])
    idx2 = pd.Index(shap_df.columns[high_importance_features[::-1]][1:])

    rest_importance = shap_df.iloc[0][less_important_features].sum()
    text1 = ['{:.2f}'.format(rest_importance)]
    text1 += ['{:.2f}'.format(v) for v in shap_df.iloc[0][high_importance_features[::-1][1:]]]

    importances = [rest_importance]
    shap_h_importances = shap_df.iloc[0][high_importance_features[::-1][1:]].to_list()
    importances += shap_h_importances

    pred, proba = prediction['prediction'], prediction['probability']
    
    fig = go.Figure(go.Waterfall(
        x = idx1.append(idx2),
        textposition = "outside",
        text = text1,
        base = base_value,
        y = importances,
        connector = {"line":{"dash":"dot","width":1, "color":"rgb(63, 63, 63)"}},
        name = 'Predicted probability of default = {:.2f}'.format([proba, 1 - proba][1-pred]),
    ))
    output = base_value + rest_importance + shap_df.iloc[0][high_importance_features[::-1][1:10]].sum()
    fig.add_hline(y=output, line_width=1, line_dash="dash", line_color="black")
    fig.add_hline(y=base_value, line_width=1, line_dash="dash", line_color="black")
    fig.add_annotation(
        xref="x domain",
        x=1.2,
        y=base_value + rest_importance + shap_df.iloc[0][high_importance_features].sum(),
        text="Output = {:.2f}".format(output),
        showarrow=False, 
    )
    fig.add_annotation(
        xref='x domain',
        x=1.25,
        y=base_value,
        text="Base value = {:.2f}".format(base_value),
        showarrow=False, 
    )
    shap_cumsum = np.asarray(base_value + rest_importance + shap_df.iloc[0][high_importance_features][::-1][1:].cumsum())
    ymin = min(shap_cumsum)
    ymax = max(shap_cumsum)
    ylim = [ymin*0.7, ymax*1.1]

    fig.update_layout(
            title = "Probability of default explained by shap values",
            showlegend = True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[ylim[0],ylim[1]]),
            width=700,
            height=600,
            margin=dict(
                r=150
            )
    )

    return fig
    
# Datasets
DATASETS = ("loan application (raw)",
"previous credits (raw)",
"previous credits monthly balance",
"previous POS & card loans monthly balance",
"previous credit card monthly balance",
"previous applications",
"repayment history",
 "processed data")


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Home Credit Default Dashboard'

customer_selection = dcc.Dropdown(
        id="customer-selection",
        options=client_ids,
        value=client_ids[0],
        style={'width':'75%'}
    )

customer_input_group = dbc.InputGroup(
    [
        dbc.InputGroupText('Client ID'),
        customer_selection
    ],
    size='sm'
)

show_prediction_card = dbc.Card(
    [
        dbc.CardHeader("Client select and predict"),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5('Select'),
                                customer_input_group
                            ],  
                            md=6
                        ),
                        dbc.Col(
                            [
                                html.H5('Predict'),
                                html.H2(id="predicted-target", style={"text-align": "center"})
                            ], 
                            md=6
                        )
                    ]
                )
            ]
        )
    ],
)

data_selection_dropdown = dcc.Dropdown(
            id='choice-dataset',
            options=[d for d in DATASETS],
            value = 'loan application (raw)',
            style={'width':'50%'}
)

data_selection_input_group = dbc.InputGroup(
    children=[
        dbc.InputGroupText("Choose dataset"),
        data_selection_dropdown
        
    ],
    size='sm'
)

client_features_card = dbc.Card(
    [
        dbc.CardHeader("Selected client features"),
        dbc.CardBody(
            [
                dbc.Row(dbc.Col(data_selection_input_group), style={'padding-bottom':'20px'}),
                dbc.Row(
                    dbc.Col(
                        dash_table.DataTable(
                            style_table={'overflowX': 'auto', 'height': "450px"},
                            fixed_rows={'headers': True},
                            id='client-data'
                        )
                    ),
                )
            ]
        )
    ]
)

xaxis_selection = dcc.Dropdown(
    value='NAME_CONTRACT_TYPE',
    id='xaxis-column'
    )

yaxis_selection = dcc.Dropdown(
    value='AMT_INCOME_TOTAL',
    id='yaxis-column'
    )

viz_type = dcc.Dropdown(
    options=['box', 'scatter'],
    value='box',
    id='viz-type'
)

color_selection = dcc.Dropdown(
    value='CODE_GENDER',
    id='color-selection'
    )

viz_card = dbc.Card(
    [
        dbc.CardHeader("Explore selected client features"),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col([html.H5("Select X axis"),xaxis_selection], md=4),
                        dbc.Col([html.H5("Select Y axis"),yaxis_selection], md=4), 
                        dbc.Col([html.H5("Select type of plot"),viz_type], md=4),
                    ]
                ),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col([html.H5("Additional grouping variable:"), color_selection], md=4)
                    ]
                ),
                html.Hr(),
                dbc.Row(
                        dbc.Col(
                            dcc.Graph(
                            id="credit-default",
                            style={"height": "450px"}
                            )
                        )
                    )
            ]
        )
    ]
)

feat_importance_card = dbc.Card(
    [
        dbc.CardHeader('Local model interpretation'),
        dbc.CardBody(
            dcc.Graph(
                id='feat-importances',
                style={'height':'600px'},
            )
        )
    ]
)

global_feat_importance_card = dbc.Card(
    [
        dbc.CardHeader('Global model interpretation'),
        dbc.CardBody(
            dcc.Graph(
                id='global-feat-importances',
                style={'height':'500px'},
                figure=fig
            )
        )
    ]
)

app.layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Store(id='intermediate-value'),
        html.H1('Home Credit Default Dashboard'),
        html.Hr(),
        dbc.Row([
            dbc.Col(
                children=[
                    show_prediction_card,
                    html.Br(),
                    client_features_card,
                    html.Br(),
                    global_feat_importance_card
                ],
                md=6,
            ),
            dbc.Col(
                children=[
                    viz_card,
                    html.Br(),
                    feat_importance_card
                ],
                md=6,
            )
        ], align='start'),
    ]
)    

@app.callback(
    Output('intermediate-value', 'data'),
    Input('customer-selection', 'value'))
def fetch_api_response(selected_id):
    df = data_dict['processed data']
    client_idx = df[df.SK_ID_CURR==selected_id].index[0]
    client_features = df[df.SK_ID_CURR==selected_id].to_dict(orient='index')[client_idx]
    response = requests.post('http://127.0.0.1:8000/predict', json=client_features)
    prediction = response.json()
    return prediction

@app.callback(
    Output('predicted-target', 'children'),
    Input('intermediate-value', 'data'))
def show_pred_result(prediction):
        pred, proba = prediction['prediction'], prediction['probability']
        message_dict = {
            0:'No default',
            1:'Default'
        }
        return '{} with {:.0f}% probability'.format(message_dict[pred], 100*proba)

@app.callback(
    Output('xaxis-column', 'options'),
    Output('xaxis-column', 'value'),
    Input('choice-dataset', 'value'))
def set_columns_options(selected_dataset):
    columns = data_dict[selected_dataset].columns 
    return columns, columns[0]

@app.callback(
    Output('color-selection', 'options'),
    Output('color-selection', 'value'),
    Input('choice-dataset', 'value'))
def set_columns_options(selected_dataset):
    df = data_dict[selected_dataset]
    columns = df.columns
    if selected_dataset != 'processed data':
        return [col for col in columns if df[col].nunique()<=2], columns[0]
    else:
        return [col for col in columns if df[col].dtype=='object'], columns[0]

@app.callback(
    Output('yaxis-column', 'options'),
    Output('yaxis-column', 'value'),
    Input('xaxis-column', 'value'),
    Input('choice-dataset', 'value')
    )
def set_columns_options(selected_var, selected_dataset):
    df = data_dict[selected_dataset]
    if selected_dataset != 'processed data':
        cat_cols = [col for col in df.columns if df[col].dtype=='object']
        num_cols = [col for col in df.columns if df[col].dtype!='object']
    else:
        cat_cols = [col for col in df.columns if df[col].nunique()<=2]
        num_cols = [col for col in df.columns if df[col].nunique()>2]

    if selected_var in cat_cols:    
        return num_cols, num_cols[0]
    else:
        columns = df.columns
        return columns, columns[0]

# @app.callback(
#     Output('customer-selection', 'options'),
#     Input('choice-dataset', 'value'))
# def set_client_ids(selected_dataset):
#     df = data_dict[selected_dataset]
#     if selected_dataset == 'previous credits monthly balance':
#         return df['SK_ID_BUREAU'].to_list()
#     else:
#         return df['SK_ID_CURR'].to_list()

@app.callback(
    Output('client-data', 'data'),
    Output('client-data', 'columns'),
    Input('customer-selection', 'value'),
    Input('choice-dataset', 'value'))
def display_client_data(selected_id, selected_dataset):
    
    if selected_dataset == 'previous credits monthly balance':
        bb_id = bureau.loc[bureau.SK_ID_CURR==selected_id, 'SK_ID_BUREAU']
        #data = bureau_balance[bureau_balance.SK_ID_BUREAU.isin(bb_id)].T.reset_index()
        data = bureau_balance[bureau_balance.SK_ID_BUREAU.isin(bb_id)]
    else:
        df = data_dict[selected_dataset]
        #data = df[df.SK_ID_CURR==selected_id].T.reset_index()
        data = df[df.SK_ID_CURR==selected_id]

    #data.columns = ['Feature', 'Value']

    return data.to_dict('records'), [{"name": i, "id": i} for i in data.columns]


@app.callback(
    Output('feat-importances', 'figure'),
    Input('customer-selection', 'value'),
    Input('intermediate-value', 'data')
    )
def update_feat_importances(client_id, prediction):
    
    return waterfall_plot(client_id, prediction)

@app.callback(
    Output('credit-default', 'figure'),
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('color-selection', 'value'),
    Input('customer-selection', 'value'),
    Input('choice-dataset', 'value'),
    Input('viz-type', 'value')
    )
def update_graph(xaxis_column_name,
yaxis_column_name,
color_sel,
selected_id,
selected_dataset,
viz_type
):
    # if df_type == 'raw data':
    #     d = raw_df
    # else:
    #     d = df

    d = data_dict[selected_dataset]
    if selected_dataset == 'previous credits monthly balance':
        bb_ids = bureau.loc[bureau.SK_ID_CURR==selected_id, 'SK_ID_BUREAU']
        #data = bureau_balance[bureau_balance.SK_ID_BUREAU.isin(bb_id)].T.reset_index()
        client_data = d[d.SK_ID_BUREAU.isin(bb_ids)]
    else:
        client_data = d[d.SK_ID_CURR==selected_id]
    
    # Add traces
    if viz_type == 'scatter':
        fig1 = px.scatter(d, x=xaxis_column_name, y=yaxis_column_name, color=color_sel, opacity=0.5)

    elif viz_type == 'box':
        fig1 = px.box(d, x=xaxis_column_name, y=yaxis_column_name, color=color_sel)

    fig2 = px.scatter(client_data, x=xaxis_column_name, y=yaxis_column_name, color=color_sel)
    fig2.update_traces({'marker_symbol':'star', 'marker_size':15, 'marker_color':'green'})
    
    fig1.add_trace(fig2.data[0])
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig1.update_xaxes(title_text=xaxis_column_name)
    fig1.update_yaxes(title_text=yaxis_column_name, showgrid=True, gridwidth=1, gridcolor='Lightgrey')

    return fig1


if __name__ == '__main__':
    app.run_server(debug=True)