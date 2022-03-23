from dash import Dash, dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
import json
import joblib
import time
from contextlib import contextmanager

import requests
import re


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, external_stylesheets])
server = app.server
app.title = 'Home Credit Default Dashboard'



with open('config.json', 'r') as f:
    CONFIG = json.load(f)   

QUERY_URL = CONFIG["QUERY_URL_LOCAL"]
PATH = CONFIG["PATH"]

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Load feature definitions
# load column definitions
hc_columns_definitions = pd.read_csv('../HomeCredit_columns_description.csv')[['Row', 'Description']]
def return_column_definition(column:str):
    
    group = r'(MEAN|MIN|MAX|VAR|SUM|SIZE|BURO|PREV|ACTIVE|CLOSED|APPROVED|REFUSED|POS|INSTAL|CC)'
    root = re.sub(group+'_?|_?'+group+'$',"",column)
    special_cols_def = {
             'DAYS_EMPLOYED_PERC' :'DAYS_EMPLOYED / DAYS_BIRTH',
             'INCOME_CREDIT_PERC' : 'AMT_INCOME_TOTAL / AMT_CREDIT',
             'INCOME_PER_PERSON' : 'AMT_INCOME_TOTAL / CNT_FAM_MEMBERS',
             'ANNUITY_INCOME_PERC' : 'AMT_ANNUITY / AMT_INCOME_TOTAL',
             'PAYMENT_RATE' : 'AMT_ANNUITY / AMT_CREDIT',
             'APP_CREDIT_PERC' : 'AMT_APPLICATION / AMT_CREDIT',
             'PAYMENT_PERC' : 'AMT_PAYMENT / AMT_INSTALMENT',
             'PAYMENT_DIFF' : 'AMT_INSTALMENT - AMT_PAYMENT',
             'DPD' : 'DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT',
             'DBD' : 'DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT',
             'COUNT' : 'Number of installments accounts'
         }
    
    if root in special_cols_def.keys():
        return special_cols_def[root]
    
    elif root in hc_columns_definitions.Row.to_list():
        description = hc_columns_definitions.loc[hc_columns_definitions['Row']==root, 'Description'].to_numpy()[0]
        return description
    
    else:
        col_name = '_'.join(root.split('_')[:-1])
        return return_column_definition(col_name)


# 3. Load data
with timer('Loading loan application...'):
    app_train = pd.read_csv(PATH+'application_train.csv')
with timer('Loading previous credits (raw)...'):
    bureau = pd.read_csv(PATH+'bureau.csv')
with timer('Loading previous credits monthly balance...'):
    bureau_balance = pd.read_csv(PATH+'bureau_balance.csv')
with timer('Loading previous applications...'):
    previous_app = pd.read_csv(PATH+'previous_application.csv')
with timer('Loading previous POS & card loans monthly balance...'):
    pos_cash = pd.read_csv(PATH+'POS_CASH_balance.csv')
with timer('Loading repayment history...'):
    installment_payments = pd.read_csv(PATH+'installments_payments.csv')
with timer('Loading previous credit card monthly balance...'):
    credit_card_balance = pd.read_csv(PATH+'credit_card_balance.csv')
with timer('Loading processed data'):
    processed_data = pd.read_csv(PATH+'features_test.csv')

# Les index récupérables sont restreints aux clients sur lesquels on applique le modèle
client_ids = processed_data['SK_ID_CURR'].sort_values().to_list()

data_dict = {
    "loan application (raw)":app_train.drop('TARGET', axis=1),
    'previous credits (raw)':bureau,
    'previous credits monthly balance': bureau_balance,
    'previous applications': previous_app,
    'previous POS & card loans monthly balance': pos_cash,
    'repayment history': installment_payments,
    'previous credit card monthly balance': credit_card_balance,
    'processed data': processed_data
}

feats = data_dict['processed data'].drop(['SK_ID_CURR'], axis=1).columns
model = joblib.load(open(PATH+'lgb.pkl', "rb"))
response = requests.get(QUERY_URL+'base_value')
base_value = response.json()

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

def waterfall_plot(shap_values, prediction):
    """Affichage des feature imortances locales par watefall plot. """
    shap_df = pd.DataFrame(data=shap_values, columns=feats)
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

    proba = prediction['probability']
    
    fig = go.Figure(go.Waterfall(
        x = idx1.append(idx2),
        textposition = "outside",
        text = text1,
        base = base_value,
        y = importances,
        connector = {"line":{"dash":"dot","width":1, "color":"rgb(63, 63, 63)"}},
        name = 'Predicted probability of default = {:.2f}'.format(proba),
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
    ymin = min(np.concatenate([[base_value], shap_cumsum]))
    ymax = max(np.concatenate([[base_value], shap_cumsum]))
    ylim = [ymin*0.9, ymax*1.1]

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


customer_selection = dcc.Dropdown(
        id="customer-selection",
        options=client_ids,
        value=client_ids[0],
        style={'width':'70%', 'height':'40px'}
    )

customer_input_group = dbc.InputGroup(
    [
        dbc.InputGroupText('Client ID'),
        customer_selection
    ],
    size='md', style={'padding-top':'10px'}
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
                                html.H4('Select Client to display'),
                                customer_input_group
                            ],  
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.H4('Model prediction', style={"text-align": "center"}),
                                html.H2(id="predicted-target", style={"text-align": "center"})
                            ], 
                            md=3
                        ),
                        dbc.Col(
                            [
                                html.H4(id="prediction-viz-title", style={}),
                                dcc.Graph(
                                    id="prediction-indicator",
                                ),
                                
                            ], align="center", md=6
                        ),   
                    ], align='start'
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
                            style_table={'overflowX': 'auto', 'height': "auto"},
                            style_cell={
                                'height': 'auto',
                                # all three widths are needed
                                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                'whiteSpace': 'normal'
                            },
                            fixed_rows={'headers': True},   
                            id='client-data'
                        ), align='stretch'
                    ),
                )
            ]
        )
    ]
)

xaxis_selection = dcc.Dropdown(
    value='EXT_SOURCE_3',
    id='xaxis-column',
    options=[col for col in processed_data.columns if col not in ['SK_ID_CURR']],
    style={"font-size":'85%', 'width':'100%'}
    )

yaxis_selection = dcc.Dropdown(
    value='AMT_INCOME_TOTAL',
    id='yaxis-column',
    options=[col for col in processed_data.columns if col not in ['SK_ID_CURR']],
    style={"font-size":'85%', 'width':'100%'}
    )

viz_type = dcc.RadioItems(
    options=['box', 'scatter', 'histogram'],
    value='histogram',
    id='viz-type',
    style={"font-size":'85%', 'width':'100%'},
    labelStyle={'display': 'block'},
    inputStyle={"margin-right": "20px"}
)
cols = [col for col in processed_data.columns if processed_data[col].nunique()<=2]
grouping_selection = dcc.Dropdown(
    options=cols,
    value=cols[0],
    id='grouping-selection',
    style={"font-size":'85%', 'width':'100%'}
    )

viz_card = dbc.Card(
    [
        dbc.CardHeader("Explore selected client features"),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        html.Div("Select X axis", style={'align':'center'}),
                                        xaxis_selection,
                                        html.Div(id='xaxis-definition', style={'font-size':'12px'}),
                                    ], style={
                                    #    'padding-top':'30px', 
                                    'padding-bottom':'20px'}
                                ),
                                dbc.Row(
                                    [
                                        html.Div("Select Y axis", style={'align':'canter'}),
                                        yaxis_selection,
                                        html.Div(id='yaxis-definition', style={'font-size':'12px'}),
                                    ], style={'padding-bottom':'20px'}
                                ), 
                                dbc.Row(
                                    [
                                        html.Div("Additional grouping variable", style={'align':'center'}),
                                        grouping_selection,
                                        html.Div(id='groupsel-definition', style={'font-size':'12px'}),
                                    ], style={'padding-bottom':'20px'}
                                ),
                                dbc.Row(
                                    [
                                        html.Div("Select type of plot", style={'align':'center'}),
                                        viz_type
                                    ]
                                ),
                            ], md=2, style={'padding-left':'30px', 'border-right':'2px solid #cccccc'}
                        ),
                        dbc.Col(
                                dcc.Graph(
                                    id="credit-default",
                                    style={"height": "450px"}
                                ),
                                md=6, style={'border-right':'2px solid #cccccc'}
                        ),
                        dbc.Col(
                            dash_table.DataTable(
                                style_table={'overflowY': 'auto', 'width': 'auto', 'height':'350px'},
                                style_cell={
                                    'height': 'auto',
                                    # all three widths are needed
                                    'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                    'whiteSpace': 'normal'
                                    },
                                    fixed_rows={'headers': True},   
                                    id='most-important-values'
                            ), 
                            md=4,
                            #style={'padding-top':'50px'},
                        )
                    ], align='center'
                ),
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
                style={'height':'600px'},
                figure=fig
            )
        )
    ]
)

app.layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Store(id='intermediate-value'),
        dcc.Store(id='intermediate-value-shap'),
        html.H1('Home Credit Default Dashboard'),
        html.Hr(),
        dbc.Tabs([
            dbc.Tab([
                dbc.Row(
                    [
                        dbc.Col(show_prediction_card)
                    ], style={'margin-bottom':'20px'}
                ),
                dbc.Row(
                    [
                        dbc.Col(viz_card),
                        html.Br()
                    ], 
                    style={'margin-bottom':'20px'}
                ), 
                dbc.Row(
                    [
                        dbc.Col(global_feat_importance_card, md=6),
                        dbc.Col(feat_importance_card, md=6)
                    ], style={'margin-bottom':'20px'}
                )
            ], label="Data Visualisation"),
            dbc.Tab(
                [
                    dbc.Row(
                        [
                            client_features_card
                        ]
                    ),
                ], label="Data tables"
            )
        ]),
    ]
)    

@app.callback(
    Output('intermediate-value', 'data'),
    Input('customer-selection', 'value'))
def fetch_api_response(selected_id):
    df = data_dict['processed data']
    client_idx = df[df.SK_ID_CURR==selected_id].index[0]
    jsnified_client_features = json.dumps(df[df.SK_ID_CURR==selected_id].to_dict(orient='index')[client_idx], allow_nan=True)
    response = requests.post(QUERY_URL+'predict', data=jsnified_client_features)
    prediction = response.json()
    print(prediction)
    return prediction

@app.callback(
    Output('intermediate-value-shap', 'data'),
    Input('customer-selection', 'value'))
def fetch_api_shap(selected_id):
    df = data_dict['processed data']
    client_idx = df[df.SK_ID_CURR==selected_id].index[0]
    response = requests.post(QUERY_URL+'shap_values', json=[int(client_idx)])
    shap_values = np.asarray(response.json())
    return shap_values

@app.callback(
    Output('predicted-target', 'children'),
    Input('intermediate-value', 'data'))
def show_pred_result(prediction):
        proba = prediction['probability']
        return '{:.1f}%'.format(100*proba)

@app.callback(
    Output('xaxis-definition', 'children'),
    Input('xaxis-column', 'value'))
def display_column_definition(selected_column):
    text = return_column_definition(selected_column)
    return "Definition: "+text

@app.callback(
    Output('yaxis-definition', 'children'),
    Input('yaxis-column', 'value'))
def display_column_definition(selected_column):
    try:
        text = return_column_definition(selected_column)
    except:
        return ""
    return "Definition: "+text


@app.callback(
    Output('groupsel-definition', 'children'),
    Input('grouping-selection', 'value'))
def display_column_definition(selected_column):
    text = return_column_definition(selected_column)
    return "Definition: "+text


@app.callback(
    Output('yaxis-column', 'options'),
    Output('yaxis-column', 'value'),
    Input('xaxis-column', 'value'),
    )
def set_columns_options(selected_var, 
):
    df = processed_data.drop(['SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV'], axis=1, errors='ignore')
    cat_cols = [col for col in df.columns if df[col].nunique()<=2]
    num_cols = [col for col in df.columns if df[col].nunique()>2]

    if selected_var in cat_cols:    
        return num_cols, num_cols[0]
    else:
        columns = df.columns
        return columns, columns[0]


@app.callback(
    Output('client-data', 'data'),
    Output('client-data', 'columns'),
    Input('customer-selection', 'value'),
    Input('choice-dataset', 'value'))
def display_client_data(selected_id, selected_dataset):
    
    if selected_dataset == 'previous credits monthly balance':
        bureau_df = data_dict['previous credits (raw)'] 
        bb_id = bureau_df.loc[bureau_df.SK_ID_CURR==selected_id, 'SK_ID_BUREAU']
        del bureau_df
        bb_df = data_dict['previous credits monthly balance']
        data = bb_df[bb_df.SK_ID_BUREAU.isin(bb_id)].drop(['SK_ID_BUREAU'], axis=1)
    else:
        df = data_dict[selected_dataset]
        data = df[df.SK_ID_CURR==selected_id].drop(['SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV'], axis=1, errors='ignore')

    return data.to_dict('records'), [{"name": i, "id": i} for i in data.columns]

@app.callback(
    Output('yaxis-column', 'disabled'),
    Input('viz-type', 'value')
)
def disable_dropdown(viz_type):
    if viz_type=='histogram':
        return True
    else:
        return False

@app.callback(
    Output('feat-importances', 'figure'),
    Input('intermediate-value-shap', 'data'),
    Input('intermediate-value', 'data'),
    )
def update_feat_importances(shap_values, prediction):
    return waterfall_plot(shap_values, prediction)

@app.callback(
    Output('credit-default', 'figure'),
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('grouping-selection', 'value'),
    Input('customer-selection', 'value'),
    Input('viz-type', 'value')
    )
def update_graph(xaxis_column_name,
yaxis_column_name,
hue,
selected_id,
viz_type
):

    d = processed_data
    client_data = d[d.SK_ID_CURR==selected_id].drop(['SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV'], axis=1, errors='ignore')
    if hue:
        # Add traces
        if viz_type == 'scatter':
            fig1 = px.scatter(d, x=xaxis_column_name, y=yaxis_column_name, color=hue, opacity=0.5)
            fig1.update_yaxes(title_text=yaxis_column_name, showgrid=True, gridwidth=1, gridcolor='Lightgrey')

        elif viz_type == 'box':
            fig1 = px.box(d, x=xaxis_column_name, y=yaxis_column_name, color=hue)
            fig1.update_yaxes(title_text=yaxis_column_name, showgrid=True, gridwidth=1, gridcolor='Lightgrey')
        
        elif viz_type == 'histogram':
            fig1 = px.histogram(d, x=xaxis_column_name, color=hue, barmode='group')
            fig1.update_yaxes(title_text='Count', showgrid=True, gridwidth=1, gridcolor='Lightgrey')
        
        if not client_data.empty:
            fig2 = px.scatter(client_data, x=xaxis_column_name, y=yaxis_column_name, color=hue)
            fig2.update_traces({'marker_symbol':'star', 'marker_size':15, 'marker_color':'green'})
            fig1.add_trace(fig2.data[0])
    else:
        # Add traces
        if viz_type == 'scatter':
            fig1 = px.scatter(d, x=xaxis_column_name, y=yaxis_column_name, opacity=0.5)
            fig1.update_yaxes(title_text=yaxis_column_name, showgrid=True, gridwidth=1, gridcolor='Lightgrey')

        elif viz_type == 'box':
            fig1 = px.box(d, x=xaxis_column_name, y=yaxis_column_name)
            fig1.update_yaxes(title_text=yaxis_column_name, showgrid=True, gridwidth=1, gridcolor='Lightgrey')
        
        elif viz_type == 'histogram':
            fig1 = px.histogram(d, x=xaxis_column_name)
            fig1.update_yaxes(title_text='Count', showgrid=True, gridwidth=1, gridcolor='Lightgrey')
        
        if not client_data.empty:
            fig2 = px.scatter(client_data, x=xaxis_column_name, y=yaxis_column_name)
            fig2.update_traces({'marker_symbol':'star', 'marker_size':15, 'marker_color':'green'})
            fig1.add_trace(fig2.data[0])
    
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig1.update_xaxes(title_text=xaxis_column_name)
    

    return fig1

@app.callback(
    Output('most-important-values', 'data'),
    Output('most-important-values', 'columns'),
    Input('customer-selection', 'value'),
    Input('intermediate-value-shap', 'data')
    )
def display_client_data(selected_id, shap_values):
    
    df = data_dict['processed data']
    shap_df = pd.DataFrame(data=shap_values, columns=feats)
    high_importance_features_index = abs(shap_df.iloc[0].values).argsort()[::-1][:10]
    high_importance_features = df.columns[high_importance_features_index]
    data = df.loc[df.SK_ID_CURR==selected_id, high_importance_features].T.reset_index()
    data.columns = ["Feature", "Current Value"]
    return data.to_dict('records'), [{"name": i, "id": i} for i in data.columns]

@app.callback(
    Output('prediction-indicator', 'figure'),
    Input('intermediate-value', 'data')
    )
def plot_prediction(prediction):
    proba = prediction['probability']
    color = "green" if proba < 0.36 else "orange" if 0.36<=proba<0.64 else "red"
    fig = go.Figure(go.Indicator(
        mode = "number+gauge", 
        value = 100*proba,
        number = {'font':{'size':18}, 'valueformat':'.1f', 'suffix':'%'},
        domain = {'x': [.25, 1], 'y': [0, 1]},
        gauge = {
            'shape': "bullet",
            'axis': {'range': [None, 100], 'tick0':0, 'dtick':10},
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 0.75,
                'value': 36},
            'bar': {'color': color},
        }
            ))
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=100
    #title={'text':"<b>Predicted probability of default</b>", 'x':0.25}
    )

    return fig

@app.callback(
    Output('prediction-viz-title', 'children'),
    Output('prediction-viz-title', 'style'),
    Input('intermediate-value', 'data')
    )
def prediction_title(prediction):
    
    proba = prediction["probability"]
    if proba < 0.36:
        return "No risk of default", {'text-align': 'center', 'color':'green'}
    elif 0.36 <= proba < 0.64:
        return "Substantial risk of default", {'text-align': 'center', 'color':'orange'}
    else:
        return "Very high risk of default", {'text-align': 'center', 'color':'red'}

if __name__ == '__main__':
    app.run_server(debug=True)