from curses import raw
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

import pandas as pd
import p7
import lightgbm_with_simple_features as lgbmsf
import json
app = Dash(__name__)

with open('./config.json', 'r') as f:
    CONFIG = json.load(f)

PATH = CONFIG["PATH"]
NUM_ROWS = CONFIG["NUM_ROWS"]

_, raw_test_df, _, raw_target_test_df = p7.prepare_data(num_rows=NUM_ROWS, raw=True, perc_filled=0.8)
raw_df = pd.concat([raw_test_df, raw_target_test_df], axis=1)
print('raw data created')

_, test_df, _, target_test_df =  p7.prepare_data(num_rows=NUM_ROWS, raw=False, perc_filled=0.8)
df = pd.concat([test_df, target_test_df], axis=1)
print('processed data created')

cat_cols = [col for col in raw_df.columns if raw_df[col].dtype=='object']
raw_num_cols = [col for col in raw_df.columns if raw_df[col].dtype!='object']
num_cols = [col for col in df.columns if df[col].nunique()>2]

app.layout = html.Div([
    html.Div([
        dcc.RadioItems(
            ['raw', 'processed'],
            'raw',
            id='df-type',
        ),

        dcc.Dropdown(
            id='xaxis-column'
        ),

    html.Hr(),
    
    ], style={'width': '48%', 'display': 'inline-block'}),

        

        # html.Div([
        #     dcc.Dropdown(
        #         raw_df.columns,
        #         'TARGET',
        #         id='yaxis-column'
        #     ),
        #     dcc.RadioItems(
        #         ['Linear', 'Log'],
        #         'Linear',
        #         id='yaxis-type',
        #         inline=True
        #     )
        # ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    dcc.Graph(id='credit_default'),

    #dcc.Slider(
    #    df['Year'].min(),
    #    df['Year'].max(),
    #    step=None,
    #    id='year--slider',
    #    value=df['Year'].max(),
    #    marks={str(year): str(year) for year in df['Year'].unique()},
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
    Output('credit_default', 'figure'),
    Input('xaxis-column', 'value'),
    #Input('yaxis-column', 'value'),
    #Input('xaxis-type', 'value'),
    #Input('yaxis-type', 'value'),
    Input('df-type', 'value')
    )
def update_graph(xaxis_column_name, df_type, #yaxis_column_name,
                 #xaxis_type, #yaxis_type,
                 #year_value
                 ):
    #dff = df[df['Year'] == year_value]
    if df_type == 'raw':
        data = raw_df
    else:
        data=df
    
    fig = px.histogram(data, x=xaxis_column_name, color="TARGET",
                   marginal="box", # or violin, rug
                   hover_data=data.columns,
                   histnorm='percent',
                   barmode='group')

    
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    #fig.update_xaxes(title=xaxis_column_name,
    #                 type='linear' if xaxis_type == 'Linear' else 'log')

    #fig.update_yaxes(type='linear' if yaxis_type == 'Linear' else 'log')

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)