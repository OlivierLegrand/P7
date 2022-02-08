# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import json
import pandas as pd



df = pd.read_csv('../data/filled_data.csv')
with open('../main_features.json', 'r') as f:
    main_features = json.load(f)

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in main_features])
        ),
        html.Tbody([
            html.Tr([
                html.Td(round(dataframe.iloc[i][col], 2)) for col in main_features
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

fig = fig = px.histogram(df, x="DAYS_EMPLOYED_PERC", color="TARGET",
                   hover_data=df.columns)


app = Dash(__name__)

app.layout = html.Div([
    html.H4(children='HomeCredit Default Risk'),
    generate_table(df),
    dcc.Graph(
        id='life-exp-vs-gdp',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
