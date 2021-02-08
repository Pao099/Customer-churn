import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import plotly.express as px
import plotly as py
import plotly.graph_objs as go
import warnings
import dash
# import dash_auth
import dash_core_components as dcc
import dash_html_components as html
warnings.filterwarnings('ignore')
from IPython.display import IFrame
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
import dash
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
# app = dash.Dash(__name__)
# app=JupyterDash(__name__)
#Read in data file, bankchurners.csv
data=pd.read_csv('BankChurners.csv')

#Data Cleaning
#Drop unwanted columns
df=data.drop(['CLIENTNUM','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
         'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], errors='ignore',axis=1)
df2=pd.read_csv('BankChurners2.csv')
df3= df2.iloc[:,1:]
#Separate data into categorical, continuous and discreet numeric data types for ease of identification and analysis
cat_data=df.select_dtypes('object').columns.to_list()
cont_data=df.select_dtypes('float64').columns.to_list()
disc_data=df.select_dtypes('int64').columns.to_list()
all_data=df.columns.to_list()
all_data2=df3.columns.to_list()

app.title = 'Customer Churn Prediction'
app.layout = html.Div([
    html.Div([
        html.H2("Bank Customer Attrition Data Representation",
                style={'color': 'blue', 'font-style': 'regular', 'font-weight': 'bold'}),
        html.Div([html.H3("Scatterplots", style={'color': 'purple', 'font-style': 'regular', 'font-weight': 'bold'}),
                  html.P("Select from the list below to display a scatterplot of correlations:"),
                  html.H5("All Data"),
                  dcc.Dropdown(
                      id='contdata',
                      value=all_data[11],
                      options=[{'label': x, 'value': x}
                               for x in all_data],
                      multi=False,
                      clearable=False
                  ), dcc.Dropdown(
                id='contdata2',
                value=all_data[12],
                options=[{'label': x, 'value': x}
                         for x in all_data],
                multi=False,
                clearable=False
            ),
                  dcc.Graph(id="scatterplot")]),
        html.Div([html.H3("Box Plots and Histograms",
                          style={'color': 'purple', 'font-style': 'regular', 'font-weight': 'bold'}),
                  html.P("Data Columns:"),
                  html.P("Select from the list below to display the Histogram and Box plots:"),
                  dcc.Dropdown(
                      id='alldata',
                      value=all_data[0],
                      options=[{'label': x, 'value': x}
                               for x in all_data],
                      multi=False,
                      clearable=False
                  ), dcc.Graph(id="histogram")]),

        html.Div([html.H4("Correlations Heatmap: Factor Correlation with Attrition",
                          style={'color': 'purple', 'font-style': 'regular', 'font-weight': 'bold'}),
                  html.Div(dcc.Graph(id='heatmap', figure=px.imshow(df3.corr(), title="Heatmap")))
                  ]),
        html.Div([html.H4("Barcharts: Display Correlation with Attrition",
                          style={'color': 'purple', 'font-style': 'regular', 'font-weight': 'bold'}),
                  dcc.Graph(id='barplots',
                            figure=px.bar(df3.corr()['Attrition_Flag'].sort_values(ascending=False), title="Barplots"))
                  ]),
        html.Div([html.H4("Representative Pie Chart:",
                          style={'color': 'purple', 'font-style': 'regular', 'font-weight': 'bold'}),
                  html.H6("Red represents churned customers, Blue Represents existing customers"),
                  dcc.Graph(id='piechart',
                            figure=px.pie(df, values=df.Attrition_Flag.value_counts().values, labels=df.Attrition_Flag,
                                          title="Pie Chart"))

                  ])
    ])

])


@app.callback(
    dash.dependencies.Output("histogram", "figure"),
    [dash.dependencies.Input("alldata", "value")])
def histogram(prop):
    fig = px.histogram(df, x=prop, color="Attrition_Flag", marginal="box", hover_data=df.columns,
                       title=f"Histogram of {prop}")
    return fig


@app.callback(
    dash.dependencies.Output("scatterplot", "figure"),
    [dash.dependencies.Input("contdata", "value")], [dash.dependencies.Input("contdata2", "value")])
def scatterplot(v1, v2):
    fig = px.scatter(
        df, x=v1, y=v2, color="Attrition_Flag",
        color_continuous_scale="Attrition_Flag",
        render_mode="webgl", title=f"Scatterplot of {v2} vs {v1}"
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False, mode='external')