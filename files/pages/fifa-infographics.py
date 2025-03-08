import pandas as pd
import dash
from dash import dcc, html, callback
import plotly.express as px
from dash.dependencies import Input, Output

dash.register_page(__name__, path='/fifa-infographics', name="fifa-infographics")

layout = html.Div([
    html.H1("Infographics"),
    # Add your content here
])
