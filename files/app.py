import pandas as pd
import dash
from dash import dcc, html, callback
import plotly.express as px
from dash.dependencies import Input, Output

app = dash.Dash(__name__, suppress_callback_exceptions=True, pages_folder='pages', use_pages=True, external_stylesheets=[
    '/assets/style.css','/assets/fifa.css','/assets/olympics_explore.css','assets/ipl_teams.css','assets/ipl_players.css','assets/ipl_page.css',
    'https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300&family=Jost:ital,wght@0,100..900;1,100..900&family=Roboto:ital,wght@0,100;0,300;0,400;0,500;0,700;0,900;1,100;1,300;1,400;1,500;1,700;1,900&family=Source+Serif+4:ital,opsz,wght@0,8..60,200..900;1,8..60,200..900&display=swap',
    
])

# Get page names and relative paths
# page_info = [{'name': page['name'], 'relative_path': page['relative_path']} for page in dash.page_registry.values()]
# print(page_info)
app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.H2('Global Sports DataVista')
            ], id='h-one'),
            html.Div([
                html.Div(className='line'),
                html.Div(
                    className='navbar',
                    children=[
                        html.Ul(
                            children=[
                                html.Li(
                                    children=[
                                        html.Div(
                                            className='dropdown',
                                            children=[
                                                # The dropdown button
                                                html.Button(
                                                    className='dropdown-btn',
                                                    children=[
                                                        dcc.Link('Home', href='/')
                                                    ]
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                html.Li(
                                    children=[
                                        html.Div(
                                            className='dropdown',
                                            children=[
                                                # The dropdown button
                                                html.Button('Olympics', className='dropdown-btn'),
                                                # Dropdown content
                                                html.Div(
                                                    className='dropdown-content',
                                                    children=[
                                                        html.A('Visual Journey', href='/Olympics-page1'),
                                                 
                                                    ]
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                html.Li(
                                    children=[
                                        html.Div(
                                            className='dropdown',
                                            children=[
                                                # The dropdown button
                                                html.Button('FIFA World Cup', className='dropdown-btn'),
                                                # Dropdown content
                                                html.Div(
                                                    className='dropdown-content',
                                                    children=[
                                                        html.A('Dashboard', href='/fifa-dashboard'),
                    
                                              
                                                    ]
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                html.Li(
                                    children=[
                                        html.Div(
                                            className='dropdown',
                                            children=[
                                                # The dropdown button
                                                html.Button('Indian Premier League (IPL)', className='dropdown-btn'),
                                                # Dropdown content
                                                html.Div(
                                                    className='dropdown-content',
                                                    children=[
                                                        html.A('Teams Dashboard', href='/ipl-team-dashboard'),
                                                        html.A('Players Dashboard', href='/ipl-player-dashboard'),
                                                        html.A('Visual Journey', href='/ipl-page'),
                                                     
                                                    ]
                                                )
                                            ]
                                        )
                                    ]
                                ),
                            ]
                        )
                    ]
                )
                ,
                    html.Div(className='line'),
            ], id='h-two'),
        ], id='header'),
        dash.page_container
    ])
])


if __name__ == "__main__":
    app.run_server(debug=True)
