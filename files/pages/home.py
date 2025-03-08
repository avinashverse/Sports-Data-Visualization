import pandas as pd
import dash
from dash import dcc, html, callback
import plotly.express as px
from dash.dependencies import Input, Output

dash.register_page(__name__, path='/', name="home")

layout = html.Div([
        html.Div(id='welcome', children=[
            html.Div(id='text', children=[
                html.H1('Illuminate Sports:'),
                html.H2('Unveiling Insights,'),
                html.H2('Safeguarding Success,'),
                html.H2('Empowering Champions.'),
                html.H4('Interactive Visualizations')
            ]),
            html.Div(id='photo', children=[
                html.Img(src="/assets/sports2.png")
            ])
        ]),
        html.Div(id='next', children=[
            html.Div(className='line'),
            html.P(
                "At Global Sports DataVista, we provide illuminating insights in the world of sports. We are dedicated to unveiling the hidden stories behind the numbers, safeguarding success through data-driven strategies, and empowering champions with actionable analytics. "
                "Whether it's the thrill of IPL, the excitement of FIFA, or the grandeur of the Olympics, we delve deep into the data to uncover valuable insights. "
                "Join us on this journey of discovery and transformation in the realm of global sports."
            ),
            html.A(html.Button('Explore'), href='#sections1'),
            html.Div(className='line')

        ]),
        html.Div(id='last',
                 children=[
                    html.Div(className='sections', id='sections1',
                             children=[
                                 html.Div(html.Img(src="assets/olympics.png"), className='imgdiv'),
                                 html.H4("Olympics"),
                                 html.Div(html.A(html.Button('Visualize'), href='/Olympics-page1'),className='buttons'),
                                
                             ]),
                    html.Div(className='sections', id='sections2',
                             children=[
                                 html.Div(html.Img(src="assets/World Cup logo.png",id="fifaimg"), className='imgdiv'),
                                 html.H4("FIFA World Cup"),
                                 html.Div(html.A(html.Button('Dashboard'), href='/fifa-dashboard'),className='buttons'),
                                
                             ]),
                    html.Div(className='sections', id='sections3',
                             children=[
                                 html.Div(html.Img(src="assets/ipl.png",id="iplimg"), className='imgdiv'),
                                 html.H4("Indian Premier League"),
                                 html.Div(children=[html.Div(html.A(html.Button('Dashboard 1'), href='/ipl-team-dashboard'),className='buttons'),
                                 html.Div(html.A(html.Button('Dashboard 2'), href='/ipl-player-dashboard'),className='buttons'),
                                 html.Div(html.A(html.Button('Visualize'), href='/ipl-page'),className='buttons')],className='buttonsdiv'),
                                 
                             ]),
        ]),
])
