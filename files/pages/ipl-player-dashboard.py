import pandas as pd
import dash
import numpy as np
from dash import dcc, html, callback
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

dash.register_page(__name__, path='/ipl-player-dashboard', name="ipl-player-dashboard")
ball = pd.read_csv('assets/deliveries.csv')
match = pd.read_csv('assets/matches.csv')
match.loc[197, 'MatchNumber'] = 'Qualifier 1'
match.loc[195, 'MatchNumber'] = 'Qualifier 2'

match.loc[257, 'MatchNumber'] = 'Qualifier 1'
match.loc[255, 'MatchNumber'] = 'Qualifier 2'
match['MatchNumber'] = match['MatchNumber'].str.replace('Elimination Final', 'Eliminator')
match = match[(match['MatchNumber'] != '3rd Place Play-Off') & (match['MatchNumber'] != 'Semi Final')]
df = pd.merge(ball, match, on='ID')
df['Team1'] = df['Team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
df['Team2'] = df['Team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
df['WinningTeam'] = df['WinningTeam'].str.replace('Delhi Daredevils', 'Delhi Capitals')
df['BattingTeam'] = df['BattingTeam'].str.replace('Delhi Daredevils', 'Delhi Capitals')


df['Team1'] = df['Team1'].str.replace('Kings XI Punjab', 'Punjab Kings')
df['Team2'] = df['Team2'].str.replace('Kings XI Punjab', 'Punjab Kings')
df['WinningTeam'] = df['WinningTeam'].str.replace('Kings XI Punjab', 'Punjab Kings')
df['BattingTeam'] = df['BattingTeam'].str.replace('Kings XI Punjab', 'Punjab Kings')

df['Team1'] = df['Team1'].str.replace('Rising Pune Supergiant', 'Rising Pune Supergiants')
df['Team2'] = df['Team2'].str.replace('Rising Pune Supergiant', 'Rising Pune Supergiants')
df['WinningTeam'] = df['WinningTeam'].str.replace('Rising Pune Supergiant', 'Rising Pune Supergiants')
df['BattingTeam'] = df['BattingTeam'].str.replace('Rising Pune Supergiant', 'Rising Pune Supergiants')

df['Team1'] = df['Team1'].str.replace('Rising Pune Supergiantss', 'Rising Pune Supergiants')
df['Team2'] = df['Team2'].str.replace('Rising Pune Supergiantss', 'Rising Pune Supergiants')
df['WinningTeam'] = df['WinningTeam'].str.replace('Rising Pune Supergiantss', 'Rising Pune Supergiants')
df['BattingTeam'] = df['BattingTeam'].str.replace('Rising Pune Supergiantss', 'Rising Pune Supergiants')

venue_replacements = {
    'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium',
    'Brabourne Stadium, Mumbai': 'Brabourne Stadium',
    'Dr DY Patil Sports Academy, Mumbai': 'Dr DY Patil Sports Academy',
    'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium',
    'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium',
    'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium',
    'Wankhede Stadium, Mumbai': 'Wankhede Stadium',
    'Maharashtra Cricket Association Stadium, Pune': 'Maharashtra Cricket Association Stadium',
    'Punjab Cricket Association IS Bindra Stadium': 'Punjab Cricket Association Stadium, Mohali',
    'Eden Gardens, Kolkata': 'Eden Gardens',
    'MA Chidambaram Stadium, Chennai': 'MA Chidambaram Stadium',
    'M Chinnaswamy Stadium': 'M.Chinnaswamy Stadium',
    'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association Stadium'
}

# Replacing stadium names in the venue data
for old_name, new_name in venue_replacements.items():
    df['Venue'] = df['Venue'].str.replace(old_name, new_name)

df['BowlingTeam'] = df.apply(lambda x: x['Team2'] if x['BattingTeam'] == x['Team1'] else x['Team1'], axis=1)
df['Season'] = df['Season'].str.replace('2007/08', '2008')
df['Season'] = df['Season'].str.replace('2009/10', '2010')
df['Season'] = df['Season'].str.replace('2020/21', '2020')

df['Toss Winner = Match Winner'] = df.apply(lambda row: 1 if row['TossWinner'] == row['WinningTeam'] else 0, axis=1)
# Calculate whether TossWinner was not the WinningTeam
df['Toss Winner = Match Loser'] = df.apply(lambda row: 1 if row['TossWinner'] != row['WinningTeam'] else 0, axis=1)
match['Season'] = match['Season'].str.replace('2007/08', '2008')
match['Season'] = match['Season'].str.replace('2009/10', '2010')
match['Season'] = match['Season'].str.replace('2020/21', '2020')
match['Team1'] = match['Team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match['Team2'] = match['Team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match['WinningTeam'] = match['WinningTeam'].str.replace('Delhi Daredevils', 'Delhi Capitals')


match['Team1'] = match['Team1'].str.replace('Kings XI Punjab', 'Punjab Kings')
match['Team2'] = match['Team2'].str.replace('Kings XI Punjab', 'Punjab Kings')
match['WinningTeam'] = match['WinningTeam'].str.replace('Kings XI Punjab', 'Punjab Kings')

match['Team1'] = match['Team1'].str.replace('Rising Pune Supergiant', 'Rising Pune Supergiants')
match['Team2'] = match['Team2'].str.replace('Rising Pune Supergiant', 'Rising Pune Supergiants')
match['WinningTeam'] = match['WinningTeam'].str.replace('Rising Pune Supergiant', 'Rising Pune Supergiants')

match['Team1'] = match['Team1'].str.replace('Rising Pune Supergiantss', 'Rising Pune Supergiants')
match['Team2'] = match['Team2'].str.replace('Rising Pune Supergiantss', 'Rising Pune Supergiants')
match['WinningTeam'] = match['WinningTeam'].str.replace('Rising Pune Supergiantss', 'Rising Pune Supergiants')

team_secondary_colors = {
    'Chennai Super Kings': '#1451A2',
    'Delhi Capitals': '#EC4310',
    'Gujarat Titans': '#1451A2',
    'Kolkata Knight Riders': '#FFEF2C',
    'Lucknow Super Giants': '#F91B02',
    'Mumbai Indians': '#C8B60E',
    'Punjab Kings': '#FFF5ED',
    'Rajasthan Royals': '#030072',
    'Royal Challengers Bangalore': '#1545ED',
    'Sunrisers Hyderabad': '#320300'
}
team_primary_colors = {
        'Chennai Super Kings': '#F0C015',
        'Delhi Capitals': '#323FA9',
        'Gujarat Titans': '#4E6579',
        'Kolkata Knight Riders': '#3C1582',
        'Lucknow Super Giants': '#303F79',
        'Mumbai Indians': '#0F2AC6',
        'Punjab Kings': '#FD0118',
        'Rajasthan Royals': '#FF0081',
        'Royal Challengers Bangalore': '#DA0001',
        'Sunrisers Hyderabad': '#E33307',
        'Deccan Chargers': '#7e8282',
        'Rising Pune Supergiants': '#03f4fc',
        'Gujarat Lions': '#fa4428',
        'Kochi Tuskers Kerala': '#6511ed',
        'Pune Warriors': '#36eef5'
}
player_pics = {
        'CH Gayle': 'assets/Players_photo/CH Gayle.png',
        'JJ Bumrah': 'assets/Players_photo/JJ Bumrah.png',
        'KL Rahul': 'assets/Players_photo/KL Rahul.png',
        'Mohammed Shammi': 'assets/Players_photo/Mohammed Shammi.png',
        'MS Dhoni': 'assets/Players_photo/MS Dhoni.png',
        'RG Sharma': 'assets/Players_photo/RG Sharma.png',
        'V Kohli': 'assets/Players_photo/V Kohli.png',
}
team_jersey = {
    'Chennai Super Kings': 'assets/Team_Jerseys/Chennai Super Kings.png',
    'Delhi Capitals': 'assets/Team_Jerseys/Delhi Capitals.png',
    'Gujarat Titans': 'assets/Team_Jerseys/Gujarat Titans.png',
    'Kolkata Knight Riders': 'assets/Team_Jerseys/Kolkata Knight Riders.png',
    'Lucknow Super Giants': 'assets/Team_Jerseys/Lucknow Super Giants.png',
    'Mumbai Indians': 'assets/Team_Jerseys/Mumbai Indians.png',
    'Punjab Kings': 'assets/Team_Jerseys/Punjab Kings.png',
    'Rajasthan Royals': 'assets/Team_Jerseys/Rajasthan Royals.png',
    'Royal Challengers Bangalore': 'assets/Team_Jerseys/Royal Challengers Bangalore.png',
    'Sunrisers Hyderabad': 'assets/Team_Jerseys/Sunrisers Hyderabad.png'
}
teams_present=team_jersey.keys()
def get_curr_team(player_name):
        player_df = df[(df['batter'] == player_name) | (df['bowler'] == player_name)]
        curr_team = player_df.drop_duplicates(subset=['Season']).head(1)

        return curr_team['BattingTeam'].values[0] if curr_team['batter'].values[0] == player_name else curr_team['BowlingTeam'].values[0]


player_names=np.union1d(df['batter'].unique(), df['bowler'].unique())
dropdown_options = [{'label': team, 'value': team} for team in player_names]

# Dropdown component
dropdown = dcc.Dropdown(
    id='player-dropdown',
    options=dropdown_options,
    value='V Kohli',  
    style={'color': 'black'} 
)






layout = html.Div(
        id='i-t-page',
        children=[
            html.Div(
                id='i-p-row1',
                children=[
                    html.Div(
                        id='i-p-box1',
                        className="boxes",
                        children=[
                          
                          html.H2(id='box1-heading'),
                          html.H1(id='box1-reading'),
                
                        ]
                    ),
                    html.Div(
                        id='i-p-box2',
                        className="boxes",
                        children=[
                          html.H2(id='box2-heading'),
                          html.H1(id='box2-reading'),
                
                        ]
                    ),
                    html.Div(
                        id='i-p-box3',
                        className="boxes",
                        children=[
                          html.H2(id='box3-heading'),
                          html.H1(id='box3-reading'),
                
                        ]
                    ),
                    html.Div(
                        id='i-p-box4',
                        className="boxes",
                        children=[
                          html.H2(id='box4-heading'),
                          html.H1(id='box4-reading'),
                
                        ]
                    ),
                    html.Div(
                        id='i-p-box5',
                        className="boxes",
                        children=[
                          html.H2(id='box5-heading'),
                          html.H1(id='box5-reading'),
                
                        ]
                    ),
                    
                    
                ]
            ),
            html.Div(
                id='i-p-row2',
                children=[
                    html.Div(
                        id='i-p-player',
                        children=[
                            html.H2("IPL Player Selector"),
                            dropdown,
                            html.H1(id='selected-player-name'), 
                            html.Div(id='player-img-display'),
                
                        ]
                    ),
                    html.Div(
                        id='i-p-next',
                        children=[
                            html.Div(
                                id='i-p-graph',
                                children=[
                                    html.Div(
                                        className='i-p-plot',
                                        children=[
                                            html.H1(id='i-p-runs-heading'),
                                            dcc.Graph(id='i-p-runs')
                                        ]
                                        
                                    )
                                ]
                            ),
                            html.Div(
                                id='i-p-graph',
                                children=[
                                    html.Div(
                                        className='i-p-plot',
                                        children=[
                                            html.H1(id='i-p-wickets-heading'),
                                            dcc.Graph(id='i-p-wickets')
                                  
                                        ]
                                        
                                    )
                                ]
                            ),
                        ]
                    ),
                    
                ]
            ),
            html.Div(
                id='i-p-row3',
                children=[
                    html.Div(
                        className='i-p-graph',id='i-p-graph-over',
                        children=[
                            html.Div(
                                className='i-p-plot',
                                children=[
                                    html.H1(id='i-p-over-heading'),
        
                                    dcc.Graph(id='i-p-over')
                                ]
                                
                            )
                        ]
                    ),
                    html.Div(
                        className='i-p-graph',id='i-p-graph-against',
                        children=[
                            html.Div(
                                className='i-p-plot',
                                children=[
                                    html.H1(id='i-p-against-heading'),
                                    html.H3(id='i-p-against-heading-smaller'),
                                    dcc.Graph(id='i-p-against')
                                ]
                                
                            )
                            
                        ]
                    ),
                    
                    
                ]
            ),
            html.Div(id="ending")
            
           
        ]
    )




@callback(
    [Output('box1-heading', 'children'),
     Output('box2-heading', 'children'),
     Output('box3-heading', 'children'),
     Output('box4-heading', 'children'),
     Output('box5-heading', 'children'),
     Output('box1-reading', 'children'),
     Output('box2-reading', 'children'),
     Output('box3-reading', 'children'),
     Output('box4-reading', 'children'),
     Output('box5-reading', 'children')],
    [Input('player-dropdown', 'value')]
)
def get_player_stats(input_value, df=df, match=match):
    player_name=input_value
    count_batter = df[df['batter'] == player_name].shape[0]
    count_bowler = df[df['bowler'] == player_name].shape[0]

    df['Date'] = pd.to_datetime(df['Date'])
    is_batter = count_batter > count_bowler

    if is_batter:
        batter_df = df[df['batter'] == player_name]
        runs_scored = batter_df['batsman_run'].sum()
        num_got_out = batter_df['isWicketDelivery'].sum()
        balls_faced = batter_df.shape[0]
        strike_rate = np.round(runs_scored * 100 / balls_faced, 4)
        average = np.round(runs_scored / num_got_out, 4)
        man_of_the_match = match[match['Player_of_Match'] == player_name].shape[0]
        max_score = batter_df.groupby('ID')['batsman_run'].sum().max()

        return 'Runs Scored','Strike Rate','Average','Man of the Match','Highest Score',runs_scored,strike_rate,average,man_of_the_match,max_score
        

    else:
        bowler_df = df[df['bowler'] == player_name]
        wickets_taken = bowler_df['isWicketDelivery'].sum()
        runs_conceded = bowler_df['total_run'].sum()
        balls_bowled = bowler_df.shape[0]
        economy = np.round(runs_conceded * 6 / balls_bowled, 4)
        bowling_average = np.round(runs_conceded / wickets_taken, 4)
        man_of_the_match = match[match['Player_of_Match'] == player_name].shape[0]

        # Find the best bowling figures
        match_stats = bowler_df.groupby('ID').agg({'isWicketDelivery': 'sum', 'total_run': 'sum'})
        match_stats = match_stats.rename(columns={'isWicketDelivery': 'wickets', 'total_run': 'runs'})
        best_figures = match_stats.loc[match_stats['wickets'].idxmax()]
        best_figures_str = f"{best_figures['wickets']}/{best_figures['runs']}"

        return 'Wickets Taken','Economy','Bowling Average','Man of the Match','Best Bowling Figures', wickets_taken,economy,bowling_average,man_of_the_match,best_figures_str
        

@callback(
    Output('selected-player-name', 'children'),
    [Input('player-dropdown', 'value')]
)
def update_selected_player_name(input_value):
    return f'{input_value}'

@callback(
    Output('i-p-runs-heading', 'children'),
    [Input('player-dropdown', 'value')]
)
def update_player_runs_name(input_value):
    count_batter = df[df['batter'] == input_value].shape[0]
    count_bowler = df[df['bowler'] == input_value].shape[0]

    isBatter = True if count_batter > count_bowler else False

    if isBatter:
        return f'Runs Scored by {input_value} in Each Season'
    else:
        return f'Wickets Taken by {input_value} in Each Season'

@callback(
    Output('i-p-runs', 'figure'),
    [Input('player-dropdown', 'value')]
)
def generate_player_stats_chart(input_value, df=df):
    player_name=input_value
    count_batter = df[df['batter'] == player_name].shape[0]
    count_bowler = df[df['bowler'] == player_name].shape[0]
    isBatter = count_batter > count_bowler
    
    if isBatter:
        batter_df = df[df['batter'] == player_name]
        runs_per_season = batter_df.groupby(['Season'])['batsman_run'].sum().reset_index()
        team_df = batter_df.drop_duplicates(subset=['Season'])[['Season', 'BattingTeam']].reset_index(drop=True)
        merged_df = runs_per_season.merge(team_df, on='Season', how='left')

        trace = go.Bar(
            x=merged_df['Season'],
            y=merged_df['batsman_run'],
            marker_color=[team_primary_colors.get(team, '#808080') for team in merged_df['BattingTeam']],
            text=merged_df['batsman_run'],  # Annotate with runs scored
            hovertext=['Team: ' + team for team in merged_df['BattingTeam']]  # Display team name on hover
        )

        layout = go.Layout(
         
            xaxis_title='Season',
            yaxis_title='Runs Scored',
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white',
            xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
            yaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
        )

    else:
        bowler_df = df[df['bowler'] == player_name]
        wickets_per_season = bowler_df.groupby(['Season'])['isWicketDelivery'].sum().reset_index()
        team_df = bowler_df.drop_duplicates(subset=['Season'])[['Season', 'BowlingTeam']].reset_index(drop=True)
        merged_df = wickets_per_season.merge(team_df, on='Season', how='left')

        trace = go.Bar(
            x=merged_df['Season'],
            y=merged_df['isWicketDelivery'],
            marker_color=[team_primary_colors.get(team, '#808080') for team in merged_df['BowlingTeam']],
            text=merged_df['isWicketDelivery'],  # Annotate with wickets taken
            hovertext=['Team: ' + team for team in merged_df['BowlingTeam']]  # Display team name on hover
        )

        layout = go.Layout(
            xaxis_title='Season',
            yaxis_title='Wickets Taken',
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white',
            xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
            yaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
        )

    fig = go.Figure(data=[trace], layout=layout)
    return fig

@callback(
    Output('i-p-wickets-heading', 'children'),
    [Input('player-dropdown', 'value')]
)
def update_player_wickets(input_value):
    count_batter = df[df['batter'] == input_value].shape[0]
    count_bowler = df[df['bowler'] == input_value].shape[0]

    isBatter = True if count_batter > count_bowler else False

    if isBatter:
        return f'Current Form and Expected Runs for {input_value}'
    else:
        return f'Current Form and Expected Wickets for {input_value}'

@callback(
    Output('i-p-wickets', 'figure'),
    [Input('player-dropdown', 'value')]
)
def generate_player_form_chart(input_value, df=df):
    player_name=input_value
    count_batter = df[df['batter'] == player_name].shape[0]
    count_bowler = df[df['bowler'] == player_name].shape[0]
    df['Date'] = pd.to_datetime(df['Date'])
    is_batter = count_batter > count_bowler

    if is_batter:
        batter_df = df[df['batter'] == player_name]
        current_form = batter_df.groupby('Date')['batsman_run'].sum().tail(5).reset_index()
        expected_next_match = round(current_form['batsman_run'].mean())

        trace_runs = go.Scatter(
            x=current_form['Date'],
            y=current_form['batsman_run'],
            mode='lines+markers',
            name='Runs Scored',
            line=dict(color='#e967fd')
        )

        # Create a trace for the dashed line
        trace_prediction = go.Scatter(
            x=[current_form['Date'].iloc[-1], current_form['Date'].iloc[-1] + pd.DateOffset(days=1)],
            y=[current_form['batsman_run'].iloc[-1], expected_next_match],
            mode='lines+markers',
            line=dict(dash='dash', color='white'),
            name='Expected Runs',
            opacity=0.6
        )

        # Create the layout
        layout = go.Layout(
            xaxis_title='Date',
            yaxis_title='Runs Scored',
            xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
            yaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white',
            legend=dict(
                x=0,
                y=1.2,
                orientation='h'
            )
        )

    else:
        bowler_df = df[df['bowler'] == player_name]
        current_form = bowler_df.groupby('Date')['isWicketDelivery'].sum().tail(5).reset_index()
        expected_next_match = round(current_form['isWicketDelivery'].mean())

        trace_wickets = go.Scatter(
            x=current_form['Date'],
            y=current_form['isWicketDelivery'],
            mode='lines+markers',
            name='Wickets Taken',
            line=dict(color='#e967fd')
        )

        # Create a trace for the dashed line
        trace_prediction = go.Scatter(
            x=[current_form['Date'].iloc[-1], current_form['Date'].iloc[-1] + pd.DateOffset(days=1)],
            y=[current_form['isWicketDelivery'].iloc[-1], expected_next_match],
            mode='lines+markers',
            line=dict(dash='dash', color='white'),
            name='Expected Wickets',
            opacity=0.6
        )

        # Create the layout
        layout = go.Layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white',
            xaxis_title='Date',
            yaxis_title='Wickets Taken',
            xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
            yaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
            legend=dict(
                x=0,
                y=1.15,
                orientation='h'
            )
        )

    fig = go.Figure(data=[trace_runs, trace_prediction] if is_batter else [trace_wickets, trace_prediction], layout=layout)
    return fig


@callback(
    Output('i-p-over-heading', 'children'),
    [Input('player-dropdown', 'value')]
)
def update_player_wickets(input_value):
    count_batter = df[df['batter'] == input_value].shape[0]
    count_bowler = df[df['bowler'] == input_value].shape[0]

    isBatter = True if count_batter > count_bowler else False

    if isBatter:
        return f'Runs scored by {input_value} in each ball of the over'
    else:
        return f'Wickets taken by {input_value} in each ball of the over'

@callback(
    Output('i-p-over', 'figure'),
    [Input('player-dropdown', 'value')]
)
def generate_player_heatmap_chart(input_value, df=df):
    player_name=input_value
    count_batter = df[df['batter'] == player_name].shape[0]
    count_bowler = df[df['bowler'] == player_name].shape[0]
    df['Date'] = pd.to_datetime(df['Date'])
    is_batter = count_batter > count_bowler

    if is_batter:
        batter_df = df[(df['batter'] == player_name) & (df['ballnumber'] <= 6)]
        data_grouped = batter_df.groupby(['overs', 'ballnumber'])['total_run'].sum().reset_index()
        z_label = 'Runs'
    else:
        bowler_df = df[(df['bowler'] == player_name) & (df['ballnumber'] <= 6)]
        data_grouped = bowler_df.groupby(['overs', 'ballnumber'])['isWicketDelivery'].sum().reset_index()
        z_label = 'Wickets'

    heatmap_data = data_grouped.pivot(index='overs', columns='ballnumber', values=data_grouped.columns[-1]).fillna(0)
    heatmap_data.index = heatmap_data.index.astype(str)
    heatmap_data.columns = heatmap_data.columns.astype(str)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values[::-1],
        x=list(heatmap_data.columns),
        y=list(heatmap_data.index[::-1]),
        colorscale='RdBu_r',
        colorbar=dict(title=z_label)
    ))

    # Update layout
    fig.update_layout(
        xaxis_title='Ball Number',
        yaxis_title='Overs',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
        yaxis=dict(showgrid=True, gridcolor='#f0ebeb')
    )

    # Show the plot
    return fig


@callback(
    Output('i-p-against-heading', 'children'),
    [Input('player-dropdown', 'value')]
)
def update_player_against(input_value):
    count_batter = df[df['batter'] == input_value].shape[0]
    count_bowler = df[df['bowler'] == input_value].shape[0]

    isBatter = True if count_batter > count_bowler else False

    if isBatter:
        return f'Average vs Strike Rate for {input_value} Against Each Team'
    else:
        return f'Bowling Average vs Economy for {input_value} Against Each Team'

@callback(
    Output('i-p-against-heading-smaller', 'children'),
    [Input('player-dropdown', 'value')]
)
def update_player_against(input_value):
    count_batter = df[df['batter'] == input_value].shape[0]
    count_bowler = df[df['bowler'] == input_value].shape[0]

    isBatter = True if count_batter > count_bowler else False

    if isBatter:
        return f'Color represents opponent team and Bubble Size represents Runs scored'
    else:
        return f'Color represents opponent team and Bubble Size represents Wickets taken'

@callback(
    Output('i-p-against', 'figure'),
    [Input('player-dropdown', 'value')]
)
def generate_player_performance_chart(input_value, df=df):
    player_name = input_value
    count_batter = df[df['batter'] == player_name].shape[0]
    count_bowler = df[df['bowler'] == player_name].shape[0]
    is_batter = count_batter > count_bowler

    if is_batter:
        batter_df = df[df['batter'] == player_name]
        runs_scored = batter_df.groupby('BowlingTeam')['batsman_run'].sum()
        balls_faced = batter_df.groupby('BowlingTeam').size()
        dismissals = batter_df[batter_df['isWicketDelivery'] == 1].groupby('BowlingTeam').size()

        # Calculate strike rate and average against each team
        strike_rate = (runs_scored / balls_faced) * 100
        average = runs_scored / dismissals

        # Create a DataFrame with the required data
        team_stats = pd.DataFrame({
            'BowlingTeam': runs_scored.index,
            'RunsScored': runs_scored.values,
            'StrikeRate': strike_rate.values,
            'Average': average.values
        })
        marker_colors = {idx: team_primary_colors.get(team, 'black') for idx, team in enumerate(team_stats['BowlingTeam'])}

        # Plot the scatter plot with rings
        fig = px.scatter(team_stats, x='Average', y='StrikeRate', size='RunsScored', hover_name='BowlingTeam',
                         labels={'Average': 'Average', 'StrikeRate': 'Strike Rate', 'RunsScored': 'Runs Scored'},
                         size_max=15, opacity=1)

    else:
        bowler_df = df[df['bowler'] == player_name]
        wickets_taken = bowler_df.groupby('BattingTeam')['isWicketDelivery'].sum()
        balls_bowled = bowler_df.groupby('BattingTeam').size()
        runs_conceded = bowler_df.groupby('BattingTeam')['total_run'].sum()

        # Calculate bowling average and economy against each team
        bowling_average = runs_conceded / wickets_taken
        economy = (runs_conceded / balls_bowled) * 6

        # Create a DataFrame with the required data
        team_stats = pd.DataFrame({
            'BattingTeam': wickets_taken.index,
            'WicketsTaken': wickets_taken.values,
            'BowlingAverage': bowling_average.values,
            'Economy': economy.values
        })

        marker_colors = {idx: team_primary_colors.get(team, 'black') for idx, team in enumerate(team_stats['BattingTeam'])}

        # Plot the scatter plot with rings
        fig = px.scatter(team_stats, x='BowlingAverage', y='Economy', size='WicketsTaken', hover_name='BattingTeam',
                         labels={'BowlingAverage': 'Bowling Average', 'Economy': 'Economy', 'WicketsTaken': 'Wickets Taken'},
                         size_max=15, opacity=1)

    # Update layout to show rings instead of circles
    fig.update_traces(marker=dict(symbol='circle-open', line=dict(color='black', width=3)),
                      line=dict(color='black', width=1),
                      marker_color=[marker_colors.get(idx) for idx in range(len(team_stats))])

    # Update layout
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white',
                      xaxis=dict(showgrid=True, gridcolor='#f0ebeb'), yaxis=dict(showgrid=True, gridcolor='#f0ebeb'))

    # Show the plot
    return fig


@callback(
    Output('player-img-display', 'children'),
    [Input('player-dropdown', 'value')]
)
def display_player_img(selected_team):
    players_present= player_pics.keys()
    if selected_team in players_present:
        img_path=player_pics[selected_team]
        return html.Img(src=img_path)
    else:
        team_name=get_curr_team(selected_team)
        if team_name in teams_present:
           img_path= team_jersey[team_name]
           return html.Img(src=img_path)
        else:
           return html.Div('Jersey not available')