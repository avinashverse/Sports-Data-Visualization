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

dash.register_page(__name__, path='/ipl-team-dashboard', name="ipl-team-dashboard")

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

def matches_played(df, team):
    return df[(df.Team1 == team) | (df.Team2 == team)].shape[0]

def matches_won(df, team):
    return df[df.WinningTeam == team].shape[0]

def matches_no_result(df, team):
    return df[((df.Team1 == team) | (df.Team2 == team)) & (df.WinningTeam.isnull())].shape[0]

def point_table(season):
    df = match[match.Season == season]
    
    new_df = pd.DataFrame()
    
    new_df['Team Name'] = np.union1d(df.Team1.unique(), df.Team2.unique())
    
    new_df["Matches Played"] = new_df["Team Name"].apply(lambda x: matches_played(df, x))
    new_df["Matches Won"] = new_df["Team Name"].apply(lambda x: matches_won(df, x))
    new_df["No Result"] = new_df["Team Name"].apply(lambda x: matches_no_result(df, x))
    new_df["Points"] = new_df["Matches Won"]*2 + new_df["No Result"]
    
    new_df.sort_values("Points", ascending = False, inplace=True)
    new_df.set_index("Team Name", inplace=True)
    
    return new_df

def get_points_table(season):
    df3 = point_table(season)
    df_ = df[df.Season == season].copy()

    df3["SeasonPosition"] = df3.Points.rank(ascending=False, method= 'first').astype('object')
    df3["SeasonPosition"] = df3["SeasonPosition"].apply(lambda x: str(int(x)) + 'th')
    df_["LoosingTeam"] = pd.concat([df_[df_.WinningTeam == df_.Team1]["Team2"],
                                df_[df_.WinningTeam == df_.Team2]["Team1"]])

    final = df_[df_["MatchNumber"] == "Final"]
    wining_team = final.WinningTeam.values[0]
    runner = final.LoosingTeam.values[0]
    df3.at[wining_team, "SeasonPosition"] = "Winner"
    df3.at[runner, "SeasonPosition"] = "Runner Up"

    if season not in ['2008', '2009', '2010']:
        q = df_[df_["MatchNumber"] == "Qualifier 2"]
        e = df_[df_["MatchNumber"] == "Eliminator"]
        third = q.LoosingTeam.values[0]
        fourth = e.LoosingTeam.values[0]
    else:
        third = None
        fourth = None
        for i in range(len(df3)):
            position = df3['SeasonPosition'][i]
            if position not in ['Winner', 'Runner Up']:
                if third is None:
                    third = df3.index[i]
                else:
                    fourth = df3.index[i]
                    break
        

    df3.at[third, "SeasonPosition"] = "Third"
    df3.at[fourth, "SeasonPosition"] = "Fourth"

    return df3.reset_index()
def get_position(season, team):
        
        points_table = get_points_table(season)

        position = points_table[points_table['Team Name'] == team]['SeasonPosition'].values[0]

        return position


# Secondary colors for IPL teams
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





# Dropdown options based on team names
dropdown_options = [{'label': team, 'value': team} for team in team_secondary_colors.keys()]

# Dropdown component
dropdown = dcc.Dropdown(
    id='team-dropdown',
    options=dropdown_options,
    value='Royal Challengers Bangalore',  
    style={'color': 'black'} 
)
layout = html.Div(
        id='i-t-page',
        children=[
            html.Div(
                id='i-t-row1',
                children=[
                    html.Div(
                        id='i-t-team',
                        children=[
                            html.H2("IPL Team Selector"),
                            dropdown,
                            html.H1(id='selected-team-name'), 
                            html.Div(id='team-jersey-display')
                            # html.Img(src="assets/Team_Jerseys/Chennai Super Kings2.png")
                        ]
                    ),
                    html.Div(
                        id='i-t-next',
                        children=[
                            html.Div(
                                id='i-t-graph-pow-death',
                                children=[
                                    html.Div(
                                        className='i-t-plot',
                                        children=[
                                            html.H1(id='i-t-plot-pow-death-heading'),
                                            dcc.Graph(id='i-t-plot-pow-death')
                                        ]
                                        
                                    )
                                ]
                            ),
                            html.Div(
                                id='i-t-graph-pow-death',
                                children=[
                                    html.Div(
                                        className='i-t-plot',
                                        children=[
                                            html.H1(id='i-t-plot-pow-death-heading-wicket'),
                                            dcc.Graph(id='i-t-plot-pow-death-wicket')
                                        ]
                                        
                                    )
                                ]
                            ),
                        ]
                    ),
                    
                ]
            ),
            html.Div(
                id='i-t-row2',
                children=[
                    html.Div(
                        id='i-t-2-col1',
                        children=[
                            html.Div(
                                className='i-t-plot',
                                children=[
                                    html.H1('Average vs Strike Rate'),
                                    html.H3('More the Strike Rate and Batting Average better it is!'),
                                    dcc.Graph(id='i-t-plot-pow-death-strike')
                                ]
                                
                            )
                        ]
                    ),
                    html.Div(
                        id='i-t-2-col2',
                        children=[
                            html.Div(
                                className='i-t-plot',
                                children=[
                                    html.H1('Bowling Average vs Economy'),
                                    html.H3('Lesser the Economy and Bowling Average better it is!'),
                                    dcc.Graph(id='i-t-plot-pow-death-eco')
                                ]
                                
                            )
                            
                        ]
                    ),
                    
                    
                ]
            ),
            html.Div(
                id='i-t-row3',
                children=[
                    html.Div(
                        id='i-t-3-col1',
                        children=[
                            html.Div(
                                className='i-t-plot',
                                children=[
                                    html.H1('Number of Fours and Sixes by Season'),
                                    html.H3(id="i-t-plot-num-heading"),
                                    dcc.Graph(id='i-t-plot-num')
                                ]
                                
                            )
                        ]
                    ),
                    html.Div(
                        id='i-t-3-col2',
                        children=[
                            html.Div(
                                className='i-t-plot',
                                children=[
                                    html.H1(id='i-t-plot-compare-heading'),
                                    dcc.Graph(id='i-t-plot-compare')
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
    Output('selected-team-name', 'children'),
    [Input('team-dropdown', 'value')]
)
def update_selected_team_name(input_value):
    return f'{input_value}'

@callback(
    Output('i-t-plot-pow-death-heading', 'children'),
    [Input('team-dropdown', 'value')]
)
def bar_plots_name(input_value):
    return f'Top Batsmen for {input_value}'

@callback(
    Output('i-t-plot-pow-death', 'figure'),
    [Input('team-dropdown', 'value')]
)
def generate_shaded_bar_plots(input_value, df=df):
    # plt.style.use('dark_background')

    # def generate_shades(color, num_shades):
    #     return sns.light_palette(color, n_colors=num_shades + 3, reverse=True)

    # team_primary_color = team_primary_colors.get(input_value, '#808080')
    # custom_colors = generate_shades(team_primary_color, 10)

    team_df_powerplay = df[(df['BattingTeam'] == input_value) & (df['overs'] < 6)]
    top_10_powerplay = team_df_powerplay.groupby('batter')['batsman_run'].sum().reset_index().sort_values(by='batsman_run', ascending=False).head(10)

    team_df_death = df[(df['BattingTeam'] == input_value) & (df['overs'] >= 15)]
    top_10_death = team_df_death.groupby('batter')['batsman_run'].sum().reset_index().sort_values(by='batsman_run', ascending=False).head(10)
    custom_colors_death=[team_secondary_colors[input_value] for i in range(10)]
    custom_colors_pow=[team_primary_colors[input_value] for i in range(10)]
    # Create Plotly bar charts instead of Matplotlib plots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Top 10 Powerplay Scorers', 'Top 10 Death Overs Scorers'))

    fig.add_trace(go.Bar(y=top_10_powerplay['batter'], x=top_10_powerplay['batsman_run'], orientation='h', name='Powerplay', marker=dict(color=custom_colors_pow)), row=1, col=1)
    fig.add_trace(go.Bar(y=top_10_death['batter'], x=top_10_death['batsman_run'], orientation='h', name='Death Overs', marker=dict(color=custom_colors_death)), row=1, col=2)

    fig.update_layout(barmode='stack',plot_bgcolor='black',  # Set plot background color to black
        paper_bgcolor='black',  # Set paper background color to black
        font_color='white',
        xaxis_title='Runs',  # Set x-axis label
        yaxis_title='Batsmen',  # Set y-axis label
        legend=dict(
            orientation='h',  # Horizontal orientation for legend
            yanchor='top',  # Anchor legend to the top
            y=1.25  # Adjust the position of the legend
        )
    )

    return fig


@callback(
    Output('i-t-plot-pow-death-heading-wicket', 'children'),
    [Input('team-dropdown', 'value')]
)
def bar_plots_name(input_value):
    return f'Top Bowler for {input_value}'

@callback(
    Output('i-t-plot-pow-death-wicket', 'figure'),
    [Input('team-dropdown', 'value')]
)
def generate_shaded_bar_plots_bowlers(input_value, df=df):

    team_df_powerplay_bowlers = df[(df['BowlingTeam'] == input_value) & (df['overs'] < 6) & (df['isWicketDelivery'] == 1)]
    top_10_powerplay_bowlers = team_df_powerplay_bowlers.groupby('bowler')['isWicketDelivery'].sum().reset_index().sort_values(by='isWicketDelivery', ascending=False).head(10)

    team_df_death_bowlers = df[(df['BowlingTeam'] == input_value) & (df['overs'] >= 15) & (df['isWicketDelivery'] == 1)]
    top_10_death_bowlers = team_df_death_bowlers.groupby('bowler')['isWicketDelivery'].sum().reset_index().sort_values(by='isWicketDelivery', ascending=False).head(10)
    custom_colors_death=[team_secondary_colors[input_value] for i in range(10)]
    custom_colors_pow=[team_primary_colors[input_value] for i in range(10)]
    # Create Plotly bar charts
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Top 10 Powerplay Wicket-takers', 'Top 10 Death Overs Wicket-takers'))

    fig.add_trace(go.Bar(y=top_10_powerplay_bowlers['bowler'], x=top_10_powerplay_bowlers['isWicketDelivery'], orientation='h', name='Powerplay', marker=dict(color=custom_colors_pow)), row=1, col=1)
    fig.add_trace(go.Bar(y=top_10_death_bowlers['bowler'], x=top_10_death_bowlers['isWicketDelivery'], orientation='h', name='Death Overs', marker=dict(color=custom_colors_death)), row=1, col=2)

    fig.update_layout(barmode='stack', plot_bgcolor='black', paper_bgcolor='black', font_color='white', xaxis_title='Wickets', yaxis_title='Bowler', legend=dict(orientation='h', yanchor='top', y=1.25))

    return fig




@callback(
    Output('i-t-plot-pow-death-strike', 'figure'),
    [Input('team-dropdown', 'value')]
)
def generate_batter_performance_plot(input_value, df=df):
    team_df = df[df['BattingTeam'] == input_value]

    batter_stats = team_df.groupby('batter').agg({'batsman_run': 'sum', 'isWicketDelivery': 'sum', 'ID': 'count'}).reset_index()
    batter_stats.columns = ['Batter', 'Runs', 'Outs', 'Balls Faced']

    batter_stats = batter_stats[batter_stats['Balls Faced'] >= 150]

    batter_stats['Strike Rate'] = batter_stats['Runs'] * 100 / batter_stats['Balls Faced']
    batter_stats['Average'] = batter_stats['Runs'] / batter_stats['Outs']

    team_color = team_primary_colors.get(input_value, '#808080')
    team_secondary_color = team_secondary_colors.get(input_value, '#808080')

    most_strike_rate_batter = batter_stats.loc[batter_stats['Strike Rate'].idxmax()]

    # Create the scatter plot
    fig = px.scatter(batter_stats, x='Average', y='Strike Rate', hover_name='Batter',
                     hover_data={'Average': True, 'Strike Rate': True, 'Runs': True},
                     labels={'Average': 'Average', 'Strike Rate': 'Strike Rate', 'Runs': 'Runs'},
                 color='Runs', color_continuous_scale=[[0, team_color], [1, team_secondary_color]])

    # Add layout configurations
    fig.update_layout( xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      yaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                       plot_bgcolor='black', paper_bgcolor='black', font_color='white')


    # Add annotation text for the most strike rate batsman
    fig.add_annotation(x=most_strike_rate_batter['Average'], y=most_strike_rate_batter['Strike Rate'],
                       text=f"{most_strike_rate_batter['Batter']} is the Super Striker for {input_value}",
                       font=dict(color='black'),
                       showarrow=True,
                       arrowhead=2,
                       arrowsize=1,
                       arrowwidth=2,
                       arrowcolor="#b0aeae",
                       ax=-20,
                       ay=-80,
                       bordercolor="#b0aeae",
                       borderwidth=2,
                       borderpad=4,
                       bgcolor="#dedede",
                       opacity=0.9
                      )

    fig.update_traces(marker=dict(line=dict(color='black', width=1)))

    return fig

@callback(
    Output('i-t-plot-pow-death-eco', 'figure'),
    [Input('team-dropdown', 'value')]
)
def generate_bowler_performance_plot(input_value, df=df):
    team_df = df[df['BowlingTeam'] == input_value]

    bowler_stats = team_df.groupby('bowler').agg({'extras_run': 'sum', 'total_run': 'sum', 'isWicketDelivery': 'sum', 'ballnumber': 'count'}).reset_index()
    bowler_stats.columns = ['Bowler', 'Extras', 'Runs Conceeded', 'Wickets', 'Balls Bowled']

    bowler_stats = bowler_stats[bowler_stats['Balls Bowled'] >= 100]

    bowler_stats['Bowling Average'] = bowler_stats['Runs Conceeded'] / bowler_stats['Wickets']

    bowler_stats['Economy'] = bowler_stats['Runs Conceeded'] * 6 / bowler_stats['Balls Bowled']

    team_color = team_primary_colors.get(input_value, '#808080')
    team_secondary_color = team_secondary_colors.get(input_value, '#808080')

    most_economical_bowler = bowler_stats.loc[bowler_stats['Economy'].idxmin()]

    # Create the scatter plot
    fig = px.scatter(bowler_stats, x='Bowling Average', y='Economy', hover_name='Bowler',
                     hover_data={'Wickets': True, 'Economy': True, 'Runs Conceeded': True},

                     labels={'Bowling Average': 'Bowling Average', 'Economy': 'Economy', 'Runs Conceeded': 'Runs Conceeded'},
                     color='Wickets', color_continuous_scale=[[0, team_color], [1, team_secondary_color]])

    # Add layout configurations
    fig.update_layout(xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      yaxis=dict(showgrid=True, gridcolor='#f0ebeb'),plot_bgcolor='black', paper_bgcolor='black', font_color='white'
                      )

    
    fig.add_annotation(x=most_economical_bowler['Bowling Average'], y=most_economical_bowler['Economy'],
                       text=f"{most_economical_bowler['Bowler']} is the Most Economical Bowler for {input_value}",
                       font=dict(color='black'),
                       showarrow=True,
                       arrowhead=2,
                       arrowsize=1,
                       arrowwidth=2,
                       arrowcolor="#b0aeae",
                       ax=-20,
                       ay=80,
                       bordercolor="#b0aeae",
                       borderwidth=2,
                       borderpad=4,
                       bgcolor="#dedede",
                       opacity=0.9
                      )

    fig.update_traces(marker=dict(line=dict(color='black', width=1)))

    return fig


@callback(
    Output('i-t-plot-num-heading', 'children'),
    [Input('team-dropdown', 'value')]
)
def bar_plots_name(input_value):
    return f'Hover to see position of {input_value} in that season'

@callback(
    Output('i-t-plot-num', 'figure'),
    [Input('team-dropdown', 'value')]
)
def generate_fours_sixes_chart(input_value,df=df):
    team=input_value
    
    
    team_sixes = df[(df['BattingTeam'] == team) & (df['batsman_run'] == 6)].groupby('Season').size().reset_index().rename(columns={0: 'Sixes'})
    team_fours = df[(df['BattingTeam'] == team) & (df['batsman_run'] == 4)].groupby('Season').size().reset_index().rename(columns={0: 'Fours'})

    team_sixes = team_sixes.sort_values(by='Season')
    team_fours = team_fours.sort_values(by='Season')

    team_color = team_primary_colors.get(team, '#808080')
    team_secondary_color = team_secondary_colors.get(team, '#808080')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=team_fours['Season'],
        y=team_fours['Fours'],
        name='Fours',
        marker_color=team_color,
        text=team_fours['Fours'],
        textposition='inside',
        hoverinfo='text',
        hovertext=[f'Position: {get_position(season, team)}' for season in team_fours['Season']]
    ))

    # Add bars for sixes on top of fours
    fig.add_trace(go.Bar(
        x=team_sixes['Season'],
        y=team_sixes['Sixes'],
        name='Sixes',
        marker_color=team_secondary_color,
        text=team_sixes['Sixes'],
        textposition='inside',
        hoverinfo='text',
        hovertext=[f'Position: {get_position(season, team)}' for season in team_fours['Season']]
    ))

    fig.update_layout(
        xaxis_title='Season',
        yaxis_title='Count',
        barmode='stack',
        xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
        yaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
        plot_bgcolor='black', paper_bgcolor='black', font_color='white'
    )

    return fig

@callback(
    Output('i-t-plot-compare-heading', 'children'),
    [Input('team-dropdown', 'value')]
)
def bar_plots_name(input_value):
    return f'Percentage of Wins for {input_value} Against Each Opponent'

@callback(
    Output('i-t-plot-compare', 'figure'),
    [Input('team-dropdown', 'value')]
)
def generate_win_percentage_chart(input_value, match_data=match):
    team=input_value
    team_df = match_data[(match_data['Team1'] == team) | (match_data['Team2'] == team)]
    team_df['OpponentTeam'] = np.where(team_df['Team1'] == team, team_df['Team2'], team_df['Team1'])
    team_df['TeamIsWinner'] = np.where(team_df['WinningTeam'] == team, 1, 0)

    win_percentage = (team_df.groupby('OpponentTeam')['TeamIsWinner'].sum() * 100 / team_df.groupby('OpponentTeam')['TeamIsWinner'].size()).reset_index().rename(columns={
        'TeamIsWinner': 'Win%'
    })

    win_percentage = win_percentage.sort_values(by='Win%', ascending=True)
    team_color = team_primary_colors.get(team, '#808080')

    fig = go.Figure(go.Bar(
        y=win_percentage['OpponentTeam'],
        x=win_percentage['Win%'],
        orientation='h',
        marker_color=team_color,
        text=win_percentage['Win%'].apply(lambda x: f"{x:.1f}%")  # Annotate with win percentage
    ))

    fig.update_layout(
        xaxis_title='Opponent Team',
        yaxis_title='Win Percentage (%)',
        plot_bgcolor='black', paper_bgcolor='black', font_color='white',
        xaxis=dict(showgrid=True, gridcolor='#f0ebeb', range=[-0.5, 115]),
        yaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
    )

    # Adjust text position to be outside the bars
    fig.update_traces(textposition='outside')

    return fig

@callback(
    Output('team-jersey-display', 'children'),
    [Input('team-dropdown', 'value')]
)
def display_team_jersey(selected_team):
    jersey_path = team_jersey.get(selected_team)
    if jersey_path:
        return html.Img(src=jersey_path)
    else:
        return html.Div('Jersey not available')