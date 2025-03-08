import dash
import pandas as pd
import numpy as np
from dash import html, dcc, callback, Input, Output
from dash.dependencies import Input, Output
from datetime import datetime, timedelta
import pycountry_convert as pc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from PIL import Image

dash.register_page(__name__, path='/ipl-page', name="ipl-page")

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

    df_ = match[match.Season == season]
    
    new_df = pd.DataFrame()
    
    new_df['Team Name'] = np.union1d(df_.Team1.unique(), df_.Team2.unique())
    
    new_df["Matches Played"] = new_df["Team Name"].apply(lambda x: matches_played(df_, x))
    new_df["Matches Won"] = new_df["Team Name"].apply(lambda x: matches_won(df_, x))
    new_df["No Result"] = new_df["Team Name"].apply(lambda x: matches_no_result(df_, x))
    new_df["Points"] = new_df["Matches Won"]*2 + new_df["No Result"]
    
    new_df.sort_values("Points", ascending = False, inplace=True)
    new_df.set_index("Team Name", inplace=True)
    
    return new_df

def strike_rate_average(df):
    batter_stats = df.groupby('batter').agg({'batsman_run': 'sum', 'isWicketDelivery': 'sum', 'ID': 'count'}).reset_index()
    batter_stats.columns = ['Batter', 'Runs', 'Outs', 'Balls Faced']

    batter_stats = batter_stats[batter_stats['Balls Faced'] >= 200]

    batter_stats['Strike Rate'] = batter_stats['Runs'] * 100 / batter_stats['Balls Faced']
    batter_stats['Average'] = batter_stats['Runs'] / batter_stats['Outs']
    
    fig = px.scatter(batter_stats, x='Average', y='Strike Rate', hover_name='Batter',
                 hover_data={'Average': True, 'Strike Rate': True, 'Runs': True},
                 title='Average vs Strike Rate<br><b><span style="font-size:14px">KL Rahul has the best Average, Andre Russell has the best Strike Rate</span></b>',
                 labels={'Average': 'Average', 'Strike Rate': 'Strike Rate', 'Runs': 'Runs'},
                 height=600, width=600, color_discrete_sequence=['#E178C5'])

    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      yaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      height=600)
    fig.update_traces(marker=dict(line=dict(color='black', width=1)))

    fig.add_shape(type="rect",
                  x0=batter_stats['Average'].mean(), y0=batter_stats['Strike Rate'].mean(),
                  x1=50, y1=batter_stats['Strike Rate'].max() + 2,
                  line=dict(color="green", width=2), fillcolor="green", opacity=0.1)  # Quadrant I

    fig.add_shape(type="rect",
                  x0=2, y0=batter_stats['Strike Rate'].mean(),
                  x1=batter_stats['Average'].mean(), y1=batter_stats['Strike Rate'].max() + 2,
                  line=dict(color="red", width=2), fillcolor="red", opacity=0.1)  # Quadrant II

    fig.add_shape(type="rect",
                  x0=2, y0=75,
                  x1=batter_stats['Average'].mean(), y1=batter_stats['Strike Rate'].mean(),
                  line=dict(color="orange", width=2), fillcolor="orange", opacity=0.1)  # Quadrant III

    fig.add_shape(type="rect",
                  x0=batter_stats['Average'].mean(), y0=75,
                  x1=50, y1=batter_stats['Strike Rate'].mean(),
                  line=dict(color="blue", width=2), fillcolor="blue", opacity=0.1)  # Quadrant IV

    # Add quadrant labels
    fig.add_annotation(x=38, y=160,
                       text="Best<br>Batters", showarrow=False, font=dict(size=15, color='black'))
    fig.add_annotation(x=10, y=160,
                       text="Super<br>Strikers", showarrow=False, font=dict(size=15, color='black'))
    fig.add_annotation(x=38, y=100,
                       text="Trust them<br>Win the Game", showarrow=False, font=dict(size=15, color='black'))

    fig.add_annotation(x=46.5, y=133,
                       text="KL Rahul", showarrow=True,
                       arrowhead=2, arrowsize=1, arrowwidth=2,
                       ax=40, ay=-40,
                       font=dict(size=15, color='#1D24CA'))

    fig.add_annotation(x=30.5, y=169,
                       text="Andre Russell", showarrow=True,
                       arrowhead=2, arrowsize=1, arrowwidth=2,
                       ax=-40, ay=-40,
                       font=dict(size=15, color='#1D24CA'))

    return fig

fig_avg_sr = strike_rate_average(df)


def super_over_plot(df):
    df['Date'] = pd.to_datetime(df['Date'])

    first_inning_super_over = df[df['innings'] == 3].groupby('Date')['batsman_run'].sum().reset_index()
    second_inning_super_over = df[df['innings'] == 4].groupby('Date')['batsman_run'].sum().reset_index()

    batting_team = df[df['innings'] == 3].drop_duplicates('Date')[['Date', 'BattingTeam']]
    bowling_team = df[df['innings'] == 4].drop_duplicates('Date')[['Date', 'BattingTeam']]

    first_inning_super_over = pd.merge(first_inning_super_over, batting_team, on='Date')
    second_inning_super_over = pd.merge(second_inning_super_over, bowling_team, on='Date')

    super_over = pd.merge(first_inning_super_over, second_inning_super_over, on='Date').rename(columns={
        'batsman_run_x': '1st Innings Score',
        'batsman_run_y': '2nd Innings Score',
        'BattingTeam_x': 'Team 1',
        'BattingTeam_y': 'Team 2'
    })

    super_over['Winner'] = np.where(super_over['1st Innings Score'] > super_over['2nd Innings Score'], super_over['Team 1'], super_over['Team 2'])
    
    trace1 = go.Scatter(
        x=super_over['Date'],
        y=[0] * len(super_over),
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='1st Innings Score Reference Line'
    )

    # Create trace for the line plot of (2nd Innings Score - 1st Innings Score)
    trace2 = go.Scatter(
        x=super_over['Date'],
        y=super_over['2nd Innings Score'] - super_over['1st Innings Score'],
        mode='lines+markers',
        name='Score Difference',
        text=[f"{team1} vs {team2}<br>Winner: {winner}" for team1, team2, winner in zip(super_over['Team 1'], super_over['Team 2'], super_over['Winner'])]
    )

    # Create the layout
    layout = go.Layout(
        title='Super Over Score Difference (Team2 - Team1)',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Score Difference'),
        hovermode='closest',
        height=400, width=700,
        legend=dict(orientation='h', yanchor='top', y=-0.25, xanchor='right', x=0.8)
    )

    # Add annotation for the match between Punjab Kings and Delhi Capitals on 2020-09-20
    annotation = dict(
        x='2020-09-20',
        y=super_over.loc[super_over['Date'] == '2020-09-20', '2nd Innings Score'].values[0] - super_over.loc[super_over['Date'] == '2020-09-20', '1st Innings Score'].values[0],
        xref="x",
        yref="y",
        text="Only tied Super Over: Punjab v/s Mumbai",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#b0aeae",
        ax=-80,
        ay=-80,
        bordercolor="#b0aeae",
        borderwidth=2,
        borderpad=4,
        bgcolor="#dedede",
        opacity=0.9
    )

    # Create figure
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Add annotation to the figure
    fig.update_layout(annotations=[annotation])

    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      yaxis=dict(showgrid=True, gridcolor='#f0ebeb'))

    return fig

fig_super_over = super_over_plot(df)


def get_knockout_wordcloud(df):
    knockout_match_types = ['Final', 'Qualifier 2', 'Eliminator', 'Qualifier 1']

    knockout_matches = df[df['MatchNumber'].isin(knockout_match_types)]

    knockout_matches['Result'] = np.where(knockout_matches['BattingTeam'] == knockout_matches['WinningTeam'], 1, 0)
    knockout_matches.drop_duplicates(subset=['innings', 'ID'], inplace=True)

    won = knockout_matches.groupby('BattingTeam')['Result'].sum().sort_index()
    lost = (knockout_matches.groupby('BattingTeam')['Result'].size() - knockout_matches.groupby('BattingTeam')['Result'].sum()).sort_index()
    
    num_knockouts_played = knockout_matches.groupby('BattingTeam')['Result'].size().reset_index().rename(columns={'BattingTeam': 'Team', 'Result': 'Num Knockout Games'})
    num_knockouts_played['Team'] = [
        'CSK',  # Chennai Super Kings
        'DCH',   # Deccan Chargers
        'DC',   # Delhi Capitals
        'GL',   # Gujarat Lions
        'GT',   # Gujarat Titans
        'KKR',  # Kolkata Knight Riders
        'LSG',  # Lucknow Super Giants
        'MI',   # Mumbai Indians
        'PBKS', # Punjab Kings
        'RR',   # Rajasthan Royals
        'RPS',  # Rising Pune Supergiants
        'RCB',  # Royal Challengers Bangalore
        'SRH'   # Sunrisers Hyderabad
    ]

    custom_colors = {
        'CSK': '#F0C015',
        'DCH': '#7e8282',
        'DC': '#323FA9',
        'GL': '#fa4428',
        'GT': '#4E6579',
        'KKR': '#3C1582',
        'LSG': '#303F79',
        'MI': '#0F2AC6',
        'PBKS': '#FD0118',
        'RR': '#FF0081',
        'RPS': '#03f4fc',
        'RCB': '#DA0001',
        'SRH': '#E33307'
    }

    data = num_knockouts_played.values
    word_freq = {team: freq for team, freq in data}

    def color_func(word, **kwargs):
        return custom_colors.get(word, 'black')

    wordcloud = WordCloud(width=800, height=800, background_color='white', color_func=color_func)

    # Generate the word cloud using the word frequencies
    wordcloud.generate_from_frequencies(word_freq)

    fig = px.imshow(wordcloud)
    
    fig.update_layout(
        title={
            'text': "Number of Knockout Matches Played by IPL Teams",
            'font': {'size': 20},
            'pad': {'t': 20}
        }
    )
    fig.update_layout(xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                      yaxis=dict(showgrid=False, showticklabels=False, zeroline=False))
    return fig

fig_wordcloud = get_knockout_wordcloud(df)


def get_knockout_per_team(df):
    knockout_match_types = ['Final', 'Qualifier 2', 'Eliminator', 'Qualifier 1']

    knockout_matches = df[df['MatchNumber'].isin(knockout_match_types)]

    knockout_matches['Result'] = np.where(knockout_matches['BattingTeam'] == knockout_matches['WinningTeam'], 1, 0)
    knockout_matches.drop_duplicates(subset=['innings', 'ID'], inplace=True)
    won = knockout_matches.groupby('BattingTeam')['Result'].sum().sort_index()
    lost = (knockout_matches.groupby('BattingTeam')['Result'].size() - knockout_matches.groupby('BattingTeam')['Result'].sum()).sort_index()
    
    team_stats = pd.DataFrame({'Team': won.index,
                           'Wins': won.values,
                           'Losses': lost.values})

    # Create a stacked bar chart using Plotly
    fig = go.Figure(data=[
        go.Bar(name='Wins', y=team_stats['Team'], x=team_stats['Wins'], orientation='h', marker_color='#0C359E'),
        go.Bar(name='Losses', y=team_stats['Team'], x=team_stats['Losses'], orientation='h', marker_color='#EE99C2')
    ])

    # Update layout
    fig.update_layout(barmode='stack', title='Wins and Losses for Knockout Matches by Team<br><b><span style="font-size:14px">Gujarat Titans and Deccan Chargers have won all their knockout matches, LSG and Gujarat Lions have lost them all!</span></b>',
                      xaxis_title='Team', yaxis_title='Count', legend_title='Result')

    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      yaxis=dict(showgrid=True, gridcolor='#f0ebeb'), height=500)
    
    return fig

fig_knockout_team = get_knockout_per_team(df)

def bat_first_wins(df):
    top_10_stadiums = list(df['Venue'].value_counts().index[:9])
    top_10_stadium_data = df[df['Venue'].isin(top_10_stadiums)]
    top_10_stadium_data = top_10_stadium_data[top_10_stadium_data['innings'] == 1]

    top_10_stadium_data['bat_first_wins'] = np.where(top_10_stadium_data['BattingTeam'] == top_10_stadium_data['WinningTeam'], 1, 0)
    
    bat_first_wins = top_10_stadium_data.groupby('Venue')['bat_first_wins'].sum()
    ball_first_wins = top_10_stadium_data.groupby('Venue').size() - bat_first_wins

    # Calculate win percentage when batting first
    win_percentage_bat_first = bat_first_wins / (ball_first_wins + bat_first_wins) * 100

    # Create a DataFrame with the win percentage data
    win_percentage_data = pd.DataFrame({'Venue': bat_first_wins.index,
                                        'Batting First Win Percentage': win_percentage_bat_first})

    # Sort the DataFrame by win percentage when batting first
    win_percentage_data_sorted = win_percentage_data.sort_values(by='Batting First Win Percentage', ascending=True)

    # Create a bar chart using Plotly Express
    fig = px.bar(win_percentage_data_sorted, x='Batting First Win Percentage', y='Venue',
                 title='Batting First Win Percentage<br><b><span style="font-size:14px">Bowl first at Sawai Mann Singh Stadium, bat first at Chidambaram Stadium.</span></b>',
                 orientation='h', color_discrete_sequence=['#5F5D9C'])

    for i in range(len(win_percentage_data_sorted)):
        fig.add_annotation(x=win_percentage_data_sorted.iloc[i]['Batting First Win Percentage'],
                           y=win_percentage_data_sorted.iloc[i]['Venue'],
                           text=f"{win_percentage_data_sorted.iloc[i]['Batting First Win Percentage']:.2f}%",
                           showarrow=False,
                           font=dict(size=10),
                           xshift=25)

    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      yaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      height=400, width=650)

    # Show the plot
    return fig

fig_bat_wins = bat_first_wins(df)






def bat_first_stadium(df):
    top_10_stadiums = list(df['Venue'].value_counts().index[:9])
    top_10_stadium_data = df[df['Venue'].isin(top_10_stadiums)]
    top_10_stadium_data = top_10_stadium_data[top_10_stadium_data['innings'] == 1]

    top_10_stadium_data['bat_first_wins'] = np.where(top_10_stadium_data['BattingTeam'] == top_10_stadium_data['WinningTeam'], 1, 0)

    bat_first_wins = top_10_stadium_data[top_10_stadium_data['bat_first_wins'] == 1]

    count_matches_per_stadium = bat_first_wins.groupby('Venue')['ID'].nunique()
    avg_win_score = bat_first_wins.groupby('Venue')['total_run'].sum() / count_matches_per_stadium
    sorted_avg_win_score = avg_win_score.sort_values(ascending=True)

    fig = go.Figure()

    # Create a list of marker colors based on the venue name
    marker_colors = '#874CCC'
    marker_opacities = [0.4 if venue != 'M.Chinnaswamy Stadium' else 0.9 for venue in sorted_avg_win_score.index]

    fig.add_trace(go.Bar(
        y=sorted_avg_win_score.index,
        x=sorted_avg_win_score.values,
        name='Confidence Interval',
        marker_color=marker_colors,
        marker_opacity=marker_opacities,
        orientation='h'
    ))

    # Update layout to increase height
    fig.update_layout(
        title='Venue Wise Safe Score to Defend<br><b><span style="font-size:14px">Struggle for Batters: Chinnaswamy Stadium demands the highest threshold of runs to secure victory safely!</span></b>',
        yaxis_title='Venue',
        xaxis_title='Safe Score Batting First',
        legend=dict(x=0.7, y=0.95),
        height=400, width=650,
        annotations=[
            dict(
                x=score,
                y=venue,
                text=str(round(score)),
                xanchor='left',
                yanchor='middle',
                showarrow=False,
                font=dict(color='black', size=12),
            ) for venue, score in zip(sorted_avg_win_score.index, sorted_avg_win_score.values)
        ]
    )

    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                    yaxis=dict(showgrid=True, gridcolor='#f0ebeb'))
    
    return fig

fig_bat_first = bat_first_stadium(df)


def ball_first_stadium(df):
    top_10_stadiums = list(df['Venue'].value_counts().index[:9])
    top_10_stadium_data = df[df['Venue'].isin(top_10_stadiums)]
    top_10_stadium_data = top_10_stadium_data[top_10_stadium_data['innings'] == 1]

    top_10_stadium_data['bat_first_wins'] = np.where(top_10_stadium_data['BattingTeam'] == top_10_stadium_data['WinningTeam'], 1, 0)

    ball_first_wins = top_10_stadium_data[top_10_stadium_data['bat_first_wins'] == 0]

    count_matches_per_stadium = ball_first_wins.groupby('Venue')['ID'].nunique()
    avg_win_score = ball_first_wins.groupby('Venue')['total_run'].sum() / count_matches_per_stadium
    sorted_avg_win_score = avg_win_score.sort_values(ascending=False)

    fig = go.Figure()

    # Create a list of marker colors based on the venue name
    marker_colors = '#9B3922'
    marker_opacities = [0.4 if venue != 'Feroz Shah Kotla' else 0.9 for venue in sorted_avg_win_score.index]

    fig.add_trace(go.Bar(
        y=sorted_avg_win_score.index,
        x=sorted_avg_win_score.values,
        name='Confidence Interval',
        marker_color=marker_colors,
        marker_opacity=marker_opacities,
        orientation='h'
    ))

    # Update layout to increase height
    fig.update_layout(
        title='Venue Wise Safe Score to Chase<br><b><span style="font-size:14px">Struggle for Bowlers: At Feroz Shah Kotla, low opposition scores secure chase victories!</span></b>',
        yaxis_title='Venue',
        xaxis_title='Safe Score Batting First',
        legend=dict(x=0.7, y=0.95),
        height=400, width=650,
        annotations=[
            dict(
                x=score,
                y=venue,
                text=str(round(score)),
                xanchor='left',
                yanchor='middle',
                showarrow=False,
                font=dict(color='black', size=12),
            ) for venue, score in zip(sorted_avg_win_score.index, sorted_avg_win_score.values)
        ]
    )

    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                    yaxis=dict(showgrid=True, gridcolor='#f0ebeb'))

    return fig

fig_ball_first = ball_first_stadium(df)


def create_toss_plot(df):
    df['Toss Winner = Match Winner'] = df.apply(lambda row: 1 if row['TossWinner'] == row['WinningTeam'] else 0, axis=1)
    # Calculate whether TossWinner was not the WinningTeam
    df['Toss Winner = Match Loser'] = df.apply(lambda row: 1 if row['TossWinner'] != row['WinningTeam'] else 0, axis=1)
    df_ = df[df['MatchNumber'] == 'Eliminator']

    # Sum up the counts
    result_overall = df_[['Toss Winner = Match Winner', 'Toss Winner = Match Loser']].sum()  # time taking so store

    knockout_df = df[df['MatchNumber'] == 'Qualifier 1']
    knockout_df['Toss Winner = Match Winner'] = knockout_df.apply(lambda row: 1 if row['TossWinner'] == row['WinningTeam'] else 0, axis=1)
    # Calculate whether TossWinner was not the WinningTeam
    knockout_df['Toss Winner = Match Loser'] = knockout_df.apply(lambda row: 1 if row['TossWinner'] != row['WinningTeam'] else 0, axis=1)

    # Sum up the counts
    result_knockout = knockout_df[['Toss Winner = Match Winner', 'Toss Winner = Match Loser']].sum()  # time taking so store

    labels_overall = result_overall.index
    sizes_overall = result_overall.values
    labels_final = result_knockout.index
    sizes_final = result_knockout.values

    palette = ['#00B1D2FF', '#FDDB27FF']

    # Plotting
    fig = go.Figure()

    # First Pie Plot
    fig.add_trace(go.Pie(
        labels=labels_overall,
        values=sizes_overall,
        hole=0.3,
        marker=dict(colors=palette),
        domain=dict(x=[0, 0.5]),
        name="Overall",
        textinfo='percent',
        textfont_size=14,
        sort=False
    ))

    # Second Pie Plot
    fig.add_trace(go.Pie(
        labels=labels_final,
        values=sizes_final,
        hole=0.3,
        marker=dict(colors=palette),
        domain=dict(x=[0.5, 1.0]),
        name="Knockouts",
        textinfo='percent',
        textfont_size=14,
        sort=False,
        pull=[0.1, 0]
    ))

    # Update layout
    fig.update_layout(
        title='Comparison of Toss Impact<br><b><span style="font-size:14px">Toss does not sway results overall, but in knockouts, it is a game-changer!</span></b><br><br>',
        title_font_size=20,
        annotations=[
            dict(text='Overall', x=0.2, y=0, showarrow=False, font=dict(size=14)),
            dict(text='Knockouts', x=0.84, y=0, showarrow=False, font=dict(size=14))
        ]
    )

    return fig

fig_toss = create_toss_plot(df)


def over_wise_runs(df):
    df_ = df[df['ballnumber'].between(1, 6)]
    runs_data = df_.groupby(['overs', 'ballnumber'])['total_run'].sum().reset_index()

    heatmap_data = runs_data.pivot(index='overs', columns='ballnumber', values='total_run').fillna(0)
    heatmap_data.index = heatmap_data.index.astype(str)
    heatmap_data.columns = heatmap_data.columns.astype(str)


    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values[::-1],
        x=list(heatmap_data.columns),
        y=list(heatmap_data.index[::-1]),
        colorscale='RdBu_r',
        colorbar=dict(title='Runs')
    ))

    # Update layout
    fig.update_layout(
        title='Runs scored in each ball of the over<br><b><span style="font-size:14px">The death overs and last powerplay over have the max runs scored!</span></b>',
        xaxis_title='Ball Number',
        yaxis_title='Overs',
        height=700, width=600
    )

    fig.add_shape(type="rect",
                x0=-0.7, y0=len(heatmap_data) - 6.7, x1=5.7, y1=len(heatmap_data) - 5.3,
                line=dict(color="black", width=2),
                fillcolor="LightSalmon", opacity=0.2)

    fig.add_annotation(x=2.5, y=len(heatmap_data) - 6,
                    text="Last Powerplay Over",
                    showarrow=False,
                    font=dict(size=15, color="#FFF7FC"), opacity=0.8)


    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                    yaxis=dict(showgrid=True, gridcolor='#f0ebeb'))
    # Show the plot
    return fig

fig_over_runs = over_wise_runs(df)


def over_wise_wickets(df):
    df_ = df[df['ballnumber'].between(1, 6)]

    # Group by overs and balls, count wickets
    wickets_data = df_.groupby(['overs', 'ballnumber'])['isWicketDelivery'].sum().reset_index()

    # Pivot the DataFrame to have overs on the y-axis, balls on the x-axis, and wickets taken as values
    heatmap_data = wickets_data.pivot(index='overs', columns='ballnumber', values='isWicketDelivery').fillna(0)

    # Convert index and columns to string
    heatmap_data.index = heatmap_data.index.astype(str)
    heatmap_data.columns = heatmap_data.columns.astype(str)

    # Create a heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values[::-1],  # Reverse the rows to have overs in ascending order
        x=list(heatmap_data.columns),
        y=list(heatmap_data.index[::-1]),
        colorscale='RdBu_r',  # Choose the colorscale
        colorbar=dict(title='Wickets Taken')  # Add colorbar with title
    ))

    # Update layout
    fig.update_layout(
        title='Wickets taken in each ball of the over<br><b><span style="font-size:14px">Most teams loose their wickets in the death overs!</span></b>',
        xaxis_title='Ball Number',
        yaxis_title='Overs',
        height=700, width=600
    )

    fig.add_shape(type="rect",
                x0=-0.7, y0=len(heatmap_data) - 17.2, x1=5.7, y1=len(heatmap_data) - 20.7,
                line=dict(color="black", width=2),
                fillcolor="LightSalmon", opacity=0.15)

    fig.add_annotation(x=2.5, y=len(heatmap_data) - 19,
                    text="Death Overs",
                    showarrow=False,
                    font=dict(size=17, color="#378CE7"), opacity=0.8)


    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                    yaxis=dict(showgrid=True, gridcolor='#f0ebeb'))

    # Show the plot
    return fig

fig_over_wickets = over_wise_wickets(df)


def top_batters(df):
    batsman_runs = df.groupby('batter')['batsman_run'].sum().reset_index()

    # Sort the DataFrame by total runs scored in descending order
    top_20_scorers = batsman_runs.sort_values(by='batsman_run', ascending=False).head(20).sort_values(by='batsman_run', ascending=True)

    # Set the opacity values for specific batters
    marker_color = ['#15F5BA' if batsman in ['V Kohli', 'RG Sharma', 'MS Dhoni'] else '#8CB9BD' for batsman in top_20_scorers['batter']]
    opacity_values = [0.8 if batsman in ['V Kohli', 'RG Sharma', 'MS Dhoni'] else 0.5 for batsman in top_20_scorers['batter']]

    fig = go.Figure(go.Bar(
        y=top_20_scorers['batter'],  # Batsmen names on y-axis for horizontal orientation
        x=top_20_scorers['batsman_run'],  # Total runs on x-axis
        marker_color=marker_color,  # Bar color
        marker_opacity=opacity_values,  # Opacity values for markers
        text=top_20_scorers['batsman_run'],  # Text to be displayed on hover
        textposition='outside',  # Position of the hover text
        orientation='h',  # Horizontal orientation
    ))

    # Update layout
    fig.update_layout(
        title='Top 20 Run Scorers<br><b><span style="font-size:14px">Jersey Number 7, 18, 45 all present!</span></b>',
        xaxis_title='Total Runs',
        yaxis_title='Batsmen',
        template='plotly_white',  # White background template
        height=600
    )

    return fig

fig_top_batters = top_batters(df)


def top_bowlers(df):
    # Grouping by bowler and counting the number of wickets taken
    bowler_wickets = df[df['isWicketDelivery'] == 1].groupby('bowler')['isWicketDelivery'].count().reset_index()
    bowler_wickets.columns = ['bowler', 'wickets_taken']

    # Sorting the DataFrame by total wickets taken in descending order
    top_20_bowlers = bowler_wickets.sort_values(by='wickets_taken', ascending=False).head(20).sort_values(by='wickets_taken', ascending=True)

    # Create a horizontal bar plot
    fig = go.Figure(go.Bar(
        y=top_20_bowlers['bowler'],  # Bowlers' names on y-axis for horizontal orientation
        x=top_20_bowlers['wickets_taken'],  # Total wickets on x-axis
        marker_color='#9F70FD',  # Bar color
        text=top_20_bowlers['wickets_taken'],  # Text to be displayed on hover
        textposition='outside',  # Position of the hover text
        orientation='h',  # Horizontal orientation
    ))

    # Update layout
    fig.update_layout(
        title='Top 20 Wicket Takers<br><b><span style="font-size:14px">Top 2 bowlers are not from India!</span></b>',
        xaxis_title='Total Wickets',
        yaxis_title='Bowlers',
        template='plotly_white',  # White background template
        height=600
    )

    fig.add_shape(type="rect",
                  x0=-5, y0=17.5,
                  x1=215, y1=19.5,
                  line=dict(color="green", width=2), fillcolor="green", opacity=0.1)

    return fig

fig_top_bowlers = top_bowlers(df)















layout = html.Div([
    html.Div([
        html.H1("IPL: Inside the Matches"),
        
        html.H3("Explore the highs, lows, and unforgettable moments of the IPL through captivating data visualizations."),
        html.Div([
            html.Img(src="assets/ipl_logo.png", id="ipl-logo")
        ], id="logo"),
        html.Div([
            html.P([
                html.Span('A'),
                "s anticipation builds for the next IPL season, our website offers an immersive journey into the heart of cricket's most thrilling tournament, exploring its evolution, iconic moments, and standout achievements."
            ]),
            html.P([
                "Embark on a journey with us as we unravel the data behind the IPL, offering insights, perspectives, and a deeper understanding of cricket's most electrifying tournament. Whether you're a die-hard fan, a casual observer, or simply captivated by the excitement of the game, our visual exploration of the IPL guarantees to inform, engage, and inspire."
            ])
        ], id='explain')
    
    ], id="o-v-heading"),

    html.Div(
        className='o-v-graph-map',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.H2("Optimal Batting Lineup: Assigning Positions to Batters"),
                    html.P("In crafting the optimal batting lineup, it's essential to analyze both the strike rate and average of batters to determine their suitability for the crucial death overs and middle overs. By considering these metrics, we ensure the right players are in the right positions to maximize scoring potential and team performance.")
                ]
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_avg_sr)
                ]
            ),
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.P("Batters situated in the 1st Quadrant emerge as the top performers, boasting exceptional averages and strike rates, making them true match-winners. Conversely, those in the 2nd Quadrant excel as aggressive hitters, ideal for the high-pressure death overs. Meanwhile, players in the 3rd Quadrant showcase solid averages but slower strike rates, making them well-suited for anchoring the innings during the middle overs.")
                ]
            )
        ]
    ),

    html.Div(
        className='o-v-graph-map',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.H2("Unraveling Super Overs"),
                    html.P("Dive into the excitement of Super Overs, those thrilling moments where match results are decided after a tie. We meticulously analyze every Super Over bowled to date, unveiling their impact over time.")
                ]
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_super_over)
                ]
            ),
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.P("While Super Overs have been relatively rare, the intensity peaked in 2020 with three occurring in a single month. In the past, Super Overs often resulted in decisive victories, but recent trends show a shift towards much tighter contests.")
                ]
            )
        ]
    ),

    html.Div(
        className='o-v-graph-map',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.H2("Deciphering Teams' Performance in Crucial Knockout Matches"),
                    html.P("Delve into a thorough analysis of teams' performances in pivotal knockout matches, uncovering the strategies, strengths, and pivotal moments that determine success. Explore how the best teams rise to the occasion when the stakes are highest.")
                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_wordcloud)
                ]
            ),

            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_knockout_team)
                ]
            ),
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.P("CSK and MI lead with the most knockout appearances. Among the older teams, RCB struggles while MI shines. Gujarat made a historic IPL debut, clinching victory in their inaugural season.")
                ]
            )
        ]
    ),

    html.Div(
        className='o-v-graph-map',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.H2("Does the Toss actually matter?"),
                    html.P("Uncover the truth behind cricket's age-old debate: Does winning the toss affect the outcome of the game? Explore the data and find out if the coin toss holds the key to victory.")
                ]
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_toss)
                ]
            ),
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.P("Clearly, when considering all matches played to date, there is approximately a 50% chance that the toss winner will also win the match. However, in nail-biting knockout matches, toss winners emerge victorious in about two-thirds of the games.")
                ]
            )
        ]
    ),

    html.Div(
        className='o-v-graph-map',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.H2("Stadium Secrets: Worst for Chasing vs. Defending Teams"),
                    html.P("Explore through data visualization the stadium patterns favoring teams batting first or second after winning the toss, unveiling the best and worst choices for each strategy.")
                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_bat_first)
                ]
            ),

            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_ball_first)
                ]
            ),

            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_bat_wins)
                ]
            ),

            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.P("Wankhede stadium in Mumbai is the best to opt bowling since, even if the opposition scores 156 runs, most of the time it is easily chased. While, MA Chidambaram in Chennai turns out to be the safest to bat first, since it demands minimum first innings total to secure victory. One must bat first when playing in Chidambaram as it ensures victory 61% of times.")
                ]
            )
        ]
    ),

    html.Div(
        className='o-v-graph-map',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.H2("Unraveling Cricket's Gold Mines: Most Productive Overs and Balls"),
                    html.P("Dive into the heart of cricket's action with our analysis on the most prolific overs and balls, revealing where runs flow freely and wickets tumble frequently. Explore the game's critical moments and uncover the strategic nuances behind each delivery.")
                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_over_runs)
                ]
            ),

            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_over_wickets)
                ]
            ),
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.P("The visualizations above highlight a clear trend: during the last over of the powerplay, with fielders inside the circle, teams tend to score the most runs. Similarly, in the death overs, batters aim to maximize the score after building their innings, resulting in high run rates. Additionally, most wickets fall during the death overs as batsmen adopt an aggressive approach. Notably, the last ball of the innings sees the highest number of wickets, as players recognize the importance of scoring without hesitation, knowing there's no advantage in staying not out.")
                ]
            )
        ]
    ),

    html.Div(
        className='o-v-graph-map',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.H2("Top Performers in TATA IPL"),
                    html.P("In T20 cricket, the focus is on scoring runs and taking wickets. We track the top 20 run-scorers and wicket-takers of all time, to gain insights on the top performers.")
                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_top_batters)
                ]
            ),

            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_top_bowlers)
                ]
            ),
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.P("As expected in the 'Indian' Premier League, a significant portion of the top batters are Indian, alongside renowned international stars like Chris Gayle, David Warner, and AB de Villiers. Interestingly, the top two bowlers are not from India, highlighting the substantial impact international players have in the IPL.")
                ]
            )
        ]
    ),

    html.Div(children=[
        html.Div(
                className='o-v-graph-heading',
                children=[
                    html.H2("IPL Points Table"),
                    html.P("Select a season:"),
                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Dropdown(
                        id='season-dropdown',
                        options=[{'label': str(year), 'value': str(year)} for year in range(2008, 2023)],
                        value='2022'
                    ),
                    html.Div(id='points-table')
                ]
            ),
            
        
        
        
    ],className="o-v-graph-map", id="points-div")
])


@callback(
    Output('points-table', 'children'),
    [Input('season-dropdown', 'value')]
)
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
    df3.reset_index(inplace=True)

    return html.Table(
        className='styled-table',
        children=[
            html.Thead(
                className='thead-light',
                children=html.Tr(
                    [html.Th(col, scope='col') for col in df3.columns]
                )
            ),
            html.Tbody(
                children=[
                    html.Tr(
                        className='table-row',
                        children=[html.Td(df3.iloc[i][col]) for col in df3.columns]
                    ) for i in range(len(df3))
                ]
            )
        ]
    )