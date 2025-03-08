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

dash.register_page(__name__, path='/Olympics-page1', name="Olympics-page1")

lat_lng = pd.read_csv('assets/olym.csv')
athlete_events = pd.read_csv('assets/athlete_events.csv')
hundred_m = pd.read_csv('assets/100m_records_only.csv', sep=';')
end_date = datetime(2024, 7, 26, 0, 0, 0)  # July 26, 2024 at 00:00:00 UTC

def get_time_remaining():
    now = datetime.utcnow()
    remaining = end_date - now
    return remaining
def create_olympics_map(data):
    def country_to_continent(country_name):
        try:
            country_alpha2 = pc.country_name_to_country_alpha2(country_name)
            country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
            country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
        except:
            country_continent_name = 'Europe'
        return country_continent_name
    
    data['Continent'] = data['Country'].apply(lambda x: country_to_continent(x))
    data['Country_Count'] = data.groupby('Country')['Year'].transform('count')

    # Create the map
    fig = px.scatter_geo(data, lon='Longitude', lat='Latitude',
                         color='Continent', size='Country_Count', hover_data={'Year': True, 'Longitude': False, 'Latitude': False, 'Country_Count': False, 'Continent': False, 'Country': True},
                         hover_name='City', projection='natural earth',
                         color_discrete_sequence=['#D6589F', '#8576FF', '#7BC9FF', '#A3FFD6', '#803D3B'],
                        )

    # Update the layout
    fig.update_layout(
        title='Olympics Host Cities and Countries<br><b><span style="font-size:14px">Most Olympics are held in American Continent, No Olympics in Africa</span></b>',
        coloraxis_colorbar=dict(title='Country'),
    )
    fig.update_traces(marker=dict(line=dict(color='black', width=1)))
    
    return fig
fig_o_map = create_olympics_map(lat_lng)


def plot_top_events_by_sport(df):
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
        '#5254a3', '#63707d', '#8ca252', '#b5cf6b', '#8c6d31',
        '#bd9e39', '#8c6d31', '#9c9ede', '#cedb9c', '#8c6d31'
    ]

    top_5_sports = df.groupby('Sport').size().sort_values(ascending=False).head(5).reset_index()['Sport'].tolist()

    # Filter the data for the top 5 sports
    top_5_sports_data = df[df['Sport'].isin(top_5_sports)]

    # Count the number of rows for each Sport and Event combination
    sport_event_counts = top_5_sports_data.groupby(['Sport', 'Event']).size().reset_index(name='Number of Participants')

    # Filter the top 5 events for each sport
    top_5_events = sport_event_counts.groupby('Sport')['Number of Participants'].nlargest(5).reset_index().merge(sport_event_counts, on=['Sport', 'Number of Participants'], how='inner')[['Sport', 'Event', 'Number of Participants']]

    # Sort the data by Sport and Count in descending order
    top_5_events = top_5_events.sort_values(['Sport', 'Number of Participants'], ascending=[True, False])

    fig = px.bar(top_5_events, y='Sport', x='Number of Participants', color='Event', title='Top 5 Events by Sport', barmode='stack',
                color_discrete_sequence=colors, width=800, height=700)
    fig.update_layout(xaxis_tickangle=-45, xaxis_categoryorder='total ascending')

    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      yaxis=dict(showgrid=True, gridcolor='#f0ebeb'))
    return fig
fig_o_top_events=plot_top_events_by_sport(athlete_events)


def visualize_medal_counts_by_continent(df):
    def country_to_continent(country_name):
        try:
            country_alpha2 = pc.country_name_to_country_alpha2(country_name)
            country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
            country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
        except:
            country_continent_name = 'North America'
        return country_continent_name

    df['Continent'] = df['Team'].apply(lambda x: country_to_continent(x))
    medal_counts = df.groupby(['Continent', 'Medal']).size().unstack(fill_value=0)

    # Select only the columns for Gold, Silver, and Bronze medals
    medal_counts = medal_counts[['Gold', 'Silver', 'Bronze']]

    # Reset the index to make 'Continent' a regular column
    medal_counts = medal_counts.reset_index()
    melted_df = pd.melt(medal_counts, id_vars=['Continent'], var_name='Medal', value_name='Count')

    gold_df = melted_df[melted_df['Medal'] == 'Gold']
    silver_df = melted_df[melted_df['Medal'] == 'Silver']
    bronze_df = melted_df[melted_df['Medal'] == 'Bronze']

    gold_df = gold_df.sort_values('Count', ascending=False)
    silver_df = silver_df.sort_values('Count', ascending=False)
    bronze_df = bronze_df.sort_values('Count', ascending=False)

    custom_colors = ['#ef476f', '#f78c6b', '#ffd166', '#06d6a0', '#118ab2', '#073b4c'][::-1]

    fig = make_subplots(rows=1, cols=3, subplot_titles=('Gold Medals', 'Silver Medals', 'Bronze Medals'),
                        specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]])

    fig.add_trace(go.Pie(labels=gold_df['Continent'], values=gold_df['Count'], name='', marker=dict(colors=custom_colors), hole=0.6),
                  row=1, col=1)
    fig.add_trace(go.Pie(labels=silver_df['Continent'], values=silver_df['Count'], name='', marker=dict(colors=custom_colors), hole=0.6),
                  row=1, col=2)
    fig.add_trace(go.Pie(labels=bronze_df['Continent'], values=bronze_df['Count'], name='', marker=dict(colors=custom_colors), hole=0.6),
                  row=1, col=3)

    # Update layout
    fig.update_layout(title_text='North American and European Countries dominates the Medal Tally!')

    # Show the plot
    return fig
fig_o_medals=visualize_medal_counts_by_continent(athlete_events)


def visualize_participation_count_yearwise(df):
    summer_participation = df[df['Season'] == 'Summer']
    summer_participation['Year'] = summer_participation['Year']

    participation_count_yearwise_summer = summer_participation.groupby('Year').size().reset_index(name='Participant Count')
    missing_years_df = pd.DataFrame({'Year': [1916, 1940, 1944], 'Participant Count': [0, 0, 0]})

    participation_count_yearwise_summer = pd.concat([participation_count_yearwise_summer, missing_years_df])

    participation_count_yearwise_summer = participation_count_yearwise_summer.sort_values(by='Year')
    participation_count_yearwise_summer['Year'] = participation_count_yearwise_summer['Year'].astype(str)
    fig = px.bar(participation_count_yearwise_summer[::-1], y='Year', x='Participant Count',
                 title='Year-wise Count of Participation <br><b><span style="font-size:14px">No Olympics during World Wars</span></b>',
                 labels={'Participant Count': 'Participant Count', 'Year': 'Year'},
                 width=800, height=1200, color_discrete_sequence=['#1f91cf'])

    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                              yaxis=dict(showgrid=True, gridcolor='#f0ebeb'))

    fig.add_annotation(x=2400, y=25, text="World War I", showarrow=False,
                       font=dict(size=30, color='#909496'))
    fig.add_annotation(x=2500, y=18.5, text="World War II", showarrow=False,
                       font=dict(size=30, color='#909496'))

    return fig
fig_o_count=visualize_participation_count_yearwise(athlete_events)


def create_sport_participants_chart(df):
    sport_participants = df['Sport'].value_counts().reset_index()
    sport_participants.columns = ['Sport', 'Participant Count']
    sport_participants_sorted = sport_participants.sort_values(by='Participant Count', ascending=True)

    # Create the bar chart
    fig = px.bar(sport_participants_sorted, y='Sport', x='Participant Count', 
                 title='Count of Participants for Each Sport <br><b><span style="font-size:14px">Few sports have very Limited Participation</span></b>',
                 labels={'Participant Count': 'Participant Count', 'Sport': 'Sport'},
                 height=1200, width=800, color_discrete_sequence=['#41B06E'])

    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                              yaxis=dict(showgrid=True, gridcolor='#f0ebeb'))
    return fig
fig_o_part=create_sport_participants_chart(athlete_events)

def plot_female_participation_summer(df):
    total_participants = df.groupby('Year').size().reset_index(name='Total Count')

    # Calculate the female participation percentage
    female_participation_summer = df[(df['Sex'] == 'F') & (df['Season'] == 'Summer')].groupby('Year').size().reset_index(name='Female Count')
    female_participation_summer = pd.merge(female_participation_summer, total_participants, on='Year', how='left')
    female_participation_summer['Female Percentage'] = (female_participation_summer['Female Count'] / female_participation_summer['Total Count']) * 100
    female_participation_summer = female_participation_summer[['Year', 'Female Percentage']]

    # Add the missing year 1896 with 0% participation
    female_participation_summer = pd.concat([pd.DataFrame({'Year': [1896], 'Female Percentage': [0]}), female_participation_summer])

    # Create the area chart for Summer Olympics
    fig_summer = px.area(female_participation_summer, x='Year', y='Female Percentage',
                         title='Female Participation in Summer Olympics has been increasing',
                         color_discrete_sequence=['#C65BCF'])

    fig_summer.update_layout(title='Female Participation in Summer Olympics has been Increasing<br><b><span style="font-size:14px">1896: 0 Women Participation</span></b>')
    fig_summer.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                              yaxis=dict(showgrid=True, gridcolor='#f0ebeb'))

    return fig_summer
fig_o_summer=plot_female_participation_summer(athlete_events)

def plot_female_participation_winter(df):
    total_participants = df.groupby('Year').size().reset_index(name='Total Count')

    # Calculate the female participation percentage for Winter Olympics
    female_participation_winter = df[(df['Sex'] == 'F') & (df['Season'] == 'Winter')].groupby('Year').size().reset_index(name='Female Count')
    female_participation_winter = pd.merge(female_participation_winter, total_participants, on='Year', how='left')
    female_participation_winter['Female Percentage'] = (female_participation_winter['Female Count'] / female_participation_winter['Total Count']) * 100
    female_participation_winter = female_participation_winter[['Year', 'Female Percentage']]

    # Create the area chart for Winter Olympics
    fig_winter = px.area(female_participation_winter, x='Year', y='Female Percentage',
                         title='Female Participation in Winter Olympics has been increasing',
                         color_discrete_sequence=['#C65BCF'])

    fig_winter.update_layout(title='Female Participation in Winter Olympics has been Increasing<br><b><span style="font-size:14px">1994: Sudden rise in Women Participation</span></b>')
    fig_winter.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                              yaxis=dict(showgrid=True, gridcolor='#f0ebeb'))

    return fig_winter
fig_o_winter=plot_female_participation_winter(athlete_events)


def plot_unique_events_per_year(df):
    # Filter data for female athletes in Winter Olympics before 1994
    female_events_winter = df[(df['Sex'] == 'F') & (df['Season'] == 'Winter') & (df['Year'] < 1994)]

    # Group by year and count the number of unique events for each year
    unique_events_per_year = female_events_winter.groupby('Year')['Event'].nunique().reset_index(name='Unique Events')

    # Create the bar plot using Plotly Express
    fig = px.bar(unique_events_per_year, x='Year', y='Unique Events',
                 title='Number of Unique Events for Females Each Year in Winter Season<br><b><span style="font-size:14px">1994: Increase in women participation due to increase in Sport Events for Women</span></b>')
    
    # Update layout for better visualization
    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      yaxis=dict(showgrid=True, gridcolor='#f0ebeb'))
    
    # Show the plot
    return fig
fig_o_women_events=plot_unique_events_per_year(athlete_events)


def plot_total_medals_by_team_per_year(df):
    summer_df = df[df['Season'] == 'Summer']
    medals_df = summer_df[['Team', 'Year', 'Medal']].copy()
    medals_df['Medal'] = pd.Categorical(medals_df['Medal'], categories=['Gold', 'Silver', 'Bronze'])
    medals_count = medals_df.groupby(['Team', 'Year', 'Medal']).size().reset_index(name='Count')
    medals_pivot = medals_count.pivot_table(index=['Team', 'Year'], columns='Medal', values='Count', fill_value=0)
    medals_pivot = medals_pivot.reset_index()
    medals_pivot['Total'] = medals_pivot['Gold'] + medals_pivot['Silver'] + medals_pivot['Bronze']
    top_20_countries = medals_pivot.groupby('Team')['Total'].sum().nlargest(20).index
    medals_pivot_top_20 = medals_pivot[medals_pivot['Team'].isin(top_20_countries)]
    data_2d_top_20 = medals_pivot_top_20.set_index(['Team', 'Year'])['Total'].unstack().fillna(0).values
    fig = px.imshow(data_2d_top_20, x=medals_pivot_top_20['Year'].unique(), y=medals_pivot_top_20['Team'].unique(),
                    width=800, height=800, labels=dict(x="Year", y="Country", color="Total Medals"),
                    color_continuous_scale='RdBu_r')
    fig.update_traces(colorbar=dict(title='Total Medals'))
    fig.update_layout(title='Total Medals Won by Teams per Year',
                      xaxis=dict(title='Year'),
                      yaxis=dict(title='Team'))
    return fig
fig_o_medals_team=plot_total_medals_by_team_per_year(athlete_events)

def plot_total_medals_top_teams_per_sport(df):
    summer_df = df[df['Season'] == 'Summer']
    medals_df = summer_df[['Team', 'Sport', 'Medal']].copy()
    medals_df['Medal'] = pd.Categorical(medals_df['Medal'], categories=['Gold', 'Silver', 'Bronze'])

    medals_count = medals_df.groupby(['Team', 'Sport', 'Medal']).size().reset_index(name='Count')
    medals_pivot = medals_count.pivot_table(index=['Team', 'Sport'], columns='Medal', values='Count', fill_value=0)
    medals_pivot = medals_pivot.reset_index()
    medals_pivot['Total'] = medals_pivot['Gold'] + medals_pivot['Silver'] + medals_pivot['Bronze']
    top_20_countries = medals_pivot.groupby('Team')['Total'].sum().nlargest(30).index
    medals_pivot_top_20 = medals_pivot[medals_pivot['Team'].isin(top_20_countries)]
    top_15_sports = medals_pivot_top_20.groupby('Sport')['Total'].sum().nlargest(30).index
    medals_pivot_top_15_sports = medals_pivot_top_20[medals_pivot_top_20['Sport'].isin(top_15_sports)]
    data_2d_top_15_sports = medals_pivot_top_15_sports.set_index(['Team', 'Sport'])['Total'].unstack().fillna(0).values

    fig = px.imshow(data_2d_top_15_sports, x=medals_pivot_top_15_sports['Sport'].unique(), y=medals_pivot_top_15_sports['Team'].unique(),
                    width=850, height=800, labels=dict(x="Sport", y="Country", color="Total Medals"),
                    color_continuous_scale='RdBu_r')

    fig.update_traces(colorbar=dict(title='Total Medals', x=0.5, y=0.5))
    fig.update_layout(title='Total Medals Won by Teams (Top 30) per Sport<br><b><span style="font-size:14px">United States dominates in Athletics and Swimming</span></b>',
                      xaxis=dict(title='Sport'), yaxis=dict(title='Team'), coloraxis_showscale=True)
    return fig
fig_o_medals_sport=plot_total_medals_top_teams_per_sport(athlete_events)


def plot_us_medals_by_year_in_athletics(df):
    us_athlete = df[(df['Team'] == 'United States') & (df['Sport'] == 'Athletics')]
    medals_per_year = us_athlete.groupby('Year').size().reset_index(name='Total Medals')

    fig = px.area(medals_per_year, x='Year', y='Total Medals', 
                  title='Total Medals Won by United States Athletes in Athletics Each Season<br><b><span style="font-size:14px">1904 & 1912: Sudden Peak in Victory</span></b>',
                  color_discrete_sequence=['#C65BCF'])
    fig.update_xaxes(title_text='Year')
    fig.update_yaxes(title_text='Total Medals')

    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      yaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      showlegend=False)
    return fig
fig_o_us_athletics=plot_us_medals_by_year_in_athletics(athlete_events)


def plot_swimming_medals_analysis(df):
    us_swimmer = df[(df['Team'] == 'United States') & (df['Sport'] == 'Swimming')]
    medals_per_year = us_swimmer.groupby('Year').size().reset_index(name='Total Medals')

    fig = px.area(medals_per_year, x='Year', y='Total Medals', 
                  title='Total Medals Won by United States Athletes in Swimming Each Season<br><b><span style="font-size:14px">1968: Sudden Peak in Victory</span></b>',
                  color_discrete_sequence=['#C65BCF'])
    fig.update_xaxes(title_text='Year')
    fig.update_yaxes(title_text='Total Medals')

    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      yaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      showlegend=False)
    return fig
fig_o_us_swim=plot_swimming_medals_analysis(athlete_events)

def plot_athletics_medals_analysis(df):
    us_athlete = df[(df['Team'] == 'United States') & (df['Sport'] == 'Athletics')]
    athlete = us_athlete[us_athlete['Year'] == 1912]['Event'].value_counts()
    athlete.index = [x[16:] for x in athlete.index]
    fig = px.bar(athlete,
          title='Count of Medals won by United States in 1912 across Athletics Events<br><b><span style="font-size:14px">Most Medal were won in Running</span></b>',
          color_discrete_sequence=['#C65BCF'],
          height=600)

    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      yaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      showlegend=False)
    fig.add_annotation(x=5, y=11, text="Running Events", showarrow=False,
                       ax=30, ay=-40, font=dict(size=16, color='#7a5601'))
    fig.update_xaxes(title='Event')
    fig.update_yaxes(title='Number of Medals')
    return fig
fig_o_us_medals=plot_athletics_medals_analysis(athlete_events)
def plot_team_medals(df, team_name):
    team_df = df[(df['Season'] == 'Summer') & (df['Year'] >= 2000) & (df['Team'] == team_name)]
    team_df = team_df.groupby(['Year', 'Medal']).size().unstack().reset_index().fillna(0)
    team_df['Total Medal'] = team_df['Gold'] + team_df['Silver'] + team_df['Bronze']

    fig = px.bar(team_df, x='Year', y=['Gold', 'Silver', 'Bronze', 'Total Medal'], barmode='group',
                 color_discrete_sequence=['gold', 'silver', '#CD7F32', '#d894f7'])

    fig.update_layout(
        title=f'Medals won by {team_name} (2000 onwards)',
        xaxis_title='Year',
        yaxis_title='Number of Medals',
        legend_title='Medal Type'
    )

    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      yaxis=dict(showgrid=True, gridcolor='#f0ebeb'))
    return fig
fig_o_us_team_medal= plot_team_medals(athlete_events,'United States')

def create_age_plot(df):
    # Compute age statistics by year
    age_stats = df.groupby('Year').agg({'Age': ['min', 'max', 'mean']}).reset_index()
    min_age = age_stats['Age']['min']
    avg_age = age_stats['Age']['mean']
    max_age = age_stats['Age']['max']

    year = age_stats['Year']
    data = pd.DataFrame({'Year': year, 'Min Age': min_age, 'Avg Age': avg_age, 'Max Age': max_age})
    data_filtered = data[data['Year'] <= 1940]

    melted_data = pd.melt(data_filtered, id_vars=['Year'], var_name='Statistic', value_name='Age')

    melted_data['Year'] = melted_data['Year'].astype(str)

    line_traces = []

    # Group melted_data by 'Year'
    grouped_data = melted_data.groupby('Year')

    # Iterate over each group (year)
    for year, group in grouped_data:
        # Extract data for the three statistics
        min_age_data = group[group['Statistic'] == 'Min Age']
        avg_age_data = group[group['Statistic'] == 'Avg Age']
        max_age_data = group[group['Statistic'] == 'Max Age']

        # Create a line trace connecting the three points
        line_trace = go.Scatter(
            x=[year, year, year],
            y=[min_age_data['Age'].iloc[0], avg_age_data['Age'].iloc[0], max_age_data['Age'].iloc[0]],
            mode='lines',
            line=dict(color='#e1e2e3', width=15),
            showlegend=False,
            opacity=0.35
        )

        # Add the line trace to the list
        line_traces.append(line_trace)

    # Create the scatter plot
    scatter_plot = px.scatter(melted_data, x='Year', y='Age', color='Statistic',
                          title='From 10 to 97: Age is just a Number!',
                          labels={'Age': 'Age', 'Year': 'Year', 'Statistic': 'Statistic'},
                          color_discrete_sequence=['#b35d1b', '#859c3a', '#60d1c4'])

    # Create a list of all traces
    all_traces = list(scatter_plot.data) + line_traces

    # Create a new figure with the updated trace order
    fig = go.Figure(data=all_traces)

    # Update layout
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
        yaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
        showlegend=False,
        margin=dict(t=100),  # Increase the top margin
        title={
            'text': "From 10 to 97: Age is just a Number!",
            'y': 0.9,  # Adjust the title position (0 to 1)
            'x': 0.3,  # Adjust the title position (0 to 1)
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    fig.update_traces(marker=dict(size=15))

    fig.add_annotation(x=8, y=120, text="Oldest Athlete was <br><b>97</b> years old!",
                       font=dict(color="#6a6f75", size=12),
                       showarrow=False, align="center", ax=0, ay=-40)
    return fig

fig_o_age=create_age_plot(athlete_events)

def create_scatter_plot(df, team, sport):
    filtered_df=df[(df['Team'] == team) & (df['Sport'] == sport)]
    fig = px.scatter(filtered_df, y='Height', x='Weight', color='Sex', hover_data=['Name', 'Team'],
                     color_discrete_sequence=['#B3C8CF', '#D862BC'])

    # Update the layout
    fig.update_layout(
        title=f'Height vs. Weight for {team} participating in {sport}<br><b><span style="font-size:14px">Males generally have more weight and height</span></b>',
        xaxis_title='Height',
        yaxis_title='Weight',
        legend_title='Gender'
    )
    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      yaxis=dict(showgrid=True, gridcolor='#f0ebeb'))
    fig.update_traces(marker=dict(line=dict(color='black', width=1)))
    
    return fig

fig_o_badminton=create_scatter_plot(athlete_events, 'United States', 'Badminton')


def create_top_medal_winners_chart_us(df, team):
    team_data = df[df['Team'] == team]

    # Group by athlete name and count the number of medals won by each athlete
    athlete_medal_counts = team_data.groupby('Name')['Medal'].count().reset_index()

    # Sort the counts in descending order
    sorted_athletes = athlete_medal_counts.sort_values(by='Medal', ascending=False)

    # Get the top 5 athletes
    top_5_athletes = sorted_athletes.head(10)

    gold_list = []
    silver_list = []
    bronze_list = []
    sport_list = []

    for athlete in top_5_athletes.values:
        gold, silver, bronze = 0, 0, 0
        player_df = team_data[team_data['Name'] == athlete[0]]
        for medal in player_df['Medal']:
            if medal == 'Bronze':
                bronze += 1
            elif medal == 'Silver':
                silver += 1
            elif medal == 'Gold':
                gold += 1

        gold_list.append(gold)
        silver_list.append(silver)
        bronze_list.append(bronze)
        sport_list.append(player_df['Sport'].head(1).values[0])

    names = top_5_athletes.values.T[0]

    best_athlete_df = pd.DataFrame({
        'Name': names[::-1],
        'Gold': gold_list[::-1],
        'Silver': silver_list[::-1],
        'Bronze': bronze_list[::-1],
        'Sport': sport_list[::-1]
    })
    
    fig = px.bar(best_athlete_df, y='Name', x=['Bronze', 'Silver', 'Gold'], 
                 title=f'Top Medal Winners for {team}<br><b><span style="font-size:14px">Michael Phelps has most medals, including more golds than the second-best player\'s <br>total medal count</span></b>', barmode='stack',
                 hover_name='Name', hover_data={'Name': True, 'Sport': True},
                 color_discrete_sequence=['#CD7F32', 'silver', 'gold'],
                 height=400, width=800)

    # Update the layout
    fig.update_layout(xaxis_title='Athlete', yaxis_title='Number of Medals')

    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      yaxis=dict(showgrid=True, gridcolor='#f0ebeb'))
    
    return fig

def create_top_medal_winners_chart_india(df, team):
    team_data = df[df['Team'] == team]

    # Group by athlete name and count the number of medals won by each athlete
    athlete_medal_counts = team_data.groupby('Name')['Medal'].count().reset_index()

    # Sort the counts in descending order
    sorted_athletes = athlete_medal_counts.sort_values(by='Medal', ascending=False)

    # Get the top 5 athletes
    top_5_athletes = sorted_athletes.head(10)

    gold_list = []
    silver_list = []
    bronze_list = []
    sport_list = []

    for athlete in top_5_athletes.values:
        gold, silver, bronze = 0, 0, 0
        player_df = team_data[team_data['Name'] == athlete[0]]
        for medal in player_df['Medal']:
            if medal == 'Bronze':
                bronze += 1
            elif medal == 'Silver':
                silver += 1
            elif medal == 'Gold':
                gold += 1

        gold_list.append(gold)
        silver_list.append(silver)
        bronze_list.append(bronze)
        sport_list.append(player_df['Sport'].head(1).values[0])

    names = top_5_athletes.values.T[0]

    best_athlete_df = pd.DataFrame({
        'Name': names[::-1],
        'Gold': gold_list[::-1],
        'Silver': silver_list[::-1],
        'Bronze': bronze_list[::-1],
        'Sport': sport_list[::-1]
    })
    
    fig = px.bar(best_athlete_df, y='Name', x=['Bronze', 'Silver', 'Gold'], 
                 title=f'Top Medal Winners for {team}<br><b><span style="font-size:14px">All of them are from Hockey in India</span></b>', barmode='stack',
                 hover_name='Name', hover_data={'Name': True, 'Sport': True},
                 color_discrete_sequence=['#CD7F32', 'silver', 'gold'],
                 height=400, width=800)

    # Update the layout
    fig.update_layout(xaxis_title='Athlete', yaxis_title='Number of Medals')

    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      yaxis=dict(showgrid=True, gridcolor='#f0ebeb'))
    
    return fig
fig_o_india= create_top_medal_winners_chart_india(athlete_events, 'India')
fig_o_us= create_top_medal_winners_chart_us(athlete_events, 'United States')



def create_100m_world_records_plot(hundred_m):
    hundred_m = hundred_m.iloc[1:, :]
    hundred_m.columns = hundred_m.loc[1, :].values
    hundred_m = hundred_m.drop(1).reset_index(drop=True)
    hundred_m['Date'] = pd.to_datetime(hundred_m['Date'])
    hundred_m = hundred_m.sort_values(by='Date')
    hundred_m['duration'] = hundred_m['Date'].diff().shift(-1).dt.days / 365
    hundred_m['duration'] = hundred_m['duration'].fillna(14.7)
    hundred_m = hundred_m[['Time', 'Athlete', 'Nationality', 'Location of race', 'Date', 'duration']]
    hundred_m['Time'] = hundred_m['Time'].astype(str)
    hundred_m['duration'] = hundred_m['duration'].apply(lambda x: np.round(x, 2))
    hundred_m['Date'] = hundred_m['Date'].dt.strftime('%d %B, %Y')

    fig = px.scatter(hundred_m, x='Time', y='Athlete', size='duration',
                     hover_name='Athlete',
                     hover_data={'Athlete': False, 'Time': True, 'Date': True, 'duration': True},
                     title="Men's 100m World Records", width=800, height=1200,
                     color_discrete_sequence=['#ffd240'])

    # Customize the hover template
    hover_template = "<b>%{hovertext}</b><br><br>" + \
                     "Time: %{x} sec<br>" + \
                     "Date: %{customdata[0]}"
    hover_template += "<br>" + \
                      "Record holder for <b>%{customdata[1]:.2f}</b> years"

    fig.update_traces(customdata=hundred_m[['Date', 'duration']], hovertemplate=hover_template)

    # Reverse the y-axis order
    fig.update_yaxes(autorange="reversed")

    # Customize the layout
    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0ebeb'),
                      yaxis=dict(showgrid=True, gridcolor='#f0ebeb'))

    # Customize the marker style
    fig.update_traces(marker=dict(line=dict(color='#f5ae07', width=1), opacity=0.6))
    last_index = hundred_m.index[-1]
    fig.update_traces(marker=dict(color=['#bf8704' if i == last_index else '#ffd240' for i in range(len(hundred_m))],
                                  line=dict(color=['#7a5601' if i == last_index else '#f5ae07' for i in range(len(hundred_m))],
                                            width=1),
                      opacity=[1 if i == last_index else 0.6 for i in range(len(hundred_m))]))

    fig.add_annotation(x=15, y='Maurice Greene', text="Current Record", showarrow=False,
                       ax=30, ay=-40, font=dict(size=16, color='#7a5601'))

    return fig

fig_o_100=create_100m_world_records_plot(hundred_m)
layout = html.Div([
    html.Div([
        html.H1("Olympic Chronicles: A Visual Journey"),
        
        html.H3("Discover the progression, accomplishments, and standout moments of both the Winter and Summer Olympics using data visualization."),
        html.Div([
            html.Img(src="assets/olympics.png", id="logo")
        ], id="logo"),
        html.Div(id="countdown", children=[
            html.H2(id="countdown"),
            dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
            html.H3("until the 2024 Summer Olympics in Paris")
        ]),
        html.Div([
            html.P([
                html.Span('T'),
                'he Olympic Games captivate the world with their rich history and global participation, uniting nations in a celebration of athletic excellence. As we anticipate the 2024 Summer Olympics in Paris, the excitement and anticipation are palpable. Our website delves into the heart of these games, offering a journey through the evolution, achievements, and memorable moments of both the Winter and Summer Olympics.'
            ]),
            html.P([
                'Join us as we unravel the data behind the games, providing insights, perspectives, and a deeper understanding of what makes the Olympics a truly extraordinary event. Whether you\'re a dedicated sports enthusiast, a curious observer, or simply drawn to the magic of global competition, our visual journey through the Olympics promises to inform, engage, and inspire.'
            ])
        ], id='explain')
    
    ], id="o-v-heading"),
    html.Div(
        className='o-v-graph-map',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.H2('Unraveling the Olympic Journey'),
                    html.P('The Olympics transcend borders, uniting nations under the banner of sportsmanship and fair competition. Our exploration dives deep into the geographical tapestry of host cities, highlighting the diversity of landscapes and cultures that define each Olympic edition.')

                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_map)
                ]
                
            )
        ]
    ),
    html.Div(
        className='o-v-graph-top-events',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.P('Delving into the heart of Olympic sports, our analysis uncovers the top 5 events across various disciplines, shedding light on the most popular and fiercely contested competitions. By aggregating data on participant numbers, we unveil the pinnacle events that captivate audiences and athletes alike.')

                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_top_events)
                ]
                
            )
        ]
    ),
    html.Div(
        className='o-v-graph-medals',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.P('Uncover the geographical medal landscape of the Olympics with our interactive visualization. By mapping medal counts across continents, we highlight the dominance of North American and European countries in Olympic achievements. Delve into the data-driven narrative of Gold, Silver, and Bronze medals distribution, revealing fascinating insights into the global sporting arena. Experience the thrill of international competition and celebrate the diverse victories that shape the Olympic legacy.')

                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_medals)
                ]
                
            )
        ]
    ),
    html.Div(
        className='o-v-graph-count',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.H2('Exploring Olympic Event Popularity'),
                    html.P('Delving into the historical trajectory of Olympic participation unveils fascinating insights into the ebb and flow of global athletic engagement. Our analysis, focusing specifically on Summer Olympic Games, meticulously tracks the yearly count of participants, painting a vivid picture of how this grand event has evolved over time. Through meticulous data aggregation and visualization, we showcase the peaks and troughs of participation, highlighting significant moments such as the absence of Olympics during World War I and II. This exploration not only commemorates the spirit of sportsmanship but also serves as a testament to the resilience and continuity of the Olympic movement amidst historical challenges.')

                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_count)
                ]
                
            )
        ]
    ),
    html.Div(
        className='o-v-graph-parts',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.P('Next, explore the diverse landscape of Olympic sports through our analysis of participant counts across different disciplines. The bar chart above illustrates the number of participants in each sport, offering insights into the popularity and engagement levels across various athletic activities. From heavily contested sports with high participation rates to those with more limited engagement, this visual representation provides a comprehensive view of the sporting diversity within the Olympic Games.')

                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_part)
                ]
                
            )
        ]
    ),
    html.Div(
        className='o-v-graph-women',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.H2("Analyzing Trends in Women's Olympic Participation"),
                    html.P("Analyzing trends in female participation across different Olympic seasons provides valuable insights into the evolving landscape of women's sports. In both the Summer and Winter Olympics, there has been a significant increase in female participation over the years. By examining the percentage of female athletes relative to total participants, we uncover a pattern of growth and engagement within the Olympic sports community.")

                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_summer)
                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_winter)
                ]
                
            )
        ]
    ),
    html.Div(
        className='o-v-graph-women-events',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.P("The Winter Olympics have seen a remarkable increase in female participation, especially notable in 1994. This surge is attributed to the expansion of unique events designed for female athletes. The year 1994 symbolizes a pivotal shift towards gender equality in winter sports, showcasing a concerted effort to broaden opportunities and enhance women's engagement across a diverse range of winter sports disciplines.")

                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_women_events)
                ]
                
            )
        ]
    ),
    html.Div(
        className='o-v-graph-women',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.H2("Dynamics of Olympic Medal Achievement"),
                    html.P("Witness the evolution of global sporting prowess as we delve into the triumphs and trends of medal wins over the years and across sports disciplines. Discover the enduring dominance of North American and European nations, particularly the United States in Athletics and Swimming, showcasing their consistent excellence on the world stage. Uncover the intricate tapestry of Olympic achievements, celebrating the diversity and resilience of athletes and nations in shaping the enduring legacy of the Olympic movement.")

                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_medals_team)
                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_medals_sport)
                ]
                
            )
        ]
    ),
    html.Div(
        className='o-v-graph-women',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.H2("Exploring Olympic Domination: United States"),
                    html.P("Visualizing the dominance of United States athletes in Athletics, particularly in the Summer Olympics, provides a compelling narrative of consistent excellence. Our analysis reveals a remarkable trend of success, with notable peaks in 1904, 1912, and 1968, showcasing the resilience and prowess of American athletes on the global stage.")

                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_us_athletics)
                ]
                
            ),
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.P("Furthermore, a focused examination of the medals earned by United States athletes in the 1912 Olympics across different Athletics events illuminates a significant trend: the majority of medals were achieved in Running events. This observation underscores the prowess and specialization of American athletes in specific disciplines within Athletics, highlighting their exceptional performance and strategic focus in these events.")

                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_us_medals)
                ]
                
            ),
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.P("Analyzing the historical performance of United States athletes in Swimming provides valuable insights into their medal achievements over the years. By examining the total medals won by American swimmers each season, we can observe trends and notable moments of success. One such instance is the sudden peak in victory in 1968, which stands out as a remarkable year for American swimmers.")

                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_us_swim)
                ]
                
            ),
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.P("By focusing on the period from 2000 onwards, we can observe trends in the medal achievements across various Olympic seasons. This visualization showcases the distribution of Gold, Silver, and Bronze medals won by the United States each year, offering a comprehensive view of their success on the Olympic stage. The grouped bar chart effectively illustrates the team's consistency or fluctuations in medal counts over the years, highlighting their dominance in specific events or periods of exceptional performance.")

                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_us_team_medal)
                ]
                
            )
        ]
    ),
    html.Div(
        className='o-v-graph-women',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.H2("Exploring Physical Diversity in Olympics"),
                    html.P("Exploring age dynamics across Olympic history reveals intriguing trends. From the earliest recorded years to 1940, the age range of athletes fluctuates significantly, showcasing a diverse spectrum of participants. The scatter plot vividly portrays this fluctuation, with markers denoting minimum, average, and maximum ages for each year. The accompanying line traces offer a narrative thread, connecting these age milestones and highlighting patterns of age diversity over time. Notably, the plot underscores remarkable moments, such as the oldest athlete at 97 years old, emphasizing how age truly becomes just a number in the Olympic journey.")

                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_age)
                ]
                
            ),
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.P("This scatter plot illustrates the height-weight dynamics of United States Badminton athletes, highlighting gender-based differences. Males generally exhibit higher weights and heights than females, showcasing the team's physical diversity. The interactive plot allows for detailed exploration, revealing individual athlete information on hover. Such visualizations offer valuable insights into athlete composition and gender-based trends, aiding in performance analysis and strategic planning.")

                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_badminton)
                ]
                
            ),
            
        ]
    ),
    html.Div(
        className='o-v-graph-top-medals',
        children=[
            html.Div(
                className='o-v-graph-heading', 
                children=[
                    html.H2("Medal Mastery: Michael Fred's Triumph and the Hockey Legacy in India"),
                    html.P("The bar chart encapsulates the remarkable dominance of India's hockey legends, showcasing the top medal winners from the sport. Each stacked bar represents an athlete's bronze, silver, and gold medal tally, vividly depicted in vibrant colors. The clean design and hover functionality invite viewers to explore the stories behind these accomplished athletes. It offers a captivating insight into India's sporting prowess, revealing that all the top medal winners hail from the hockey discipline.")

                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_india)
                ]
                
            ),
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.P(" Next, we unveil a captivating snapshot of the United States' Olympic medal dominance, showcasing the top athletes who have etched their names in sporting history. Each stacked bar represents an individual athlete's remarkable medal haul, with vibrant colors depicting their gold, silver, and bronze tallies. Michael Fred's towering bar steals the show, highlighting his unparalleled achievement of not only amassing the most medals but also surpassing the second-best athlete's total medal count with his gold tally alone.")

                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_us)
                ]
                
            ),
            
        ]
    ),
    html.Div(
        className='o-v-graph-women',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.H2("Unraveling the Fastest: Men's 100m World Records"),
                    html.P("The Men's 100m World Records scatter plot offers intriguing insights into the evolution of sprinting excellence over time. A notable observation spans the dominance of Usain Bolt, who held the world record for an extended period starting from 2008, showcasing his unparalleled speed and consistency on the track. Additionally, the plot reveals instances where numerous athletes challenged and surpassed the once seemingly unbreakable barriers of 10.2 seconds and 9.9 seconds, reflecting the continuous pursuit of athletic excellence and human performance enhancement in sprinting."),
                    html.P("A fascinating trend emerges as Usain Bolt not only set but also surpassed his own world record on two separate occasions, highlighting his exceptional athleticism and ability to push the boundaries of human potential. Moreover, the plot captures a phase in history where the world record alternated between sprinting legends Carl Lewis and Leroy Burrell, showcasing a dynamic era of fierce competition and record-breaking performances among top sprinters. These observations collectively underscore the dynamic nature of sprinting history and the remarkable achievements that have shaped the Men's 100m World Records landscape over the years.")
                ]
                
            ),
            html.Div(
                className='o-v-graph-plot',
                children=[
                    dcc.Graph(className='o-v-graph-plot-graph', figure=fig_o_100)
                ]
                
            ),
            
            
        ]
    ),
    html.Div(
        className='o-v-graph-women',
        children=[
            html.Div(
                className='o-v-graph-heading',
                children=[
                    html.H2("Concluding Olympic Reflections"),
                    html.P("As we conclude this visual exploration of Olympic achievements and sporting legacies, one thing becomes abundantly clear: the Olympic Games are not just about athletic prowess; they are a testament to human resilience, dedication, and the pursuit of excellence. From the thrilling races on the track to the strategic maneuvers on the field, each Olympic event carries with it a story of triumph, determination, and the unyielding spirit of competition."),
                    html.P("As we eagerly await the 2024 Summer Olympics in Paris, let us carry forward the lessons learned from the Olympic Chronicles: a visual journey through the evolution, accomplishments, and standout moments of both the Winter and Summer Olympics. May these insights inspire us to reach new heights, push our boundaries, and embrace the true essence of sportsmanship and fair play."),
                    html.P("Join us in celebrating the magic of the Olympic movement, where nations unite, athletes shine, and dreams are realized on the grandest stage of all. Here's to the next chapter of Olympic excellence and the enduring legacy of the Olympic Games.")
                ]
                
            ),
            
            
            
        ]
    ),
    html.Div(id='ending'),
    


    
    
])

@callback(
    Output('countdown', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_countdown(n):
    remaining = get_time_remaining()
    days = remaining.days
    hours, remainder = divmod(remaining.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days}d {hours}h {minutes}m {seconds}s"
