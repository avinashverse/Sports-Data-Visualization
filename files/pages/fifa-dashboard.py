import pandas as pd
import dash
import numpy as np
from dash import dcc, html, callback
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots


dash.register_page(__name__, path='/fifa-dashboard', name="fifa-dashboard")
worldcup = pd.read_csv('assets/WorldCups.csv')
matches = pd.read_csv('assets/WorldCupMatches.csv')

def create_cumulative_goals(matches):
    # Create a new DataFrame to store cumulative goals for each team across all years
    unique_years = sorted(matches['Year'].unique())
    unique_teams = matches['Home Team Name'].unique()

    # Create a new DataFrame to store cumulative goals for each team across all years
    cumulative_goals_all_years = pd.DataFrame(columns=['Year', 'Home Team Name', 'Goals'])

    # Iterate over each team
    for team in unique_teams:
        # Initialize the cumulative goals for the team
        cumulative_goals = 0

        # Iterate over each year
        for year in unique_years:
            # Check if the team participated in the year
            team_data = matches[(matches['Year'] == year) & (matches['Home Team Name'] == team)]

            if not team_data.empty:
                # Update the cumulative goals for the team
                cumulative_goals += team_data['Home Team Goals'].sum()

            # Add a row to the cumulative_goals_all_years DataFrame
            cumulative_goals_all_years = pd.concat([cumulative_goals_all_years, pd.DataFrame({'Year': [year], 'Home Team Name': [team], 'Goals': [cumulative_goals]})])

    # Reset the index of the cumulative_goals_all_years DataFrame
    cumulative_goals_all_years = cumulative_goals_all_years.reset_index(drop=True)

    # Create the Choropleth Map
    fig_cumulative_goals = px.choropleth(cumulative_goals_all_years, 
                                         locations='Home Team Name', 
                                         locationmode='country names', 
                                         color='Goals',
                                         hover_name='Home Team Name',
                                         animation_frame='Year',
                                         color_continuous_scale='Inferno_r',
                                         )

    # Update map layout
    fig_cumulative_goals.update_coloraxes(colorbar=dict(
            title="Goals",  # Set color bar title
            xanchor="left",  # Anchor color bar to the left side
            x=0,  # Set x-coordinate to center the color bar
            y=1.1,  # Set y-coordinate above the plot
            len=0.8,  # Set the length of the color bar
            thickness=20,  # Set the thickness of the color bar
            orientation="h",  # Set color bar orientation to horizontal
    ))
    fig_cumulative_goals.update_geos(
        visible=False,
        lataxis_range=[-60, 90],  # Set latitude axis range
        lonaxis_range=[-180, 180],  # Set longitude axis range
        projection_type="equirectangular",  # Set map projection
        showcountries=True,  # Show country borders
        countrycolor="white",  # Set country border color
        countrywidth=0.5,  # Set country border width
    )

    # Customize the layout
    fig_cumulative_goals.update_layout(
        geo=dict(
            showframe=False,  # Remove map boundaries
            showcoastlines=False,  # Remove coastlines
            bgcolor="black",  # Set background color to black
            projection_type="equirectangular",  # Set map projection
        ),
        plot_bgcolor='black',  # Set plot background color to black
        paper_bgcolor='black',  # Set paper background color to black
        font_color='white',
    )

    # Show the map
    return fig_cumulative_goals
cumulative_goals_fig=create_cumulative_goals(matches)
# fig_cumulative_goals=create_cumulative_goals_choropleth(matches)
def create_worldcup_subplots(worldcup_data, years):
    # Create subplots with 2 rows and 3 columns
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f'Year {year}' for year in years])

    # Loop through each year and add bar charts to the corresponding subplot
    for i, year in enumerate(years, 1):
        # Filter data for the current year
        filtered_data = worldcup_data[worldcup_data['Year'] == year]

        # Check if data is available for the current year
        if not filtered_data.empty:
            # Extract winner, runner-up, and third-place teams
            winner = filtered_data['Winner'].iloc[0]
            runner_up = filtered_data['Runners-Up'].iloc[0]
            third_place = filtered_data['Third'].iloc[0]

            # Add bar traces for winner, runner-up, and third-place teams
            fig.add_trace(go.Bar(name='Runner-Up', x=[runner_up], y=[2], text=[runner_up], textposition='auto', marker_color='silver'), row=(i-1)//3+1, col=(i-1)%3+1)
            fig.add_trace(go.Bar(name='Winner', x=[winner], y=[3], text=[winner], textposition='auto', marker_color='gold'), row=(i-1)//3+1, col=(i-1)%3+1)
            fig.add_trace(go.Bar(name='Third Place', x=[third_place], y=[1], text=[third_place], textposition='auto', marker_color='#CD7F32'), row=(i-1)//3+1, col=(i-1)%3+1)

            # Update layout for each subplot to remove x-axis and y-axis
            fig.update_xaxes(visible=False, row=(i-1)//3+1, col=(i-1)%3+1)
            fig.update_yaxes(visible=False, row=(i-1)//3+1, col=(i-1)%3+1)

            # Show legend only for the first subplot
            if i == 1:
                fig.update_traces(showlegend=True, row=(i-1)//3+1, col=(i-1)%3+1)
            else:
                fig.update_traces(showlegend=False, row=(i-1)//3+1, col=(i-1)%3+1)

            # Set text color for "Third Place" to black
            fig.update_traces(textfont_color='black', selector=dict(name='Runner-Up'))
            fig.update_traces(textfont_color='black', selector=dict(name='Winner'))
            fig.update_traces(textfont_color='black', selector=dict(name='Third Place'))
        else:
            # Add a placeholder trace if data is not available for the current year
            fig.add_trace(go.Bar(name=f'No Data ({year})', x=['No Data'], y=[0], marker_color='lightgray'), row=(i-1)//3+1, col=(i-1)%3+1)

    # Update the layout
    fig.update_layout(
        barmode='group', # Group bars in each subplot
        plot_bgcolor='black',   # Set plot background color to black
        paper_bgcolor='black',  # Set paper background color to black
        font_color='white', 
        legend=dict(
            orientation='h',  # Horizontal orientation for the legend
            yanchor='top',    # Anchor the legend to the top
            y=1.15            # Position the legend slightly above the plot
        ) # Set font color to white

    )

    # Show the chart
    return fig
winner_fig=create_worldcup_subplots(worldcup, [2014, 2010, 2006, 2002, 1998, 1994])
def plot_goals_comparison(matches, option):
    # Drop rows with NaN or infinite values in the 'Year' column
    df = matches.dropna(subset=['Year']).replace([np.inf, -np.inf], np.nan)

    # Convert 'Year' column to integers
    df['Year'] = df['Year'].astype(int)
    
    # Grouping by year and home/away teams to calculate total goals
    if(option=='before halftime'):
        home_goals_by_year = df.groupby('Year')['Half-time Home Goals'].sum().reset_index()
        away_goals_by_year = df.groupby('Year')['Half-time Away Goals'].sum().reset_index()
        # Merging home and away goals data
        total_goals_by_year = pd.merge(home_goals_by_year, away_goals_by_year, on='Year', suffixes=('_Home', '_Away'))

        # Creating the bar chart
        fig = px.bar(total_goals_by_year, x='Year', y=['Half-time Home Goals', 'Half-time Away Goals'],
                     labels={'Year': 'Year', 'value': 'Total Goals', 'variable': 'Team'},
                     title='Total Goals Scored by Home vs. Away Teams Across Years in Half-time',
                     color_discrete_map={'Half-time Home Goals': '#f3ce49', 'Half-time Away Goals': '#d46313'},
                     barmode='group')
    else:
        home_goals_by_year = df.groupby('Year')['Home Team Goals'].sum().reset_index()
        away_goals_by_year = df.groupby('Year')['Away Team Goals'].sum().reset_index()
        # Merging home and away goals data
        total_goals_by_year = pd.merge(home_goals_by_year, away_goals_by_year, on='Year', suffixes=('_Home', '_Away'))

        # Creating the bar chart
        fig = px.bar(total_goals_by_year, x='Year', y=['Home Team Goals', 'Away Team Goals'],
                     labels={'Year': 'Year', 'value': 'Total Goals', 'variable': 'Team'},
                     
                     color_discrete_map={'Home Team Goals': 'gold', 'Away Team Goals': '#d46313'},
                     barmode='group')

        
    

   
    
    # Set plot background color to black, paper background color to black, and font color to white
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        legend=dict(
            orientation='h',  # Horizontal orientation for the legend
            yanchor='top',    # Anchor the legend to the top
            y=1.1,
            x=-0.1             # Position the legend slightly above the plot
        )
    )

    return fig
bar_fig=plot_goals_comparison(matches, "full")


df=worldcup
for index, row in df.iterrows():
    df.at[index, 'Winner'] = '0' + row['Winner']

nodes = pd.Index(df['Year'].unique().tolist() + df['Country'].unique().tolist() + df['Winner'].unique().tolist())
nodes_dict = dict(zip(nodes, range(len(nodes))))

# Create links
sources = []
labels=nodes.tolist()


for i in range(len(labels)):
    if isinstance(labels[i], str) and labels[i][0] == '0':
        labels[i] = labels[i][1:]

targets=[]
values=[]
for index, row in df.iterrows(): 
    sources.append(nodes_dict[row['Country']])
    targets.append(nodes_dict[row['Year']])
    values.append(1)
for index, row in df.iterrows(): 
    sources.append(nodes_dict[row['Year']])
    targets.append(nodes_dict[row['Winner']])
    values.append(1)

sankey_fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color='gold'
    )
)])

# Update layout
sankey_fig.update_layout(margin=dict(t=20, l=20,r=20,b=50),plot_bgcolor='black',  # Set the plot background color to black

    paper_bgcolor='black',  # Set the paper background color to black
    font_color='white',)

def create_top_10_goal_scoring_chart(matches):
    selected_stages = ['Quarter-finals', 'Semi-finals', 'Final']
    filtered_df = matches[matches['Stage'].isin(selected_stages)]

    # Group by Home Team Name and sum the goals for each stage
    goal_sums = filtered_df.groupby(['Stage', 'Home Team Name'])['Home Team Goals'].sum().reset_index()

    # Get the top 10 goal-scoring teams for each stage
    top_10_teams_per_stage = goal_sums.groupby('Stage').apply(lambda x: x.nlargest(10, 'Home Team Goals')).reset_index(drop=True)

    # Define custom colors for each stage
    stage_colors = {'Quarter-finals': 'gold',  # Orange
                    'Semi-finals': '#d46313',     # Blue
                    'Final':  '#9E6F21'}            # Green

    # Create a figure with subplots
    fig = make_subplots(rows=1, cols=len(selected_stages), subplot_titles=selected_stages)

    # Create a horizontal bar chart for each stage with custom colors
    for i, stage in enumerate(selected_stages):
        data = top_10_teams_per_stage[top_10_teams_per_stage['Stage'] == stage]

        fig.add_trace(go.Bar(
            x=data['Home Team Goals'],
            y=data['Home Team Name'],
            orientation='h',
            name=stage,
            marker_color=stage_colors[stage]  # Set custom color for the stage
        ), row=1, col=i+1)

    # Update the layout
    fig.update_layout(
 
   
        yaxis=dict(title='Team'),
        xaxis=dict(title='Goals'),
        plot_bgcolor='black',   # Set plot background color to black
        paper_bgcolor='black',  # Set paper background color to black
        font_color='white',
        legend=dict(
            orientation='h',     # Horizontal legend orientation
            yanchor='top',       # Anchor the legend to the top
            y=1.15,              # Position the legend slightly above the plot
            xanchor='center',    # Center the legend horizontally
            x=0.5,               # Position the legend in the center
            bgcolor='rgba(0,0,0,0)'  # Set legend background color to transparent
        )    # Set font color to white
    )

    return fig
teams_fig=create_top_10_goal_scoring_chart(matches)
def create_participating_teams_plot(data):
    # Create a line plot using Plotly Express for the line trace
    fig = px.line(data, x='Year', y='QualifiedTeams',
                  labels={'Year': 'Year', 'QualifiedTeams': 'Total Participating Teams'},
                  color_discrete_sequence=['gold'],  # Set line color to hex color code '#f3ce49'
                  )  # Disable legend display

    # Create a filled area under the curve trace
    fig.add_trace(go.Scatter(
        x=data['Year'],
        y=data['QualifiedTeams'],
        fill='tozeroy',  # Fill area to zero y
        mode='none',  # Plot as filled area only, without lines
        fillcolor='rgba(243,206,73,0.7)',  # Set fill color with transparency
        hoverinfo='skip'  # Skip hover info for the filled area trace
    ))

    # Update layout for better visualization and aesthetics
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',  # Display tick marks for each year
            dtick=4,  # Set tick interval to every 4 years for better readability
            title='Year',  # X-axis title
            tickfont=dict(size=12),  # Adjust font size for x-axis ticks
            tickangle=45  # Rotate x-axis ticks by 45 degrees
        ),
        yaxis=dict(
            tickfont=dict(size=12)  # Adjust font size for y-axis ticks
        ),
        plot_bgcolor='black',  # Set plot background color to black
        paper_bgcolor='black',  # Set paper background color to black
        font=dict(color='white', size=12),  # Set font color and size

        showlegend=False  # Set title font size and family
    )

    return fig

line_fig=create_participating_teams_plot(worldcup)

dropdown_options = [
    {'label': 'Before Half Time', 'value': 'before_half_time'},
    {'label': 'Full', 'value': 'full'}
]

# Dropdown component
dropdown = dcc.Dropdown(
    id='time-dropdown',
    options=dropdown_options,
    value='before_half_time',  # Default value

)
layout = html.Div(
        id='f-d-page',
        children=[
            html.Div(
                id='f-d-row1',
                children=[
                    html.Div(
                        id='f-d-logo',
                        children=[
                            html.H2('FIFA'),
                            html.H2('World Cup'),
                            html.Img(src='assets/World Cup logo.png'),
                            html.H2('1930-2014')
                        ]
                    ),
                    html.Div(
                        className='f-d-graph-bar',
                        children=[
                            html.Div(
                                id='f-d-heading',
                                children=[
                                    html.H4('Home vs. Away Goals Over Time'),
                                     
                                ]
                            ),
                            html.Div(
                                id='f-d-plot',
                                    children=[
                                        dcc.RadioItems(
                                    id='time-radio',
                                    options=[
                                        {'label': 'Before Half Time', 'value': 'before_half_time'},
                                        {'label': 'Full', 'value': ''}  # Option for no selection
                                    ],
                                    value='',  # Default value (no option selected)
                                    labelStyle={'display': 'block'},
                                ),
                                    dcc.Graph(id='f-d-plot-graph-bar')
                                ]
                                
                            )
                        ]
                    ),
                    html.Div(
                        className='f-d-graph-sankey',
                        children=[
                            html.Div(
                                id='f-d-heading',
                                children=[
                                    html.H4('Connection of Events')
                                ]
                            ),
                            html.Div(
                                id='f-d-plot',
                                children=[
                                    html.Div(
                                        id='f-d-plot-heading',
                                        children=[
                                            html.H3('Host'),
                                            html.H3('Year',id="mid"),
                                            html.H3('Winner'),

                                        ]
                                        
                                    ),

                                    dcc.Graph(id='f-d-plot-graph', figure=sankey_fig)
                                ]
                                
                            )
                        ]
                    )
                ]
            ),
            html.Div(
                id='f-d-row2',
                children=[
                    
                    html.Div(
                        className='f-d-graph-winner',
                        children=[
                            html.Div(
                                id='f-d-heading',
                                children=[
                                    html.H4('World Cup Winners, Runners-Up, and Third-Place Teams for Different Years')
                                ]
                            ),
                            html.Div(
                                id='f-d-plot',
                                children=[
                                    dcc.Graph(id='f-d-plot-graph', figure=winner_fig)
                                ]
                                
                            )
                        ]
                    ),
                    html.Div(
                        className='f-d-graph-map',
                        children=[
                            html.Div(
                                id='f-d-heading',
                                children=[
                                    html.H4('Cumulative Goals Scored by Country (Over Years)')
                                ]
                            ),
                            html.Div(
                                id='f-d-plot',
                                children=[
                                    dcc.Graph(id='cumulative-goals-map', figure=cumulative_goals_fig),
                                    
                                ]
                                
                            )
                        ]
                    )
                ]
            ),
            html.Div(
                id='f-d-row3',
                children=[
                    html.Div(
                        className='f-d-graph-line',
                        children=[
                            html.Div(
                                id='f-d-heading',
                                children=[
                                    html.H4('Total Participating Teams')
                                ]
                            ),
                            html.Div(
                                id='f-d-plot',
                                children=[
                                    dcc.Graph(id='f-d-plot-graph', figure=line_fig),
                                    
                                ]
                                
                            )
                        ]
                    ),
                    html.Div(
                        className='f-d-graph-teams',
                        children=[
                            html.Div(
                                id='f-d-heading',
                                children=[
                                    html.H4('Top 10 Goal-Scoring Teams in Different Rounds')
                                ]
                            ),
                            html.Div(
                                id='f-d-plot',
                                children=[
                                    dcc.Graph(id='f-d-plot-graph', figure=teams_fig)
                                ]
                                
                            )
                        ]
                    )
                    
                ]
            )
        ]
    )

@callback(
    Output('f-d-plot-graph-bar', 'figure'),
    Input('time-radio', 'value')
)
def plot_goals_comparison(input_value, matches=matches):
    # Drop rows with NaN or infinite values in the 'Year' column
    df = matches.dropna(subset=['Year']).replace([np.inf, -np.inf], np.nan)

    # Convert 'Year' column to integers
    df['Year'] = df['Year'].astype(int)
    
    # Grouping by year and home/away teams to calculate total goals
    if input_value == 'before_half_time':
        home_goals_by_year = df.groupby('Year')['Half-time Home Goals'].sum().reset_index()
        away_goals_by_year = df.groupby('Year')['Half-time Away Goals'].sum().reset_index()
        # Merging home and away goals data
        total_goals_by_year = pd.merge(home_goals_by_year, away_goals_by_year, on='Year', suffixes=('_Home', '_Away'))

        # Creating the bar chart
        fig = px.bar(total_goals_by_year, x='Year', y=['Half-time Home Goals', 'Half-time Away Goals'],
                     labels={'Year': 'Year', 'value': 'Total Goals', 'variable': 'Team'},
                     color_discrete_map={'Half-time Home Goals': '#f3ce49', 'Half-time Away Goals': '#d46313'},
                     barmode='group')
    else:
        home_goals_by_year = df.groupby('Year')['Home Team Goals'].sum().reset_index()
        away_goals_by_year = df.groupby('Year')['Away Team Goals'].sum().reset_index()
        # Merging home and away goals data
        total_goals_by_year = pd.merge(home_goals_by_year, away_goals_by_year, on='Year', suffixes=('_Home', '_Away'))

        # Creating the bar chart
        fig = px.bar(total_goals_by_year, x='Year', y=['Home Team Goals', 'Away Team Goals'],
                     labels={'Year': 'Year', 'value': 'Total Goals', 'variable': 'Team'},
                     color_discrete_map={'Home Team Goals': 'gold', 'Away Team Goals': '#d46313'},
                     barmode='group')

    # Set plot background color to black, paper background color to black, and font color to white
    fig.update_traces(name='Home Team Goals', selector=dict(name='Half-time Home Goals' if input_value == 'before_half_time' else 'Home Team Goals'))
    fig.update_traces(name='Away Team Goals', selector=dict(name='Half-time Away Goals' if input_value == 'before_half_time' else 'Away Team Goals'))
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        legend=dict(
            orientation='h',  # Horizontal orientation for the legend
            yanchor='top',    # Anchor the legend to the top
            y=1.1,
            x=-0.1             # Position the legend slightly above the plot
        )
    )

    return fig
