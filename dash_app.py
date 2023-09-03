
import dash
#import dash_core_components as dcc
#import dash_html_components as html

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input, Output
from datetime import datetime, timedelta
import numpy as np




app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
df = pd.read_csv('data.csv')
df['DE'] = pd.to_datetime(df['DE'])

colors ={
    'background':'#0d253f',
    'text':'white',
    'border': 'solid 3px white',
    'border-radius':'10px',
    'box-shadow': '10px white'
}
style1 = style={'color': 'white'} #for the H3 elements
style2 = style={'font-size': '80px', 'margin': '20px', 'color': 'yellow'} #for the H2 elements
style3 = style={'padding':'5px','height':'190px','background-color': '#3b6ea5','margin':'10px', 'border': '1px solid black', 'border-radius': '5px', 'box-shadow': '0 4px 8px rgb(1, 1, 1)'} #for the class name
style4 = style={'padding':'5px','height':'300px','background-color': '#3b6ea5','margin':'0px', 'border': '1px solid black', 'border-radius': '5px', 'box-shadow': '0 4px 8px rgb(1, 1, 1)'} #for the class name
style5 = style={'padding':'5px','height':'630px','background-color': '#3b6ea5','margin':'0px', 'border': '1px solid black', 'border-radius': '5px', 'box-shadow': '0 4px 8px rgb(1, 1, 1)'} #for the class name
style6 = style={'padding':'5px','height':'307px','background-color': '#3b6ea5','margin':'0px', 'border': '1px solid black', 'border-radius': '5px', 'box-shadow': '0 4px 8px rgb(1, 1, 1)'} #for the class name

style_height= style={'height': '300px'}




# Helper function to calculate the number of visits for a given date range
def calculate_visits(date_range):
    if date_range == 'last_month':
        start_date = pd.to_datetime('today') - pd.DateOffset(months=1)
    elif date_range == 'last_6_months':
        start_date = pd.to_datetime('today') - pd.DateOffset(months=6)
    elif date_range == 'last_year':
        start_date = pd.to_datetime('today') - pd.DateOffset(years=1)
    else:
        start_date = df['DE'].min()

    return len(df[df['DE'] >= start_date])

# Helper function to calculate the average satisfaction for a given date range
def calculate_average_satisfaction(date_range):
    if date_range == 'last_month':
        start_date = pd.to_datetime('today') - pd.DateOffset(months=1)
    elif date_range == 'last_6_months':
        start_date = pd.to_datetime('today') - pd.DateOffset(months=6)
    elif date_range == 'last_year':
        start_date = pd.to_datetime('today') - pd.DateOffset(years=1)
    else:
        start_date = df['DE'].min()

    return df[df['DE'] >= start_date]['S'].mean()

# Create the circular chart for type of visit
def create_type_visite_chart(date_range,filtered_df):
    
    visite_counts = filtered_df['RV'].value_counts()
    type_visite_chart = go.Figure(data=[go.Pie(
        labels=visite_counts.index,
        values=visite_counts.values,
        hole=0.7,
        domain=dict(x=[0.3, 1]),
        marker=dict(
            colors=['red','blue'],
            
            line=dict(
                color='white',
                width=2
            ),)
    )])
    
    return type_visite_chart

# Create the circular chart for garantie ou non
def create_garantie_chart(date_range,filtered_df):
    #filtered_df = df[df['DE'] >= pd.to_datetime('today') - pd.DateOffset(years=1)]
    garantie_counts = filtered_df['G'].value_counts()
    garantie_chart = go.Figure(data=[go.Pie(
        labels=garantie_counts.index,
        values=garantie_counts.values,
        hole=0.7,
        domain=dict(x=[0, 1]),
        marker=dict(
            colors=['red','blue'],
            
            line=dict(
                color='white',
                width=2
            ),)
    )])
    
    return garantie_chart

### RADAR CHART ##########################


# Calculate the number of breakdowns for each type of breakdown and each type of car
breakdown_counts = df.groupby(['TV', 'TP', 'AF']).size().reset_index(name='Nombre de pannes')

radar_graph = go.Figure()
# Add the initial traces to the radar graph (use the first fabrication year as the initial data)
initial_year = df['AF'].min()
# Iterate over each type of car to add a radar trace for it
for car_type in df['TV'].unique():
    data = breakdown_counts[(breakdown_counts['TV'] == car_type) & (breakdown_counts['AF'] == initial_year)]
    radar_graph.add_trace(go.Scatterpolar(
        r=data['Nombre de pannes'],
        theta=data['TP'],
        fill='toself',
        name=car_type  # Assign unique names to each trace
    ))



# Define the slider marks and steps
slider_marks = {year: str(year) for year in df['AF'].unique()}
slider_steps = [{'args': [[year], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate'}], 'label': str(year), 'method': 'animate'} for year in df['AF'].unique()]
slider_steps.insert(0, {'args': [['all'], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate'}], 'label': 'All', 'method': 'animate'})

# Add the slider to the radar graph
radar_graph.update_layout(
    sliders=[{
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'center',
        'currentvalue': {
            'font': {'size': 14},
            'prefix': "Date de fabrication: ",
            'visible': True,
            'xanchor': 'center',
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 0, 't': 40},
        'len': 1,
        'x': 0.5,
        'y': 0.1,
        'steps': slider_steps,
    }],
    polar=dict(
        radialaxis=dict(
            visible=True,
            #range=[0, breakdown_counts['Nombre de pannes'].max() + 1]
        ),
    ),margin=dict(t=60, l=120, b=30, r=100),
    showlegend=True,
    title={
        
        'text': "Type de Panne par Type de Voiture",
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 30, 'color': 'white'},
    },
    height=590,
    width=600,
    paper_bgcolor='#3b6ea5',
    plot_bgcolor='#3b6ea5',
    font=dict(color='white'),
)

# Add the animation frames for each year in the data
frames = [go.Frame(data=[
    go.Scatterpolar(
        r=breakdown_counts[(breakdown_counts['TV'] == car_type) & (breakdown_counts['AF'] == year)]['Nombre de pannes'],
        theta=breakdown_counts[(breakdown_counts['TV'] == car_type) & (breakdown_counts['AF'] == year)]['TP'],
        fill='toself',
        name=car_type
    ) for car_type in df['TV'].unique()
], name=str(year)) for year in df['AF'].unique()]


radar_graph.update(frames=frames)

###### car modele /bar chart #######################
#Calculate the number of visits for each car model
car_model_visits = df['MV'].value_counts().reset_index()
car_model_visits.columns = ['MV', 'Nombre de Visites']
car_model_to_type = df.drop_duplicates(subset='MV')[['MV', 'TV']].set_index('MV')['TV'].to_dict()

# Create a dictionary to map car types to colors
car_type_colors = {
    'Haval': 'blue',
    'Foton': 'green',
    'Great Wall': 'orange',
}

# Create a dictionary to map car models to their abbreviations
car_model_abbreviations = {
    'HAVAL Jolion': 'Jolion',
    'HAVAL H6': 'H6',
    'SAUVANA':'SAUVANA',
    'TUNLAND':'TUNLAND',
    'WINGLE 5 DOUBLE CABINE':'W5 DC',
    'POER':'POER',
    'Gratour V5':'G V5',
    # Add more car model abbreviations as needed
}



# Map car models to car types
car_model_visits['TV'] = car_model_visits['MV'].map(car_model_to_type)

# Map car types to colors
car_model_visits['Color'] = car_model_visits['TV'].map(car_type_colors)

# Replace the car models in the 'Modèle de voiture' column with their abbreviations
car_model_visits['MV'] = car_model_visits['MV'].map(car_model_abbreviations)

# Create the bar chart using Plotly Express
bar_chart = px.bar(
    car_model_visits,
    x='MV',
    y='Nombre de Visites',
    color='TV',
    color_discrete_map=car_type_colors,
)

        
        
bar_chart.update_layout(
    title={
        'text': 'Nombre de Visites par Voiture',
        'x': 0.5,
        'y': 0.95,  # Adjust the y position of the title
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 30, 'color': 'white'},
    },
    xaxis_tickangle=-30, 
    xaxis_title='Models des voitures',
    xaxis_title_standoff=0,
    yaxis_title='Nombre de Visites',
    paper_bgcolor='#3b6ea5',
    plot_bgcolor='#3b6ea5',
    font=dict(color='white'),
    legend=dict(title='Type de Voiture'),
    height=290,
    margin=dict(t=50,b=10),
)

###### HEATMAP chart #####################################
# Group the data by 'TP' and calculate the average for each group
grouped_data = df.groupby('TP')[['PR', 'SF', 'TAS', 'C', 'S']].mean()

# Transpose the DataFrame to get the desired format for the heatmap
heatmap_data = grouped_data.T

# Define the x-axis (columns) and y-axis (rows) for the heatmap
x_values = heatmap_data.columns
y_values = heatmap_data.index

# Define the full names for each row (type de panne)
y_labels = {
    'PR': 'Problèmes résolus',
    'SF': 'Satisfaction des frais',
    'TAS': 'Achèvement du service',
    'C': 'Coût des services',
    'S': 'Satisfaction client',
}

# Create the heatmap plot using plotly.graph_objects
heatmap = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=x_values,
    y=y_values,
    colorscale='RdYlGn',  # You can choose any colorscale here
    zmin=7.8,  # Set the minimum value for the color scale
    zmax=8.6,  # Set the maximum value for the color scale
))


# Customize the layout
heatmap.update_layout(
    title={
        'text': 'satisfaction moyenne par Panne',
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 30, 'color': 'white'},
    },
    xaxis=dict(side='top',tickangle=20),  # Place x-axis labels on top
    yaxis=dict(side='left',tickvals=list(range(len(y_values))), ticktext=[y_labels[y] for y in y_values]),
    #xaxis_title='Type de Panne',
    yaxis_title='Metrics',
    height=290,
    width=600,
    paper_bgcolor='#3b6ea5',
    font=dict(color='white'),
    margin=dict(l=50, r=50, t=115, b=10),
)


# Define a custom color palette for the bar chart (using RGB values)
custom_colors_2 = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)']

# Aggregating to get the count of each 'TP'
aggregated_counts = df['TP'].value_counts().reset_index()
aggregated_counts.columns = ['TP', 'Nombre de pannes']

# Create the bar chart
bar_chart_fig = px.bar(
    aggregated_counts,
    x='TP',  # 'Type de panne' as x-axis
    y='Nombre de pannes',  # Count or frequency as y-axis
    color='TP',  # Use 'Type de panne' for coloring
    color_discrete_sequence=custom_colors_2,
    labels={'TP': 'Type de Panne', 'Nombre de pannes': 'Nombre de Pannes'},
)
'''
current_date = pd.Timestamp.now()
last_month = current_date - pd.DateOffset(months=1)
last_year = current_date - pd.DateOffset(years=1)
last_six_months = current_date - pd.DateOffset(months=6)

# Create a dictionary to store the filtered data for each button
button_data = {
    'LM': df[df['DE'] >= df['DE'].max() - pd.DateOffset(months=1)],
    'LY': df[df['DE'] >= df['DE'].max() - pd.DateOffset(years=1)],
    'L6M': df[df['DE'] >= df['DE'].max() - pd.DateOffset(months=6)],
    'ALL': df,
}


updatemenus=[
        {
            'buttons': [
             {
                    'args': [{'x': [button_data['LM']['TP']], 'y': [button_data['LM']['TP'].value_counts()], 'title': 'Last Month'}, {'type': 'bar'}],
                    'label': 'LM',
                    'method': 'update'
                },
                {
                    'args': [{'x': [button_data['LY']['TP']], 'y': [button_data['LY']['TP'].value_counts()], 'title': 'Last Year'}, {'type': 'bar'}],
                    'label': 'LY',
                    'method': 'update'
                },
                {
                    'args': [{'x': [button_data['L6M']['TP']], 'y': [button_data['L6M']['TP'].value_counts()], 'title': 'Last 6 Months'}, {'type': 'bar'}],
                    'label': 'L6M',
                    'method': 'update'
                },
                {
                    'args': [{'x': [aggregated_counts['TP']], 'y': [aggregated_counts['Nombre de pannes']], 'title': 'All'}, {'type': 'bar'}],
                    'label': 'ALL',
                    'method': 'update'
                }
        ],
            'type':'buttons',
            'direction': 'down',
            'pad': {'r': 10, 't': 10},
            'showactive': True,
            'x': 0.1,
            'xanchor': 'left',
            'y': 1.1,
            'yanchor': 'top',
            'bgcolor': 'rgba(255, 255, 255, 0.7)',  # Set background color for the buttons
        }
    ],
'''

# Update the layout for the bar chart
bar_chart_fig.update_layout(
    title={
        'text': 'Les pannes',
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 30, 'color': 'white'},
    },
    legend_title_text='Type de Panne',  # Set the legend title
    legend_orientation='v',  # Vertical legend
    legend_x=1.02,  # Position the legend to the right of the chart
    xaxis_title_text='',  # Remove x-axis label
    xaxis_showticklabels=False,  # Hide x-axis tick labels
    plot_bgcolor='#3b6ea5',  # Set the background color
    paper_bgcolor='#3b6ea5',  # Set the plot paper background color
    font=dict(color='white'),  # Set font color to white
    height=300,  # Set the height of the chart
    margin=dict(l=50, r=100, t=70, b=0),  # Add margins for better spacing
)

#####Courbe plot #################
# Create the figure
courbe_fig = go.Figure()

#avg_metrics = df.groupby('DE')[['PR', 'SF', 'TAS', 'C', 'S']].mean()
avg_metrics = df.groupby(pd.Grouper(key='DE', freq='M'))[['PR', 'SF', 'TAS', 'C', 'S']].mean()
# Add traces for each metric
metrics = ['PR', 'SF', 'TAS', 'C', 'S']
for metric in metrics:
    courbe_fig.add_trace(go.Scatter(x=avg_metrics.index, y=avg_metrics[metric], mode='lines', name=metric))
# Update the layout
courbe_fig.update_layout(
    title={
        'text': 'Evolution de satisfaction',
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 30, 'color': 'white'},
    },
    xaxis_title='Date d\'entretien',
    yaxis_title='Valeur',
    legend_title_text='Satisfaction',
    plot_bgcolor='#3b6ea5',  # Set the background color
    paper_bgcolor='#3b6ea5',  # Set the plot paper background color
    font=dict(color='white'),  # Set font color to white
    height=300,
    margin=dict(l=50, r=100, t=50, b=0),  # Add margins for better spacing
)

app.layout= dbc.Container(
    [
    dbc.Row(
            [
                dbc.Col(html.Img(src="https://foton.tn/images/dealer/1/image/ar13a9eb76997345b9910d86085e4e127d.jpg", style={'width':'200px','height':'150px','transform':'translateY(-50px)'})),
                dbc.Col(html.H1('My Dashboard', style={'font-size':'35px'}),),
            ],style={'background-color':'white','height':'50px','overflow':'hidden'},   
            ),
            
    dbc.Row(
            [
                dbc.Col(
                        # Total number of visits div
                        html.Div(id='total-visits-div', children=[
                            html.H3('Nb total des Visits',style=style1),
                            html.H2(id='total-visits-output',style=style2)
                        ]),className="mb-4 text-center",
                        style=style3
                        ),
                
                dbc.Col(
                        # Average visits per day div
                        html.Div(id='average-visits-div', children=[
                            html.H3('moyenne par jour',style=style1),
                            html.H2(id='average-visits-output',style=style2)
                        ]),className="mb-4 text-center",
                        style=style3
                        ),
                
                dbc.Col(
                        # Average satisfaction div
                        html.Div(id='average-satisfaction-div', children=[
                            html.H3('Satisfaction moyenne',style=style1),
                            html.H2(id='average-satisfaction-output',style=style2)
                        ]),className="mb-4 text-center",
                        style=style3
                        ),
                
                dbc.Col(
                        
                            dbc.Card(
                                    # Circular chart for type of visit div
                                    dcc.Graph(id='type-visite-chart',
                                            style={},
                                            config={'displayModeBar': False},
                                            
                                            ),className="mb-4 text-center",
                                    style=style3
                                    )
                        
                        ),
                
                dbc.Col(
                        dbc.Card(
                                    # Circular chart for garantie ou non div
                                    dcc.Graph(id='garantie-chart',
                                            style={},
                                            config={'displayModeBar': False},
                                            
                                            ),className="mb-4 text-center",
                                    style=style3
                                    )
                        
                        ),
                
                dbc.Col(
                        # Controls div
                        html.Div([
                            html.H3('Controls',style=style1),

                            dcc.Dropdown(
                                id='date-range-picker',
                                options=[
                                    {'label': 'Last Month', 'value': 'last_month'},
                                    {'label': 'Last 6 Months', 'value': 'last_6_months'},
                                    {'label': 'Last Year', 'value': 'last_year'},
                                    {'label': 'All', 'value': 'all'},
                                ],
                                value='all',
                                multi=False,
                                placeholder='Select Date Range',
                                style={'width': '100%', 'color': '#333', 'margin-bottom': '10px'}
                            ),
                            
                            dcc.Dropdown(
                                    id='car-type-dropdown',
                                    options=[
                                        {'label': 'Haval', 'value': 'Haval'},
                                        {'label': 'Foton', 'value': 'Foton'},
                                        {'label': 'Great Wall', 'value': 'Great Wall'},
                                        {'label': 'All', 'value': 'All'},
                                    ],
                                    value='All',
                                    style={'width': '100%', 'color': '#333', 'margin-bottom': '10px'}
                                )
                        ], className="p-4", style={'border': '1px solid #ddd', 'border-radius': '5px'})
                        ),
            ],
            align="center",
        ),
    dbc.Row([
        dbc.Col(
                        
                            dbc.Card(
                                    dcc.Graph(
                                                id='radar-chart',
                                                figure=radar_graph,
                                            ),
                                            className="mb-4 text-center",
                                            style=style5
                                     ),width=4
                        ),
        dbc.Col(html.Div([
            dbc.Row(
                [
                    dbc.Col(
                            dbc.Card(
                                        dcc.Graph(
                                                    id='bar_chart',
                                                    figure=bar_chart,
                                                ),
                                                className="mb-4 text-center",
                                                style=style4
                                        )
                            
                            ),
                    dbc.Col(
                            dbc.Card(
                                        dcc.Graph(
                                                    id='heat_chart',
                                                    figure=heatmap,
                                                ),
                                                className="mb-4 text-center",
                                                style=style4
                                        )
                            
                            ),
                ],
                align="center",
            ),
        dbc.Row(
            [
                
                dbc.Col(
                        dbc.Card(
                                    dcc.Graph(
                                                id='bar_chart_fig',
                                                figure=bar_chart_fig,
                                            ),
                                            className="mb-4 text-center",
                                            style=style6
                                    ),width=4
                        
                        ),
                dbc.Col(
                        dbc.Card(
                                    dcc.Graph(
                                                id='courbe_fig',
                                                figure=courbe_fig,
                                            ),
                                            className="mb-4 text-center",
                                            style=style6
                                    ),width=8
                        
                        ),
            ],
            align="center",
        ),    
            ])),
        
    ]),
    
    
    ], fluid=True,style={'backgroundColor': colors['background'],'height':'100vh','width':'100vw'}
    
)

# Callback to update the total number of visits, average visits, and average satisfaction divs, and the charts
@app.callback(
    Output('total-visits-output', 'children'),
    Output('average-visits-output', 'children'),
    Output('average-satisfaction-output', 'children'),
    Output('type-visite-chart', 'figure'),
    Output('garantie-chart', 'figure'),
    Input('date-range-picker', 'value'),
    Input('car-type-dropdown', 'value')
)
def update_data_and_charts(date_range, car_type):
    filtered_df = df.copy()
    if date_range == 'last_month':
        start_date = pd.to_datetime('today') - pd.DateOffset(months=1)
    elif date_range == 'last_6_months':
        start_date = pd.to_datetime('today') - pd.DateOffset(months=6)
    elif date_range == 'last_year':
        start_date = pd.to_datetime('today') - pd.DateOffset(years=1)
    else:
        start_date = df['DE'].min()

    filtered_df = filtered_df[filtered_df['DE'] >= start_date]
    if car_type != 'All':  # Apply car type filter only if not 'All'
        filtered_df = filtered_df[filtered_df['TV'] == car_type]

    total_visits = len(filtered_df)
    visits_per_day = total_visits / (pd.to_datetime('today') - start_date).days
    average_satisfaction = filtered_df['S'].mean()

    type_visite_chart = create_type_visite_chart(start_date,filtered_df)
    garantie_chart = create_garantie_chart(start_date,filtered_df)
    
     # Update the figure for the 'Type de Visite' chart
    type_visite_chart = create_type_visite_chart(start_date, filtered_df)
    type_visite_chart['layout']['height'] = 180
    type_visite_chart['layout']['paper_bgcolor'] = '#3b6ea5'
    type_visite_chart['layout']['plot_bgcolor'] = '#3b6ea5'
    type_visite_chart['layout']['font']['color'] = 'white'
    
    
    type_visite_chart.update_layout(
        title_x=0.5,  # Center the title horizontally
        title_y=0.9,  # Adjust the vertical position of the title
        legend=dict(orientation='h', yanchor='bottom', y=1.1),
        title={
            'text': 'Type de Visite',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 30, 'color': 'white'},
        },
        margin=dict(
            l=0,  # Left margin
            r=0,  # Right margin
            b=10,  # Bottom margin
            t=100,  # Top margin
        ),        
    )
    
    
    

    # Update the figure for the 'Garantie ou Non' chart
    garantie_chart = create_garantie_chart(start_date, filtered_df)
    garantie_chart['layout']['title'] = {
        'text': 'Garantie',
        'x': 0.5,
        'y': 1,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'size': 30 
        }
    }
    garantie_chart['layout']['height'] = 180
    garantie_chart['layout']['paper_bgcolor'] = '#3b6ea5'
    garantie_chart['layout']['plot_bgcolor'] = '#3b6ea5'
    garantie_chart['layout']['font']['color'] = 'white'
    
    garantie_chart.update_layout(
        title_x=0.5,  # Center the title horizontally
        title_y=0.9,  # Adjust the vertical position of the title
        #legend=dict(orientation='h', yanchor='bottom', y=1.1),
        margin=dict(
            l=0,  # Left margin
            r=0,  # Right margin
            b=10,  # Bottom margin
            t=50,  # Top margin
        ),        
    )

    return f'{total_visits}', f'{visits_per_day:.2f}', f'{average_satisfaction:.2f}', type_visite_chart, garantie_chart





if __name__ == '__main__':
    app.run_server(debug=True)




