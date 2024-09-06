from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import requests
import plotly.express as px
import pandas as pd
from flask import request

external_stylesheets = [dbc.themes.BOOTSTRAP, '/static/css/dashboard.css']

# Create Dash application
dash_app = Dash(__name__, requests_pathname_prefix='/dashboard/', external_stylesheets=external_stylesheets)

# Dash layout with separate containers for header, controls, and charts
dash_app.layout = html.Div([
    html.Div([
        html.A(
            dbc.Button("Back to homepage", color="success", className="btn-homepage"),
            href="http://127.0.0.1:8000/homepage",
            style={'margin-right': '20px'}
        ),
        html.H1("Score performance visualization", className="dashboard-title")
    ], className="header-container dashboard-container"),

    html.Hr(className="divider"),

    html.Div([
        html.Div([
            html.Label('Choose the type of the throw:', className="label"),
            dcc.RadioItems(
                id='video-type-1',
                options=[
                    {'label': 'Free Throws', 'value': 'free-throws'},
                    {'label': '3-Point Throws', 'value': '3-point-throws'},
                    {'label': 'All', 'value': 'all'}
                ],
                value='free-throws',  # Default value
                labelStyle={'display': 'inline-block', 'margin-right': '20px'},
            ),
        ], className="input-container"),

        # html.Div(id='total-accuracy', className="total-accuracy"),

        # Toggle for choosing the filtering method
        html.Div([
            html.Label('Choose the filtering method:', className="label"),
            dcc.RadioItems(
                id='filter-method',
                options=[
                    {'label': 'Date Range', 'value': 'date-range'},
                    {'label': 'Recent Videos', 'value': 'video-slider'}
                ],
                value='date-range',  # Default value
                labelStyle={'display': 'inline-block', 'margin-right': '20px'},
            ),
        ], className="input-container"),

        html.Div([
            html.Div(id='date-label', className="label"),
            dcc.DatePickerRange(
                id='date-range',
                display_format='YYYY-MM-DD',
                style={'margin-bottom': '0'}
            ),
        ], className="input-container", id='date-range-container'),

        # Video slider
        html.Div([
            html.Label('Choose the number of recent videos:', className="label"),
            dcc.Slider(
                id='video-slider',
                min=1,
                max=1,
                value=1,
                marks=None,
                step=1
            ),
        ], className="input-container", id='video-slider-container'),
    ], className="controls-container dashboard-container"),

    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # half minute in milliseconds
        n_intervals=0
    ),

    html.Hr(className="divider"),

    html.Div([
        dcc.Graph(id='bar-chart')
    ], className="graph-container dashboard-container"),

    html.Hr(className="divider"),

    html.Div([
        dcc.Graph(id='line-chart')
    ], className="graph-container dashboard-container"),

], className="main-container")

# Callback to toggle between date picker and slider
@dash_app.callback(
    Output('date-range-container', 'style'),
    Output('video-slider-container', 'style'),
    Input('filter-method', 'value')
)
def toggle_filtering_method(filter_method):
    if filter_method == 'date-range':
        return {'display': 'block'}, {'display': 'none'}
    elif filter_method == 'video-slider':
        return {'display': 'none'}, {'display': 'block'}
    return {'display': 'block'}, {'display': 'none'}


# Callback for updating date-label
@dash_app.callback(
    Output('date-label', 'children'),
    Input('interval-component', 'n_intervals'),
)
def update_date_label(n_intervals):
    return f"Choose the timeline:"


# Callback to update date range
@dash_app.callback(
    Output('date-range', 'start_date'),
    Output('date-range', 'end_date'),
    Input('video-type-1', 'value')
)
def update_date_range(video_type):

    # Get cookie value from request
    cookie_value = request.cookies.get('basketball')

    headers = {
        'Cookie': f'basketball={cookie_value}'
    }

    response = requests.get(f"http://127.0.0.1:8000/videos/statistics/{video_type}", headers=headers)
    data = response.json()

    # Create DataFrame
    df = pd.DataFrame(data)

    if df.empty:
        return None, None

    # Convert 'date_upload' to datetime
    df['date_upload'] = pd.to_datetime(df['date_upload'])

    # Extract the date part
    df['date'] = df['date_upload'].dt.date

    # Get min and max dates
    start_date = df['date'].min()
    end_date = df['date'].max()
    
    return start_date, end_date

# Callback to update video slider
@dash_app.callback(
    Output('video-slider', 'min'),
    Output('video-slider', 'max'),
    Output('video-slider', 'value'),
    Output('video-slider', 'marks'),
    Input('video-type-1', 'value')
)
def update_video_slider(video_type):
    # Get cookie value from request
    cookie_value = request.cookies.get('basketball')

    headers = {
        'Cookie': f'basketball={cookie_value}'
    }

    response = requests.get(f"http://127.0.0.1:8000/videos/statistics/{video_type}", headers=headers)
    data = response.json()

    # Create DataFrame
    df = pd.DataFrame(data)
    
    if df.empty:
        # Return default empty slider if the dataframe is empty
        return 1, 1, 1, None

    # Get number of videos
    video_num = df.shape[0]

    # Define the marks for the slider dynamically
    marks = {i: str(i) for i in range(1, video_num + 1)}

    # Return the slider min, max, value, and marks
    return 1, video_num, video_num, marks


# Callbacks to update graphs based on selected filter method
@dash_app.callback(
    Output('bar-chart', 'figure'),
    Output('line-chart', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('video-type-1', 'value'),
    Input('filter-method', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('video-slider', 'value')
)
def update_charts(n_intervals, video_type, filter_method, start_date, end_date, slider_value):
    # Get cookie value from request
    cookie_value = request.cookies.get('basketball')

    headers = {
        'Cookie': f'basketball={cookie_value}'
    }

    response = requests.get(f"http://127.0.0.1:8000/videos/statistics/{video_type}", headers=headers)
    data = response.json()

    # Create DataFrame
    df = pd.DataFrame(data)

    if df.empty:
        return {
            'data': [],
            'layout': {'title': 'No data available'}
        }, {
            'data': [],
            'layout': {'title': 'No data available'}
        }

    if filter_method == 'date-range':
        # Convert 'date_upload' to datetime
        df['date_upload'] = pd.to_datetime(df['date_upload'])
        df['date'] = df['date_upload'].dt.date

        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()

        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        filtered_df = df.loc[mask]

    else:
        filtered_df = df.iloc[-slider_value:]

    # If the filtered DataFrame is empty, return empty graphs
    if filtered_df.empty:
        return {
            'data': [],
            'layout': {'title': 'No data available'}
        }, {
            'data': [],
            'layout': {'title': 'No data available'}
        }

    # Bar Chart
    sum_goals = sum(filtered_df['number_goals'])
    sum_misses = sum(filtered_df['number_misses'])
    total = sum_goals + sum_misses

    percentage_goals = (sum_goals / total) * 100 if total > 0 else 0
    percentage_misses = (sum_misses / total) * 100 if total > 0 else 0

    bar_fig = {
        'data': [
            {'x': ['Goals'], 'y': [sum_goals], 'type': 'bar', 'name': 'Goals', 'text': [f'<b>{percentage_goals:.2f}%</b>'], 'textposition': 'auto', 'texttemplate': '%{text}'},
            {'x': ['Misses'], 'y': [sum_misses], 'type': 'bar', 'name': 'Misses', 'text': [f'<b>{percentage_misses:.2f}%</b>'], 'textposition': 'auto', 'texttemplate': '%{text}'}
        ],
        'layout': {'title': f'Total Goals and Misses'}
    }
    
    filtered_df.reset_index(inplace=True)

    # Line Chart
    if filter_method == 'date-range':
        pivot_table = filtered_df.pivot_table(index='date', values='accuracy', aggfunc='mean').reset_index()
        line_fig = px.line(pivot_table, x='date', y='accuracy', title='Accuracy Over Time', markers=True)
        line_fig.update_layout(title={'x': 0.5})
    else:
        filtered_df['date_upload'] = pd.to_datetime(filtered_df['date_upload'])
        filtered_df['date'] = filtered_df['date_upload'].dt.date
        # print(filtered_df['date_upload'].dt.time)
        filtered_df['time'] = filtered_df['date_upload'].apply(lambda dt: dt.strftime("%H:%M:%S") + " (UTC)")

        line_fig = px.line(
            filtered_df,
            x=filtered_df.index+1,
            y='accuracy',
            title='Accuracy Over Time',
            markers=True,
            hover_data={
                'accuracy': True, 
                'date': True, 
                'time': True
            },
            labels={
                'accuracy': 'Accuracy',
                'date': 'Date',
                'time': 'Time'
            })  
        
        line_fig.update_layout(title={'x': 0.5})
        
        # Update hovertemplate to exclude x value
        line_fig.update_traces(
            hovertemplate='<br>'.join([
                'Accuracy=%{y}',
                'Date=%{customdata[0]}',
                'Time=%{customdata[1]}'
            ]))  

        line_fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(1, len(filtered_df) + 1)),  # Only integer tick values
            title_text='Video Number')
        
        line_fig.update_yaxes(range=[0, 105])

    return bar_fig, line_fig

# Function to return the Dash app
def get_dash_app():
    return dash_app
