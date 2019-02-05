from dash_package.dash_queries import *
from dash.dependencies import Input, Output



app.layout = html.Div([
    html.H1('DO GOOGLE SEARCHES PREDICT CHANGES TO THE UNEMPLOYMENT RATE?'),
    html.Div([
        html.H2('Initial Time Series Analysis'),
       dcc.Tabs(id="tabs", children=[
            dcc.Tab(id='unemployment_ts', label='Traditional Time Series',
                children=[
                dcc.Graph(figure=
                {'data': initial_display_ue,
                'layout': {'title':'Initial Dataset'},
                })
                ]
            ),
            # # dcc.Tab(id='Felony', label='Felony Complaints',
            #     children=[
            #     dcc.Graph(figure=
            #     {'data': level_graph_creator_all("Felony")+level_graph_all_boroughs(boroughs,month_names,"Felony"),
            #     'layout': {'title':'Felonies'}})
            #     ]
            # ),
            # dcc.Tab(id='Misdemeanor', label='Misdemeanor Complaints',
            #     children=[
            #     dcc.Graph(figure=
            #     {'data': level_graph_creator_all("Misdemeanor")+level_graph_all_boroughs(boroughs,month_names,"Misdemeanor"),
            #     'layout': {'title':'Misdemeanors'}})
            #     ]
            # ),
            # dcc.Tab(id='Violation', label='Violation Complaints',
                # children=[
                # dcc.Graph(figure=
                # {'data': level_graph_creator_all("Violation")+level_graph_all_boroughs(boroughs,month_names,"Violation"),
                # 'layout': {'title':'Violations'}})
                # ]
            # ),
            ])
        ]),

    # html.H2('Crime Clusters by Primary Description'),
    # dcc.Dropdown(
    #     id='cluster-dropdown',
    #     options=drop_down_options,
    #     placeholder = "Select an Offense"
    # ),
    # html.Iframe(id='output-container',srcDoc = initial_display, width = '100%', height = '600')])
    ])
# @app.callback(
#     dash.dependencies.Output('output-container', 'srcDoc'),
#     [dash.dependencies.Input('cluster-dropdown', 'value')])
# def update_output(value):
#     if value == None:
#         value = 'initial_display'
#     srcDoc = open('dash_package/map_storage/"{}".html'.format(value), 'r').read()
#     return srcDoc
