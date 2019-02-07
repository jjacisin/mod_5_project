from dash_package.dash_queries import *
from dash.dependencies import Input, Output
import dash_table



app.layout = html.Div([
    html.H1('CAN GOOGLE SEARCHES PREDICT CHANGES TO THE UNEMPLOYMENT RATE?'),
    html.Div([
        html.H2('The Unemployment Rate (UER)'),
       dcc.Tabs(id="tabs", children=[
            dcc.Tab(id='uer_1', label='All UER',
                children=[
                dcc.Graph(figure=
                {'data': [ue_initial_display],
                'layout': {'title':'Raw UER'},
                })
                ]
            ),
            dcc.Tab(id='uer_2', label='In Scope UER',
                children=[
                dcc.Graph(figure=
                {'data': [ue_in_scope_display],
                'layout': {'title':'UER Since 2005'},
                })
                ]
            ),
            dcc.Tab(id='uer_3', label='Standardized UER',
                children=[
                dcc.Graph(figure=
                {'data':[ue_in_scope_display,ue_rolmean,ue_standardized_data,ue_rolstd],
                'layout': {'title':'UER - 12M Rolling Mean'},
                }),
                dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in df_ue_adfuller.columns],
                data=[df_ue_adfuller.to_dict("data")['Results']],
                )]
            ),
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
