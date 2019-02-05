import dash

from flask import Flask

server = Flask(__name__)
# server.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///crime_data.db'
server.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
server.config["SQLALCHEMY_ECHO"] = True


from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy(server)

app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

from dash_package.dash_dashboard import * #will run
