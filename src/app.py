import os
import datetime
import json
import logging

import pandas as pd
import awswrangler as wr

from flask import render_template, request, jsonify, Flask, redirect, url_for, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from logging.handlers import RotatingFileHandler
from sqlalchemy import inspect, create_engine, func
from sqlalchemy.sql import text, and_, or_
from wtforms import SubmitField, SelectField, SelectMultipleField, StringField, HiddenField
from wtforms.validators import DataRequired, Email

from src.openai_funcs import topic_summary, comparator_summary
from src.supporter_funcs import *

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("SQLALCHEMY_DATABASE_URI")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config['WTF_CSRF_ENABLED'] = False

db = SQLAlchemy(app)
app.app_context().push()

#establishing logger and assigning to flask app handler
logging.basicConfig(level=logging.INFO)
formatter = logging.Formatter(
    "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s")
handler = RotatingFileHandler('application.log', maxBytes=10000000, backupCount=5)  # 10MB file
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
app.logger.addHandler(handler)

@app.errorhandler(500)
def internal_server_error(error):
    app.logger.error('Server Error: %s', error)
    return "Internal server error", 500

@app.errorhandler(Exception)
def unhandled_exception(e):
    app.logger.error('Unhandled Exception: %s', e)
    return "Internal server error", 500

@app.before_request
def log_request_info():
    app.logger.info('Headers: %s', request.headers)
    app.logger.info('Body: %s', request.get_data())

class Params(db.Model):
    __tablename__ = "best_parameters"
    id = db.Column(db.String(100), primary_key=True)
    min_cluster_size = db.Column(db.Integer)
    min_samples = db.Column(db.Integer)
    cluster_selection_method = db.Column(db.String(25))
    cluster_selection_epsilon = db.Column(db.Float)
    metric = db.Column(db.String(25))
    score = db.Column(db.Float)

def ResultsTableName(file_name):        
    class Results(db.Model):
        __tablename__ = file_name.replace(".parquet", "")
        __table_args__ = {'extend_existing': True, "schema":"clustering_data"}#this is required to allow the table to be overwritten if it already exists, otherwise errors are thrown on dashboard page refresh
        doi = db.Column(db.String(100), db.ForeignKey("dw_article_exten.doi"), primary_key=True)
        article_title = db.Column(db.String(500))
        full_source_title = db.Column(db.String(500))
        citations = db.Column(db.Integer)
        year_published = db.Column(db.Integer)
        art_oa_status = db.Column(db.String(25))
        publisher_group = db.Column(db.String(100))
        coord_x = db.Column(db.Float)
        coord_y = db.Column(db.Float)
        prid_country = db.Column(db.String(250))
        prid_region = db.Column(db.String(250))
        cluster_label = db.Column(db.Integer)
        exemplar = db.Column(db.Boolean)
        gpt_label = db.Column(db.String(100))
    return Results

def AuthorsTablename(file_name):        
    class Authors(db.Model):
        __tablename__ = f"{file_name.replace('.parquet', '')}"
        __table_args__ = {'extend_existing': True, "schema":"authors"}
        index = db.Column(db.Integer, primary_key=True)
        gpt_label = db.Column(db.String(100))
        author_full_name = db.Column(db.String(500))
        research_org = db.Column(db.String(500))
        prid_country = db.Column(db.String(100))
        prid_region = db.Column(db.String(100))
        sum_published = db.Column(db.Integer)
        sum_citations = db.Column(db.Integer)
        full_source_title_list = db.Column(db.String(500))
        publisher_group_list = db.Column(db.String(500))
        avg_cites_per_article = db.Column(db.Integer)
    return Authors

def customResultsTableName(file_name):        
    class customResults(db.Model):
        __tablename__ = file_name.replace(".parquet", "")
        __table_args__ = {'extend_existing': True, "schema":"custom_clustering_data"}#this is required to allow the table to be overwritten if it already exists, otherwise errors are thrown on dashboard page refresh
        doi = db.Column(db.String(100), db.ForeignKey("dw_article_exten.doi"), primary_key=True)
        article_title = db.Column(db.String(500))
        full_source_title = db.Column(db.String(500))
        citations = db.Column(db.Integer)
        year_published = db.Column(db.Integer)
        art_oa_status = db.Column(db.String(25))
        publisher_group = db.Column(db.String(100))
        coord_x = db.Column(db.Float)
        coord_y = db.Column(db.Float)
        prid_country = db.Column(db.String(250))
        prid_region = db.Column(db.String(250))
        cluster_label = db.Column(db.Integer)
        exemplar = db.Column(db.Boolean)
        gpt_label = db.Column(db.String(100))
    return customResults

def customAuthorsTablename(file_name):        
    class customAuthors(db.Model):
        __tablename__ = f"{file_name.replace('.parquet', '')}"
        __table_args__ = {'extend_existing': True, "schema":"custom_authors"}
        index = db.Column(db.Integer, primary_key=True)
        gpt_label = db.Column(db.String(100))
        author_full_name = db.Column(db.String(500))
        research_org = db.Column(db.String(500))
        prid_country = db.Column(db.String(100))
        prid_region = db.Column(db.String(100))
        sum_published = db.Column(db.Integer)
        sum_citations = db.Column(db.Integer)
        full_source_title_list = db.Column(db.String(500))
        publisher_group_list = db.Column(db.String(500))
        avg_cites_per_article = db.Column(db.Integer)
    return customAuthors

def create_form_data():
    with app.app_context():
        df1 = wr.s3.read_csv("s3://rootbucket/topic_clustering/form_data/title_pub_subject.csv")
        df2 = wr.s3.read_csv("s3://rootbucket/topic_clustering/form_data/countries.csv")

        subjects = sorted(df1.subject_cat_desc.unique().tolist())
        countries = df2.prid_country.unique().tolist()
        countries.insert(0, '')
        publishers = sorted(pd.DataFrame(df1.groupby("publisher_group").size()).reset_index().sort_values(0, ascending=False).head(15).publisher_group.tolist())
        publishers = [x.title() for x in publishers]
        publishers.insert(0, '')
        initalised_comparitors = sorted(df1[df1.subject_cat_desc == subjects[0]].full_source_title.unique().tolist())
        initalised_comparitors = [x.title() for x in initalised_comparitors]
        initalised_comparitors.insert(0, '')
        return subjects, countries, publishers, initalised_comparitors

class QuestionForm(FlaskForm):
    subjects, countries, publishers, initalised_comparitors = create_form_data()
    subject = SelectField("Select the subject category of interest", choices = subjects, validators=[DataRequired()])
    pub_years = SelectMultipleField(
        "Select the article publication years of interest (individual years or JCR pairs)",
        choices=list(range(2018, datetime.datetime.now().year)),
        coerce=int,
        validators=[DataRequired()]
    )
    country = SelectField("Select a country of interest or leave blank include all countries", choices = countries)
    region = SelectField(
        "Select a region of interest, or leave blank", choices=["","Africa & Middle East","Asia","Australasia","Central & South America","Europe","North America", "TA7"]
    )
    publisher = SelectField(
        "Select either a publisher of interest, or leave blank",
        choices= publishers,
    )
    comparitor = SelectField(
        "Select a journal to compare against the selected subject category",
        choices = [x.upper() for x in initalised_comparitors],
        coerce=str
    )
    submit = SubmitField("Submit")

class EmailForm(FlaskForm):
    user_email = StringField("Enter your email here:", validators=[Email()])
    file_name = HiddenField()
    email_submit = SubmitField("Submit")

def get_params(file_name):
    with app.app_context():
        def object_as_dict(obj):
            return {
            c.key: getattr(obj, c.key)
            for c in inspect(obj).mapper.column_attrs
            }   
    return object_as_dict(db.session.query(Params).filter_by(id=file_name).first_or_404())

def database_write(df, table, schema):
    engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"], echo=True)
    try:
        df.to_sql(table, engine, schema = schema, if_exists = 'fail', index=False)
        return True
    except:
        return False

def init_db_and_get_labels_params(file_name, custom=False, custom_size=None):
    engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"], echo=True)
    if custom == False:
        params = get_params(file_name)
        cluster_labels = pd.read_sql_table(file_name.replace(".parquet", ""), engine, schema="clustering_data")
    elif custom == True:
        params = get_params(file_name)
        cluster_labels = pd.read_sql_table(f"[{custom_size}]"+file_name.replace(".parquet", ""), engine, schema="custom_clustering_data")
    return cluster_labels, params
    


def get_tables(file_name, custom=False):
    if custom == False:
        exemplarsTable = ResultsTableName(file_name)
        authorsTable = AuthorsTablename(file_name)
    elif custom == True:
        exemplarsTable = customResultsTableName(file_name)
        authorsTable = customAuthorsTablename(file_name)
    return exemplarsTable, authorsTable

@app.route("/", methods=["GET", "POST"])
def home():
    form = QuestionForm()
    email_form = EmailForm()
    show_modal = False  #honestly at this stage i can't even remember if the ajax call actually requires this parameter but I dont want to unpick this to find out. Come back to this later.

    if form.validate_on_submit():

        subject = form.subject.data
        pub_years = [int(x) for x in form.pub_years.data]
        country = form.country.data
        region = form.region.data
        publisher = form.publisher.data
        journal = form.comparitor.data

        file_name = gen_file_name(subject, pub_years)
        app.logger.info('File name created')
        search = search_s3(file_name)
        app.logger.info('Search complete')

        if search == True:
            if region:
                app.logger.info(f'Loading comparitor dashboard using {file_name}')
                return redirect(url_for("comparator_dashboard", file_name=file_name, comparator_type="region", comparator=region))
            elif journal:
                app.logger.info(f'Loading comparitor dashboard using {file_name}')
                return redirect(url_for("comparator_dashboard", file_name=file_name, comparator_type="journal", comparator=journal))
            elif country:
                app.logger.info(f'Loading comparitor dashboard using {file_name}')
                return redirect(url_for("comparator_dashboard", file_name=file_name, comparator_type="country", comparator=country))
            elif publisher:
                app.logger.info(f'Loading comparitor dashboard using {file_name}')
                return redirect(url_for("comparator_dashboard", file_name=file_name, comparator_type="publisher", comparator=publisher))
            else:
                app.logger.info(f'Loading comparitor dashboard using {file_name}')
                return redirect(url_for("dashboard", file_name=file_name))
            
    else:
        print(f"FORM ERRORS: {form.errors}")
        app.logger.error(f'Form validation failed with errors: {form.errors}')
        
    return render_template("form.html", form=form, email_form=email_form, show_modal=show_modal)

@app.route("/email_submit", methods=["POST"])
def email_submit():
    #submitted like this to avoid passing user emails as raw data in url's, felt safer, but what do I know.
    file_name = request.form.get("file_name")
    user_email = request.form.get("user_email")
    
    write_stub_s3(file_name, user_email)
    
    return jsonify({"message": "Data processed successfully."})

@app.route("/check_s3/<file_name>", methods=["GET"])
def check_s3(file_name):
    try:
        search = search_s3(file_name)
        if search == False:
            write_stub_s3(file_name, "None")
        return jsonify({"exists": search}), 200
    except:
        print(f"Error: {e}")
        traceback.print_exc()

@app.route("/comparitors/<subject>", methods=["GET"])
def get_comparitors(subject):
    """
    Route used to get the comparitors for the selected subject category - fetched by the main page to enable dynamic form choices
    """
    with app.app_context():
        df = pd.read_csv("s3://rootbucket/topic_clustering/form_data/title_pub_subject.csv")
        comparitors = sorted(df[df.subject_cat_desc == subject].full_source_title.unique().tolist())
        comparitors.insert(0, '')
        
        comparitorList = []

        for index, comparitor in enumerate(comparitors):
            comparitorObj = {}
            comparitorObj["id"] = index + 1
            comparitorObj["full_source_title"] = comparitor.upper()
            
            exists = False # this for/if loop is a messy way to remove duplicates from the list, but it works /shrug
            for existing_comparitor in comparitorList:
                if existing_comparitor["full_source_title"] == comparitorObj["full_source_title"]:
                    exists = True
                    break

            if not exists:
                comparitorList.append(comparitorObj)

        return jsonify({"comparitors":comparitorList}), 200

@app.route('/dashboard/<file_name>')
def dashboard(file_name):
    cluster_labels, params = init_db_and_get_labels_params(file_name)
    summary = topic_summary(cluster_labels)
    exemplarsTable, authorsTable = get_tables(file_name)
    exemplars = db.session.query(exemplarsTable).filter_by(exemplar=True).order_by(text("cluster_label")).all()
    authors = db.session.query(authorsTable).order_by(text("gpt_label, avg_cites_per_article desc")).all()
    return render_template("results.html", file_name=file_name, params=params, exemplars=exemplars, summary=summary, authors=authors, clusters=sorted([str(x) for x in cluster_labels.gpt_label.unique().tolist()]))
        
@app.route('/comparator_dashboard/<file_name>/<comparator_type>/<comparator>')
def comparator_dashboard(file_name, comparator_type, comparator):
    if comparator_type == "journal":
        return journal_logic(file_name, comparator, comparator_type)
    elif comparator_type == "publisher":
        return publisher_logic(file_name, comparator, comparator_type)
    elif comparator_type == "region":
        return region_logic(file_name, comparator, comparator_type)
    elif comparator_type == "country":
        return country_logic(file_name, comparator, comparator_type)
    else:
        return "Invalid comparator type", 400

def journal_logic(file_name, journal, comparator_type):
    cluster_labels, params = init_db_and_get_labels_params(file_name)
    summary = topic_summary(cluster_labels)
    exemplarsTable, authorsTable = get_tables(file_name)

    cluster_labels_journal = cluster_labels[cluster_labels["full_source_title"].apply(lambda x: journal in x)]
    comp_summary = comparator_summary(cluster_labels, cluster_labels_journal)

    exemplars = db.session.query(exemplarsTable).filter(and_(exemplarsTable.full_source_title == journal, exemplarsTable.exemplar == True)).order_by(text("cluster_label")).all()
    authors = db.session.query(authorsTable).filter(authorsTable.full_source_title_list.contains("'"+str(journal)+"'")).order_by(text("gpt_label, avg_cites_per_article desc")).all()
    return render_template("results_comparator.html", file_name=file_name, params=params, exemplars=exemplars, summary=summary, authors=authors, clusters=sorted([str(x) for x in cluster_labels.gpt_label.unique().tolist()]), comparator = journal, comp_summary = comp_summary, comparator_type = comparator_type)

def publisher_logic(file_name, publisher, comparator_type):
    cluster_labels, params = init_db_and_get_labels_params(file_name)
    summary = topic_summary(cluster_labels)
    exemplarsTable, authorsTable = get_tables(file_name)

    cluster_labels_publisher = cluster_labels[cluster_labels["publisher_group"]==publisher.upper()]
    comp_summary = comparator_summary(cluster_labels, cluster_labels_publisher)

    exemplars = db.session.query(exemplarsTable).filter(and_(exemplarsTable.publisher_group == publisher.upper(), exemplarsTable.exemplar == True)).order_by(text("cluster_label")).all()
    authors = db.session.query(authorsTable).filter(authorsTable.publisher_group_list.contains("'"+str(publisher.upper())+"'")).order_by(text("gpt_label, avg_cites_per_article desc")).all()
    return render_template("results_comparator.html", file_name=file_name, params=params, exemplars=exemplars, summary=summary, authors=authors, clusters=sorted([str(x) for x in cluster_labels.gpt_label.unique().tolist()]), comparator = publisher, comp_summary = comp_summary, comparator_type = comparator_type)

def region_logic(file_name, region, comparator_type):
    cluster_labels, params = init_db_and_get_labels_params(file_name)
    summary = topic_summary(cluster_labels)
    exemplarsTable, authorsTable = get_tables(file_name)

    ta7 = ["United Kingdom","Germany","Australia","New Zealand","Canada","France","Italy","Spain"]
    if region != "TA7":
            cluster_labels_region = cluster_labels[cluster_labels["prid_region"].apply(lambda x: region in x)]
    elif region == "TA7":
        cluster_labels_region = cluster_labels[cluster_labels["prid_country"].apply(lambda x: any(country in x for country in ta7))]
    comp_summary = comparator_summary(cluster_labels, cluster_labels_region)

    if region != "TA7":
            exemplars = db.session.query(exemplarsTable).filter(exemplarsTable.prid_region.contains(str(region))).filter_by(exemplar=True).order_by(text("cluster_label")).all()
            authors = db.session.query(authorsTable).filter_by(prid_region=region).order_by(text("gpt_label, avg_cites_per_article desc")).all()
    elif region == "TA7":
        exemplars = db.session.query(exemplarsTable).filter(exemplarsTable.exemplar == True).filter(or_(*[exemplarsTable.prid_country.contains(country) for country in ta7])).order_by(text("cluster_label")).all()
        authors = db.session.query(authorsTable).filter(or_(*[authorsTable.prid_country.contains(country) for country in ta7])).order_by(text("gpt_label, avg_cites_per_article desc")).all()
    return render_template("results_comparator.html", file_name=file_name, params=params, exemplars=exemplars, summary=summary, authors=authors, clusters=sorted([str(x) for x in cluster_labels.gpt_label.unique().tolist()]), comparator = region, comp_summary = comp_summary, comparator_type = comparator_type)

def country_logic(file_name, country, comparator_type):
    cluster_labels, params = init_db_and_get_labels_params(file_name)
    summary = topic_summary(cluster_labels)
    exemplarsTable, authorsTable = get_tables(file_name)

    cluster_labels_country = cluster_labels[cluster_labels["prid_country"].apply(lambda x: country in x)]
    comp_summary = comparator_summary(cluster_labels, cluster_labels_country)

    exemplars = db.session.query(exemplarsTable).filter(exemplarsTable.prid_country.contains(str(country))).filter_by(exemplar=True).order_by(text("cluster_label")).all()
    authors = db.session.query(authorsTable).filter_by(prid_country=country).order_by(text("gpt_label, avg_cites_per_article desc")).all()
    return render_template("results_comparator.html", file_name=file_name, params=params, exemplars=exemplars, summary=summary, authors=authors, clusters=sorted([str(x) for x in cluster_labels.gpt_label.unique().tolist()]), comparator = country, comp_summary = comp_summary, comparator_type = comparator_type)

@app.route('/get_data/<file_name>/<comparator_type>/<comparator>/<custom>/<custom_size>', methods=['GET'])
def get_data(file_name, comparator_type=None, comparator=None, custom=False, custom_size=None):
    """
    Used to support tooltip for the d3 visualisations on the dashboard page
    """
    custom_bool = custom.lower() == 'true'

    engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"], echo=True)
    table_prefix = "" if not custom_bool else f"[{custom_size}]"
    cluster_labels = pd.read_sql_table(f"{table_prefix}{file_name.replace('.parquet', '')}", 
                                       engine, 
                                       schema="custom_clustering_data" if custom_bool else "clustering_data")
    
    cluster_labels["gpt_label"] = cluster_labels["gpt_label"].fillna("Unclustered")

    # If comparator_type and comparator are provided, use them for filtering
    if comparator_type and comparator:
        if comparator_type == "region":
            ta7 = ["United Kingdom", "Germany", "Australia", "New Zealand", "Canada", "France", "Italy", "Spain"]
            if comparator != "TA7":
                cluster_labels = cluster_labels[cluster_labels["prid_region"].apply(lambda x: comparator in x)]
            else:
                cluster_labels = cluster_labels[cluster_labels["prid_country"].apply(lambda x: any(country in x for country in ta7))]

        elif comparator_type == "journal":
            cluster_labels = cluster_labels[cluster_labels["full_source_title"] == comparator]

        elif comparator_type == "publisher":
            cluster_labels = cluster_labels[cluster_labels["publisher_group"] == comparator.upper()]

        elif comparator_type == "country":
            cluster_labels = cluster_labels[cluster_labels["prid_country"].apply(lambda x: comparator in x)]

    labelsJSON = cluster_labels.to_json(orient="records")
    return jsonify(json.loads(labelsJSON)), 200

@app.route('/download_all/<file_name>/<custom>/<custom_size>', methods=['GET'])
def download_all(file_name, custom=False, custom_size=None):
    custom_bool = custom.lower() == 'true'
    table_prefix = "" if not custom_bool else f"[{custom_size}]"
    engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"], echo=True)
    cluster_labels = pd.read_sql_table(f"{table_prefix}{file_name.replace('.parquet', '')}", 
                                       engine, 
                                       schema="custom_clustering_data" if custom_bool else "clustering_data")
    
    cluster_labels["gpt_label"] = cluster_labels["gpt_label"].fillna("Unclustered")

    csv_data = cluster_labels.to_csv(index=False)
    response = make_response(csv_data)
    response.headers['Content-Disposition'] = f'attachment; filename={file_name.replace(".parquet", "")}_dataset.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response

@app.route('/download_exemplars/<file_name>/<comparator_type>/<comparator>/<custom>/<custom_size>', methods=['GET'])
def download_exemplars(file_name, comparator_type=None, comparator=None, custom=False, custom_size=None):
    ta7 = ["United Kingdom", "Germany", "Australia", "New Zealand", "Canada", "France", "Italy", "Spain"]
    engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"], echo=True)
    
    custom_bool = custom.lower() == 'true'
    table_prefix = "" if not custom_bool else f"[{custom_size}]"
    
    if not custom_bool:
        exemplarsTable = pd.read_sql_table(file_name.replace(".parquet", ""), engine, schema="clustering_data")
    else:
        exemplarsTable = pd.read_sql_table(table_prefix+file_name.replace(".parquet", ""), engine, schema="custom_clustering_data")
    
    if comparator_type:
        if comparator_type == "region":
            if comparator != "TA7":
                exemplarsTable = exemplarsTable[exemplarsTable["prid_region"].apply(lambda x: comparator in x)]
            else:
                exemplarsTable = exemplarsTable[exemplarsTable["prid_country"].apply(lambda x: any(country in x for country in ta7))]

        elif comparator_type == "journal":
            exemplarsTable = exemplarsTable[exemplarsTable["full_source_title"] == comparator]

        elif comparator_type == "publisher":
            exemplarsTable = exemplarsTable[exemplarsTable["publisher_group"] == comparator.upper()]

        elif comparator_type == "country":
            exemplarsTable = exemplarsTable[exemplarsTable["prid_country"].apply(lambda x: comparator in x)]

    exemplarsTable = exemplarsTable[["doi", "article_title", "full_source_title", "citations", "year_published", "art_oa_status", "publisher_group", "gpt_label"]][exemplarsTable.exemplar == True]

    csv_data = exemplarsTable.to_csv(index=False)
    response = make_response(csv_data)
    if comparator_type != "none" and comparator != "none":
        response.headers['Content-Disposition'] = f'attachment; filename={file_name.replace(".parquet", "")}({comparator}).csv'
    else:
        response.headers['Content-Disposition'] = f'attachment; filename={file_name.replace(".parquet", "")}.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response
    
@app.route('/download_authors/<file_name>/<comparator_type>/<comparator>/<custom>/<custom_size>', methods=['GET'])
def download_authors(file_name, comparator_type=None, comparator=None, custom=False, custom_size=None):
    ta7 = ["United Kingdom", "Germany", "Australia", "New Zealand", "Canada", "France", "Italy", "Spain"]
    engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"], echo=True)

    custom_bool = custom.lower() == 'true'
    table_prefix = "" if not custom_bool else f"[{custom_size}]"

    if not custom_bool:
        authorsTable = pd.read_sql_table(file_name.replace(".parquet", ""), engine, schema="authors")
    else:
        authorsTable = pd.read_sql_table(table_prefix+file_name.replace(".parquet", ""), engine, schema="custom_authors")

    authorsTable = authorsTable.sort_values(by=["gpt_label", "avg_cites_per_article"], ascending=[True, False]).drop(columns=["index"])
    
    if comparator_type:
        if comparator_type == "region":
            if comparator != "TA7":
                authorsTable = authorsTable[authorsTable["prid_region"]==comparator]
            else:
                authorsTable = authorsTable[authorsTable["prid_country"].apply(lambda x: any(country in x for country in ta7))]

        elif comparator_type == "journal":
            authorsTable = authorsTable[authorsTable["full_source_title_list"].str.contains(f"'{comparator}'")]

        elif comparator_type == "publisher":
            authorsTable = authorsTable[authorsTable["publisher_group_list"].apply(lambda x: comparator in x)]

        elif comparator_type == "country":
            authorsTable = authorsTable[authorsTable["prid_country"]==comparator]

    authorsTable = authorsTable.drop(columns=["full_source_title_list", "publisher_group_list"], axis=1)
    csv_data = authorsTable.to_csv(index=False)
    response = make_response(csv_data)
    if comparator != "none":
        response.headers['Content-Disposition'] = f'attachment; filename={file_name.replace(".parquet", "")}_authors({comparator}).csv'
    else:
        response.headers['Content-Disposition'] = f'attachment; filename={file_name.replace(".parquet", "")}_authors.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response

@app.route('/choroplethData/<file_name>/<comparator_type>/<comparator>/<custom>/<custom_size>', methods=['GET'])
def choroplethData(file_name, comparator_type=None, comparator=None, custom=False, custom_size=None):
    country_lookup = wr.s3.read_csv("s3://rootbucket/topic_clustering/test_folder/country_lookup.csv")
    ta7 = ["United Kingdom","Germany","Australia","New Zealand","Canada","France","Italy","Spain"]

    custom_bool = custom.lower() == 'true'

    table_prefix = "" if not custom_bool else f"[{custom_size}]"
    
    if not custom_bool:
        Author = AuthorsTablename(file_name)
    else:
        Author = customAuthorsTablename(table_prefix+file_name)

    if comparator_type == "region":
        if comparator != "TA7":
            result = db.session.query(
                Author.gpt_label,
                Author.prid_country,
                func.sum(Author.sum_published).label('publications')
            ).filter(Author.prid_region == comparator).group_by(
                Author.gpt_label,
                Author.prid_country
            ).all()
        else:
           result = db.session.query(
                Author.gpt_label,
                Author.prid_country,
                func.sum(Author.sum_published).label('publications')
            ).filter(Author.prid_country.in_(ta7)).group_by(
                Author.gpt_label,
                Author.prid_country
            ).all()
    elif comparator_type == "country":
        result = db.session.query(
                Author.gpt_label,
                Author.prid_country,
                func.sum(Author.sum_published).label('publications')
            ).filter(Author.prid_country == comparator).group_by(
                Author.gpt_label,
                Author.prid_country
            ).all()
    else:
        result = db.session.query(
            Author.gpt_label,
            Author.prid_country,
            func.sum(Author.sum_published).label('publications')
        ).group_by(
            Author.gpt_label,
            Author.prid_country
        ).all()

    output_dict = {}
    for row in result:
        gpt_label, prid_country, publications = row
        if gpt_label not in output_dict:
            output_dict[gpt_label] = []

        if prid_country in country_lookup['country'].tolist():
            geojson = country_lookup.loc[country_lookup['country'] == prid_country, 'geojson'].values[0]
        else:
            geojson = prid_country

        output_dict[gpt_label].append({
            "country": geojson,
            "publications": publications
        })

    return jsonify(output_dict), 200

@app.route('/custom_cluster_size/<file_name>/<new_min_cluster_size>', methods=['GET'])
def custom_cluster_size_dashboard(file_name, new_min_cluster_size):
    with app.app_context():
        params = get_params(file_name)
        change = ((params["min_cluster_size"]-int(new_min_cluster_size))/params["min_cluster_size"])*100
        new_min_samples = params["min_samples"] * change/100
        params["min_cluster_size"] = int(new_min_cluster_size)
        params["min_samples"] = round(new_min_samples) if round(new_min_samples) > 1 else 1

        exemplarsTable, authorsTable = get_tables(f"[{new_min_cluster_size}]"+file_name, custom=True)

        try:
            cluster_labels, params = init_db_and_get_labels_params(file_name, custom=True, custom_size=new_min_cluster_size)
            summary = topic_summary(cluster_labels)
            exemplars = db.session.query(exemplarsTable).filter_by(exemplar=True).order_by(text("cluster_label")).all()
            authors = db.session.query(authorsTable).order_by(text("gpt_label, avg_cites_per_article desc")).all()
            return render_template("results.html", file_name=file_name, params=params, exemplars=exemplars, summary=summary, authors=authors, clusters=sorted([str(x) for x in cluster_labels.gpt_label.unique().tolist()]), custom = True, new_min_cluster_size = new_min_cluster_size)
        except:
            articles = wr.s3.read_parquet(f"s3://rootbucket/topic_clustering/test_folder/umaps/{file_name}")
            authors = wr.s3.read_parquet(f"s3://rootbucket/topic_clustering/test_folder/authors/{file_name}")
            cluster_labels, x, y = get_cluster_labels(articles, params)
            topic_labels = create_gpt_label_dataframe(cluster_labels[cluster_labels["exemplar"]==True])
            cluster_labels = cluster_labels.merge(topic_labels, on="cluster_label", how="left")
            cluster_labels["gpt_label"] = cluster_labels["gpt_label"].apply(
                lambda x: re.sub(r'[^\w\s]', '', x) if not pd.isna(x) else x
            )
            cluster_labels["prid_country"] = cluster_labels["prid_country"].apply(lambda x: str(x))
            cluster_labels["prid_region"] = cluster_labels["prid_region"].apply(lambda x: str(x))
            group_authors_table = group_authors(cluster_labels, authors)
            database_write(cluster_labels, f"[{new_min_cluster_size}]"+file_name.replace(".parquet", ""), "custom_clustering_data")
            database_write(group_authors_table, f"[{new_min_cluster_size}]"+file_name.replace(".parquet", ""), "custom_authors")
            summary = topic_summary(cluster_labels)
            exemplars = db.session.query(exemplarsTable).filter_by(exemplar=True).order_by(text("cluster_label")).all()
            authors = db.session.query(authorsTable).order_by(text("gpt_label, avg_cites_per_article desc")).all()
            return render_template("results.html", file_name=file_name, params=params, exemplars=exemplars, summary=summary, authors=authors, clusters=sorted([str(x) for x in cluster_labels.gpt_label.unique().tolist()]), custom = True, new_min_cluster_size=int(new_min_cluster_size))

@app.route('/custom_cluster_size_comparator/<file_name>/<new_min_cluster_size>/<comparator_type>/<comparator>', methods=['GET'])
def custom_cluster_size_comparator_dashboard(file_name, new_min_cluster_size, comparator_type, comparator):
    with app.app_context():
        params = get_params(file_name)
        new_min_cluster_size = int(new_min_cluster_size)
        change = ((params["min_cluster_size"]-new_min_cluster_size)/params["min_cluster_size"])*100
        new_min_samples = params["min_samples"] * change/100
        params["min_cluster_size"] = int(new_min_cluster_size)
        params["min_samples"] = round(new_min_samples) if round(new_min_samples) > 1 else 1

        exemplarsTable, authorsTable = get_tables(f"[{new_min_cluster_size}]"+file_name, custom=True)
        for attempt in range(2):
            try:
                cluster_labels, params = init_db_and_get_labels_params(file_name, custom=True, custom_size=new_min_cluster_size)
                summary = topic_summary(cluster_labels)
                if comparator_type in ["region", "journal", "country", "publisher"]:
                    if comparator_type == "region":
                        ta7 = ["United Kingdom","Germany","Australia","New Zealand","Canada","France","Italy","Spain"]
                        if comparator != "TA7":
                                cluster_labels_region = cluster_labels[cluster_labels["prid_region"].apply(lambda x: comparator in x)]
                        elif comparator == "TA7":
                            cluster_labels_region = cluster_labels[cluster_labels["prid_country"].apply(lambda x: any(country in x for country in ta7))]
                        comp_summary = comparator_summary(cluster_labels, cluster_labels_region)

                        if comparator != "TA7":
                                exemplars = db.session.query(exemplarsTable).filter(exemplarsTable.prid_region.contains(str(comparator))).filter_by(exemplar=True).order_by(text("cluster_label")).all()
                                authors = db.session.query(authorsTable).filter_by(prid_region=comparator).order_by(text("gpt_label, avg_cites_per_article desc")).all()
                        elif comparator == "TA7":
                            exemplars = db.session.query(exemplarsTable).filter(exemplarsTable.exemplar == True).filter(or_(*[exemplarsTable.prid_country.contains(country) for country in ta7])).order_by(text("cluster_label")).all()
                            authors = db.session.query(authorsTable).filter(or_(*[authorsTable.prid_country.contains(country) for country in ta7])).order_by(text("gpt_label, avg_cites_per_article desc")).all()
                        return render_template("results_comparator.html", file_name=file_name, params=params, exemplars=exemplars, summary=summary, authors=authors, clusters=sorted([str(x) for x in cluster_labels.gpt_label.unique().tolist()]), comparator = comparator, comp_summary = comp_summary, comparator_type = comparator_type, custom = True)
                    elif comparator_type == "journal":
                        cluster_labels_journal = cluster_labels[cluster_labels["full_source_title"].apply(lambda x: comparator in x)]
                        comp_summary = comparator_summary(cluster_labels, cluster_labels_journal)
                        exemplars = db.session.query(exemplarsTable).filter(and_(exemplarsTable.full_source_title == comparator, exemplarsTable.exemplar == True)).order_by(text("cluster_label")).all()
                        authors = db.session.query(authorsTable).filter(authorsTable.full_source_title_list.contains("'"+str(comparator)+"'")).order_by(text("gpt_label, avg_cites_per_article desc")).all()
                        return render_template("results_comparator.html", file_name=file_name, params=params, exemplars=exemplars, summary=summary, authors=authors, clusters=sorted([str(x) for x in cluster_labels.gpt_label.unique().tolist()]), comparator = comparator, comp_summary = comp_summary, comparator_type = comparator_type, custom = True)
                    elif comparator_type == "country":
                        cluster_labels_country = cluster_labels[cluster_labels["prid_country"].apply(lambda x: comparator in x)]
                        comp_summary = comparator_summary(cluster_labels, cluster_labels_country)
                        exemplars = db.session.query(exemplarsTable).filter(exemplarsTable.prid_country.contains(str(comparator))).filter_by(exemplar=True).order_by(text("cluster_label")).all()
                        authors = db.session.query(authorsTable).filter_by(prid_country=comparator).order_by(text("gpt_label, avg_cites_per_article desc")).all()
                        return render_template("results_comparator.html", file_name=file_name, params=params, exemplars=exemplars, summary=summary, authors=authors, clusters=sorted([str(x) for x in cluster_labels.gpt_label.unique().tolist()]), comparator = comparator, comp_summary = comp_summary, comparator_type = comparator_type, custom = True)
                    elif comparator_type == "publisher":
                        cluster_labels_publisher = cluster_labels[cluster_labels["publisher_group"]==comparator.upper()]
                        comp_summary = comparator_summary(cluster_labels, cluster_labels_publisher)
                        exemplars = db.session.query(exemplarsTable).filter(and_(exemplarsTable.publisher_group == comparator.upper(), exemplarsTable.exemplar == True)).order_by(text("cluster_label")).all()
                        authors = db.session.query(authorsTable).filter(authorsTable.publisher_group_list.contains("'"+str(comparator.upper())+"'")).order_by(text("gpt_label, avg_cites_per_article desc")).all()
                        return render_template("results_comparator.html", file_name=file_name, params=params, exemplars=exemplars, summary=summary, authors=authors, clusters=sorted([str(x) for x in cluster_labels.gpt_label.unique().tolist()]), comparator = comparator, comp_summary = comp_summary, comparator_type = comparator_type, custom = True)
                    
            except:
                articles = wr.s3.read_parquet(f"s3://rootbucket/topic_clustering/test_folder/umaps/{file_name}")
                authors = wr.s3.read_parquet(f"s3://rootbucket/topic_clustering/test_folder/authors/{file_name}")
                cluster_labels, x, y = get_cluster_labels(articles, params)
                topic_labels = create_gpt_label_dataframe(cluster_labels[cluster_labels["exemplar"]==True])
                cluster_labels = cluster_labels.merge(topic_labels, on="cluster_label", how="left")
                cluster_labels["gpt_label"] = cluster_labels["gpt_label"].apply(
                    lambda x: re.sub(r'[^\w\s]', '', x) if not pd.isna(x) else x
                )
                cluster_labels["prid_country"] = cluster_labels["prid_country"].apply(lambda x: str(x))
                cluster_labels["prid_region"] = cluster_labels["prid_region"].apply(lambda x: str(x))
                group_authors_table = group_authors(cluster_labels, authors)
                database_write(cluster_labels, f"[{new_min_cluster_size}]"+file_name.replace(".parquet", ""), "custom_clustering_data")
                database_write(group_authors_table, f"[{new_min_cluster_size}]"+file_name.replace(".parquet", ""), "custom_authors")

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.errorhandler(Exception)
def unhandled_exception(e):
    db.session.rollback()  # Rollback the session
    app.logger.error('Unhandled Exception: %s', e)
    return "Internal server error", 500

if __name__ == "__main__":
    app.run(debug=True) 