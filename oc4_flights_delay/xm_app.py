from flask import Flask,Markup, render_template, Response, abort, request, jsonify

from sklearn.externals import joblib
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np

app = Flask(__name__)

ressources_folder = 'oc4_flights_delay/ressources/'

df_carriers = pd.read_csv(ressources_folder+'df_carriers.csv')
df_airports = pd.read_csv(ressources_folder+'df_airports.csv')
df_days = pd.read_csv(ressources_folder+'df_days.csv', names=['DAY', 'day_ordered'])
df_tail_num = pd.read_csv(ressources_folder+'df_tail_num.csv', names=['Description', 'tail_num_ordered'])

X_cols=pd.read_csv(ressources_folder+'X_cols.csv')
ridge_model = joblib.load(ressources_folder+'ridge_model.joblib') 

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
  carriers_list =''
  for carrier in df_carriers['Description'].sort_values():
    carriers_list+='<option value="'+carrier+'">'+carrier+'</option>'	
    
  airports_list =''
  for airport in df_airports['Description'].sort_values():
    airports_list+='<option value="'+airport+'">'+airport+'</option>'	
  
  tail_num_list =''
  tail_num_list+='<option value="Unknown">Unknown</option>'
  for tail_num in df_tail_num['Description'].sort_values():
    tail_num_list+='<option value="'+tail_num+'">'+tail_num+'</option>'	
    
  return render_template('index.html', carriers_list=carriers_list, airports_list= airports_list, tail_num_list=tail_num_list)
  
@app.route('/estimated_delay', methods=['POST'])
def estimate_delay():
  dep_date  = request.form['dep_date']
  arr_date  = request.form['arr_date']
  origin_airport = request.form['origin_airport']
  dest_airport = request.form['dest_airport']
  carrier = request.form['carrier']
  tail_num = request.form['tail_num']
  
  df_pre_poly = first_transform(dep_date, arr_date, origin_airport, dest_airport, carrier, tail_num)
  X= poly_transform(df_pre_poly)
  
  estimated_delay = round(ridge_model.predict(X)[0] *100)/100
  
  response = 'Your estimated delay is :'+str(estimated_delay)+' minutes'
  return response, 200
  
@app.route('/favicon.ico')
def favicon():
  return '', 200
@app.errorhandler(404)
def not_found(e):
  return '',404
  
def myround(x, base):
    return int(base * round(float(x)/base))
    
def first_transform(dep_date, arr_date, origin_airport, dest_airport, carrier, tail_num):
  dep_date = datetime.strptime(dep_date, '%Y-%m-%dT%H:%M')
  dep_day = dep_date.timetuple().tm_yday
  
  arr_date = datetime.strptime(arr_date, '%Y-%m-%dT%H:%M')
  timedelta = arr_date-dep_date
  CRS_ELAPSED_TIME_10min = myround(timedelta.seconds/60, 10)
  DEP_TIME_5min = myround(dep_date.strftime('%H%M'), 5)
  ARR_TIME_5min = myround(arr_date.strftime('%H%M'), 5)
  
  day_ordered = df_days['day_ordered'][df_days['DAY'] == dep_day].values.item(0)
  carrier_ordered = df_carriers['carrier_ordered'][df_carriers['Description'] == carrier].values.item(0)
  tail_num_ordered = df_tail_num['tail_num_ordered'][df_tail_num['Description'] == tail_num].values.item(0)
  origin_airport_id_ordered = df_airports['origin_airport_id_ordered'][df_airports['Description'] == origin_airport].values.item(0)
  dest_airport_id_ordered = df_airports['dest_airport_id_ordered'][df_airports['Description'] == dest_airport].values.item(0)
  
  
  
  df_pre_poly = pd.DataFrame(
  [day_ordered, carrier_ordered, CRS_ELAPSED_TIME_10min,
  tail_num_ordered, origin_airport_id_ordered,
  dest_airport_id_ordered, DEP_TIME_5min, ARR_TIME_5min
  ]  ).T
  df_pre_poly.columns=['day_ordered', 'carrier_ordered', 'CRS_ELAPSED_TIME_10min',
  'tail_num_ordered', 'origin_airport_id_ordered',
  'dest_airport_id_ordered', 'DEP_TIME_5min', 'ARR_TIME_5min']
  
  return df_pre_poly
  
def poly_transform(df_pre_poly):
  X = df_pre_poly
  for index, row in X_cols.iterrows():
      nb_poly_features = row['nb_poly_features']
      col_new_name = row['new_name']
      
      df_col_ordered =  pd.DataFrame(X[col_new_name].unique(), columns=[col_new_name]).sort_values(col_new_name)
          
     # df_col_ordered
      if nb_poly_features>0:
          poly = PolynomialFeatures(nb_poly_features, include_bias  =False)
          poly_transfo = poly.fit_transform(df_col_ordered)
          df_poly = pd.DataFrame(poly_transfo, index=df_col_ordered[col_new_name].values).add_prefix(row['original_name'].lower()+'_poly_')

          X = X.join(df_poly , on =col_new_name, how='left').drop(col_new_name, axis=1)

  return X