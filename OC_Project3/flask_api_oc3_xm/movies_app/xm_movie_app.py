from flask import Flask, Response, abort, request, jsonify

import json
import pandas as pd
import numpy as np

app = Flask(__name__)

methods_list = ('closest_overall', 'closest_genres', 'closest_cast', 'closest_pict', 'closest_success')
methods={}
closest_overall = np.ndarray
closest_genres = np.ndarray
closest_cast = np.ndarray
closest_pict = np.ndarray
closest_success = np.ndarray

for i in methods_list:
	locals()[i] = np.load('ressources/'+i+'.npy')

	
df_movies = pd.read_csv('ressources/movies_titles_list.csv')[['movie_title']]
df_movies['id'] = df_movies.index
print (df_movies.shape[0] ,'movies imported')

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
	content = """<h1> Welcome</h1>	<p><a href='/recommend/0'> 
	Use this format to get movies recommendations by Id</a></p>	
	<table>
	<tr>
    <th>Movie Id</th>
    <th>Movie Name</th> 
	<th>Recommandations</th>
	</tr>"""
	i = 0
	for movie in df_movies['movie_title']:
		link = '<a href="/recommend/%s"> Recommendations </a>' % (i)
		content+= '<tr><td>%s</td><td>%s</td><td>%s</td></tr>' % (i, movie, link)
		i+=1
		
	content+='</table>'
	return content, 200

@app.route('/favicon.ico')
def favicon():
	return '', 200

@app.route('/recommend/<int:movie_id>')
def first_reco(movie_id):
	return page_reco(movie_id, 1)
	
@app.route('/recommend/<int:movie_id>/<int:page_id>')
def page_reco(movie_id, page_id):
	df_reco = reco(movie_id, page_id)
	reco_json= df_reco.to_json(orient ='records')
	next_reco_link = str(request.url_root)+'recommend/'+str(movie_id)+'/'+str(page_id+1)
	content =  '{"Recommendations for %s":%s,"Next recommendations": "%s"}' % (df_movies.loc[movie_id, 'movie_title'], reco_json, next_reco_link )
	response = app.response_class(
        response=content
       , mimetype='application/json'
    )
	return response
	
def reco(movie_id, page):
	df_reco  =  pd.DataFrame()
	for i in methods_list:
		method = globals()[i]
		reco_movie_id = method[movie_id][page-1]
		df_reco = df_reco.append(df_movies.iloc[reco_movie_id])
	return df_reco


@app.errorhandler(404)
def not_found(e):
	return '',404