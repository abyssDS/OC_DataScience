import os

from movies_app.xm_movie_app import app

if __name__ == "__main__":
	app.debug = True
	host = os.environ.get('IP', '127.0.0.1')
	port = int(os.environ.get('PORT', 8080))
	app.run(host=host, port=port)
