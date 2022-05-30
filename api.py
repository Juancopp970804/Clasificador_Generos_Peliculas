#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from gensim.models import KeyedVectors
import gensim.downloader as api
from Clasificador_Peliculas import generar_Clasificacion

#TL_Words_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
TL_Words_vectors = api.load('word2vec-google-news-300')

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Prediccion de los generos de una pelicula',
    description='API para la prediccion de los generos de una pelicula')

ns = api.namespace('Clasificar',
     description='Generos de una pelicula')
   
parser = api.parser()

parser.add_argument(
    'year',
    type=int,
    required=True, 
    help='Ano de la pelicula',
    location='args')

parser.add_argument(
    'title',
    type=str,
    required=True,
    help='Titulo de la pelicula',
    location='args')

parser.add_argument(
    'plot',
    type=str,
    required=True,
    help='Descripcion o resumen de la pelicula',
    location='args')

resource_fields = api.model('Resource', {
    'Probabilidades': fields.String,
    'Categorias': fields.String,
})

@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        print(args)
        probs, categ = generar_Clasificacion(args['plot'],TL_Words_vectors)
        return {
                #    "result": 1
        #  "result": generar_Clasificacion(args['plot'],TL_Words_vectors)
        "Probabilidades": probs,
        "Categorias": categ
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
