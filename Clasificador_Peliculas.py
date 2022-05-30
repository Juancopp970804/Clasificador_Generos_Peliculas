import pandas as pd
#from gensim.models import KeyedVectors
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
import tensorflow as tf
import numpy as np

# import pickle


def plot_representation(document_i, TL_Words_vectors):
    # TL_Words_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    # print(1)

    avg_vector = 0
    for word in (document_i.lower().split()):
        try:
            avg_vector += TL_Words_vectors[word]
        except:
            continue
    avg_vector = avg_vector/len(document_i.lower().split())
    return avg_vector

def generar_Clasificacion(plot_prueba, TL_Words_vectors):
    vector_plot = pd.DataFrame()
    vector_plot[0] = plot_representation(plot_prueba, TL_Words_vectors)
    # print(vector_plot)
    vector_plot  = np.transpose(vector_plot)
    # print(vector_plot)
    

    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
            'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
            'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']


    dicc_salida = {}
    categorias_finales = []
    for i, categoria in enumerate(cols):
        try:
            del mod_load
        except:
            pass
            
        mod_load = load_model('modelos/red_' + categoria + '.h5')
        y_pred_genres = mod_load.predict(vector_plot)
        
        dicc_salida[categoria] = y_pred_genres[0][0]

        if y_pred_genres[0][0] >= 0.5:
            categorias_finales.append(categoria)

        if i == 0:
            y_pred_total = y_pred_genres
        else:
            y_pred_total = np.concatenate([y_pred_total,y_pred_genres], axis= 1)

    # print(dicc_salida)


    return dicc_salida, categorias_finales

# with open('saved_dictionary.pkl', 'rb') as f:
#     TL_Words_vectors = pickle.load(f)

# plot_prueba_1 = '''
# major benson winifred payne is being discharged from the marines .  payne is a killin '  machine ,  but the wars of the world are no longer fought on the battlefield .  a career marine ,  he has no idea what to do as a civilian ,  so his commander finds him a job  -  commanding officer of a local school ' s jrotc program ,  a bunch or ragtag losers with no hope .  using such teaching tools as live grenades and real bullets ,  payne starts to instill the corp with some hope .  but when payne is recalled to fight in bosnia ,  will he leave the corp that has just started to believe in him ,  or will he find out that killin '  ain ' t much of a livin '  ?
# '''

# print(generar_Clasificacion(plot_prueba_1, TL_Words_vectors))
