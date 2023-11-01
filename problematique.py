"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt

import helpers.classifiers as classifiers
from sklearn.model_selection import train_test_split as ttsplit
from helpers.ClassificationData import ClassificationData
from helpers.ImageCollection import ImageCollection
from keras.optimizers import Adam
import keras as K

#######################################
def problematique_APP2():
    images = ImageCollection()
    # Génère une liste de N images, les visualise et affiche leur histo de couleur
    if True:
        # Analyser quelques images pour développer des pistes pour le choix de la représentation
        points = images.prep_images()
        data = ClassificationData(points)
        data.getStats(gen_print=True)

        n_neurons = 8
        n_layers = 7

        nn = classifiers.NNClassify_APP2(data2train=data, data2test=data,
                                          n_layers=n_layers, n_neurons=n_neurons, innerActivation='tanh',
                                          outputActivation='softmax', optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy',
                                          metrics=['accuracy'],
                                          callback_list=[K.callbacks.EarlyStopping(patience=50, verbose=1, restore_best_weights=1),
                                                         classifiers.print_every_N_epochs(25)],
                                          experiment_title='NN Simple',
                                          n_epochs = 1500, savename='classification',
                                          ndonnees_random=5000, gen_output=True, view=True)

        ppv = classifiers.PPVClassify_APP2(data2train=data, data2test=data, n_neighbors=3, ndonnees_random=5000,
                                           useKmean=False, n_representants=5, experiment_title='K-ppv',
                                           gen_output=True, view=True) # erreur 10.6668% sans cluster, erreur 27.733% avec cluster

        apriori = [1 / 3, 1 / 3, 1 / 3]
        cost = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        bg1 = classifiers.BayesClassify_APP2(data2train=data, data2test=data,
                                             apriori=apriori, costs=cost,
                                             experiment_title='probabilités gaussiennes',
                                             gen_output=True, view=True)
        #N = 6
        #im_list = images.get_samples(N)
        #print(im_list)
        #images.images_display(im_list)
        #images.view_histogrammes(im_list)

    # images.generateRepresentation()
    plt.show()


######################################
if __name__ == '__main__':
    problematique_APP2()
