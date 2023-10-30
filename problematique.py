"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt

from keras.optimizers import Adam
import keras as K

from helpers.ClassificationData import ClassificationData
from helpers.ImageCollection import ImageCollection
import helpers.classifiers as classifiers


#######################################
def problematique_APP2():
    images = ImageCollection()
    # Génère une liste de N images, les visualise et affiche leur histo de couleur
    # TODO: voir L1.E4 et problématique
    if True:
        # TODO L1.E4.3 à L1.E4.5
        # Analyser quelques images pour développer des pistes pour le choix de la représentation
        points = images.prep_images()
        data = ClassificationData(points)
        data.getStats(gen_print=True)

        n_neurons = 7
        n_layers = 6

        nn = classifiers.NNClassify_APP2(data2train=data, data2test=data,
                                          n_layers=n_layers, n_neurons=n_neurons, innerActivation='relu',
                                          outputActivation='softmax', optimizer=Adam(), loss='categorical_crossentropy',
                                          metrics=['accuracy'],
                                          callback_list=[K.callbacks.EarlyStopping(patience=50, verbose=1, restore_best_weights=1),
                                                         classifiers.print_every_N_epochs(25)],     # TODO à compléter L2.E4
                                          experiment_title='NN Simple',
                                          n_epochs = 1000, savename='classification',
                                          ndonnees_random=5000, gen_output=True, view=True)
        #N = 6
        #im_list = images.get_samples(N)
        #print(im_list)
        #images.images_display(im_list)
        #images.view_histogrammes(im_list)

    # TODO L1.E4.6 à L1.E4.8
    # images.generateRepresentation()
    plt.show()


######################################
if __name__ == '__main__':
    problematique_APP2()
