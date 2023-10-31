"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt

import helpers.classifiers as classifiers
from helpers.ClassificationData import ClassificationData
from helpers.ImageCollection import ImageCollection
from keras.optimizers import Adam
import keras as K


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
        #data.getStats(gen_print=True)
        # Figure avec les ellipses et les frontières
        #data.getBorders(view=True)
        if True:
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

    # TODO L1.E4.6 à L1.E4.8
    # images.generateRepresentation()
    plt.show()


######################################
if __name__ == '__main__':
    problematique_APP2()
