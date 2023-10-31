"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt

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
        # Figure avec les ellipses et les frontières
        data.getBorders(view=True)

        ppv = classifiers.PPVClassify_APP2(data2train=data, data2test=data, n_neighbors=3, ndonnees_random=5000,
                                           useKmean=False, n_representants=5, experiment_title='K-ppv',
                                           gen_output=True, view=True) # erreur 10.6668% sans cluster, erreur 27.733% avec cluster


        #ppv.predictTest
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
