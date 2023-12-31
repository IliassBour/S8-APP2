"""
Classe "ImageCollection" pour charger et visualiser les images de la problématique
Membres :
    image_folder: le sous-répertoire d'où les images sont chargées
    image_list: une énumération de tous les fichiers .jpg dans le répertoire ci-dessus
    images: une matrice de toutes les images, (optionnelle, changer le flag load_all du constructeur à True)
    all_images_loaded: un flag qui indique si la matrice ci-dessus contient les images ou non
Méthodes pour la problématique :
    generateRGBHistograms : calcul l'histogramme RGB de chaque image, à compléter
    generateRepresentation : vide, à compléter pour la problématique
Méthodes génériques :
    generateHistogram : histogramme une image à 3 canaux de couleurs arbitraires
    images_display: affiche quelques images identifiées en argument
    view_histogrammes: affiche les histogrammes de couleur de qq images identifiées en argument
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random
from enum import IntEnum, auto

from skimage import color as skic
from skimage import io as skiio
from skimage import restoration as skir
from scipy import ndimage

import helpers.analysis as an


class ImageCollection:
    """
    Classe globale pour regrouper les infos utiles et les méthodes de la collection d'images
    """
    class imageLabels(IntEnum):
        coast = auto()
        forest = auto()
        street = auto()

    def __init__(self, load_all=False):
        # liste de toutes les images
        self.image_folder = r"data" + os.sep + "baseDeDonneesImages"
        self._path = glob.glob(self.image_folder + os.sep + r"*.jpg")
        image_list = os.listdir(self.image_folder)
        # Filtrer pour juste garder les images
        self.image_list = [i for i in image_list if '.jpg' in i]

        self.all_images_loaded = False
        self.images = []

        # Crée un array qui contient toutes les images
        # Dimensions [980, 256, 256, 3]
        #            [Nombre image, hauteur, largeur, RGB]
        if load_all:
            self.images = np.array([np.array(skiio.imread(image)) for image in self._path])
            self.all_images_loaded = True

        self.labels = []
        for i in image_list:
            if 'coast' in i:
                self.labels.append(ImageCollection.imageLabels.coast)
            elif 'forest' in i:
                self.labels.append(ImageCollection.imageLabels.forest)
            elif 'street' in i:
                self.labels.append(ImageCollection.imageLabels.street)
            else:
                raise ValueError(i)

    def prep_images(self):
        coast = []
        forest = []
        street = []
        for image_counter in range(len(self.image_list)):
            # charge une image si nécessaire
            if self.all_images_loaded:
                imageRGB = self.images[image_counter]
            else:
                imageRGB = skiio.imread(
                    self.image_folder + os.sep + self.image_list[image_counter])

            # rgb_variance = self.getRGBVariance(imageRGB)/10
            lightness_max = self.max_luminence(imageRGB) / 10
            rgb_stand_er = self.standard_error_RGB(imageRGB) * 3000
            xyz_stand_er = self.standard_error_XYZ(imageRGB) * 1000000
            noise = self.calculate_noise(imageRGB) * 10000

            data = np.array([xyz_stand_er, noise, lightness_max, rgb_stand_er])
            img_class = self.labels[image_counter]
            if img_class == 1:
                coast.append(data)
            elif img_class == 2:
                forest.append(data)
            elif img_class == 3:
                street.append(data)

        dataList = []
        npcoast = np.array(coast)
        npforest = np.array(forest)
        npstreet = np.array(street)

        dataList.append(npcoast[:250])
        dataList.append(npforest[:250])
        dataList.append(npstreet[:250])

        return dataList

    def getRGBVariance(self, image):
        # from https://www.odelama.com/data-analysis/How-to-Compute-RGB-Image-Standard-Deviation-from-Channels-Statistics/
        red = image[..., 0]
        green = image[..., 1]
        blue = image[..., 2]
        avgR = np.sum(red) / (256*256)
        avgG = np.sum(green) / (256 * 256)
        avgB = np.sum(blue) / (256 * 256)
        varR = np.var(red)
        varG = np.var(green)
        varB = np.var(blue)

        varRGB = 0.333 * (varR + varG + varB +
                          np.square(avgR) + np.square(avgG) + np.square(avgB) -
                          avgR * avgG - avgR * avgB - avgG * avgB)
        return varRGB

    def standard_err_gray(self, data):
        img = skic.rgb2gray(data)
        return np.around(np.std(img) / 256, decimals=10)
    def max_luminence(self, data):
        n_bins = 256
        imageLab = skic.rgb2lab(data)
        imageLabhist = an.rescaleHistLab(imageLab, n_bins)
        histtvaluesLab = self.generateHistogram(imageLabhist)
        lum = histtvaluesLab[1]
        return lum[np.argmax(lum)]

    def calculate_noise(self, data):
        img = skic.rgb2gray(data)
        return skir.estimate_sigma(img, average_sigmas=True)

    def standard_error_RGB(self, data):
        return np.around(np.std(data[2])/256, decimals=10)

    def standard_error_XYZ(self, data):
        img = skic.rgb2xyz(data)
        return np.around(np.std(img[1])/256, decimals=10)

    def get_samples(self, N):
        return np.sort(random.sample(range(np.size(self.image_list, 0)), N))

    def generateHistogram(self, image, n_bins=256):
        # Construction des histogrammes
        # 1 histogram per color channel
        n_channels = 3
        pixel_values = np.zeros((n_channels, n_bins))
        for i in range(n_bins):
            for j in range(n_channels):
                pixel_values[j, i] = np.count_nonzero(image[:, :, j] == i)
        return pixel_values

    def generateRGBHistograms(self):
        """
        Calcule les histogrammes RGB de toutes les images
        """
        raise NotImplementedError()

    def generateRepresentation(self):
        # produce a ClassificationData object usable by the classifiers
        raise NotImplementedError()

    def images_display(self, indexes):
        """
        fonction pour afficher les images correspondant aux indices
        indexes: indices de la liste d'image (int ou list of int)
        """
        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]

        fig2 = plt.figure()
        ax2 = fig2.subplots(len(indexes), 1)
        for i in range(len(indexes)):
            if self.all_images_loaded:
                im = self.images[i]
            else:
                im = skiio.imread(self.image_folder + os.sep + self.image_list[indexes[i]])
            ax2[i].imshow(im)

    def view_histogrammes(self, indexes):
        """
        Affiche les histogrammes de couleur de quelques images
        indexes: int or list of int des images à afficher
        """
        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]

        fig = plt.figure()
        ax = fig.subplots(len(indexes), 3)

        for image_counter in range(len(indexes)):
            # charge une image si nécessaire
            if self.all_images_loaded:
                imageRGB = self.images[image_counter]
            else:
                imageRGB = skiio.imread(
                    self.image_folder + os.sep + self.image_list[indexes[image_counter]])

            # Exemple de conversion de format pour Lab et HSV
            imageLab = skic.rgb2lab(imageRGB)
            imageHSV = skic.rgb2hsv(imageRGB)
            imageXYZ = skic.rgb2xyz(imageRGB)
            print("math variance: ", self.getRGBVariance(imageRGB))
            #print("max luminence: ", self.max_luminence(imageRGB))
            print("noise: ", self.calculate_noise(imageRGB))
            print("standard error XYZ: ", self.standard_error_RGB(imageXYZ))
            print("standard error RGB: ", self.standard_error_RGB(imageRGB))
            # Number of bins per color channel pour les histogrammes (et donc la quantification de niveau autres formats)
            n_bins = 256

            # Lab et HSV requiert un rescaling avant d'histogrammer parce que ce sont des floats au départ!
            imageLabhist = an.rescaleHistLab(imageLab, n_bins) # External rescale pour Lab
            imageHSVhist = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100


            # Construction des histogrammes
            histvaluesRGB = self.generateHistogram(imageRGB)
            histtvaluesLab = self.generateHistogram(imageLabhist)
            histvaluesHSV = self.generateHistogram(imageHSVhist)


            # permet d'omettre les bins très sombres et très saturées aux bouts des histogrammes
            skip = 5
            start = skip
            end = n_bins - skip

            # affichage des histogrammes
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[0, start:end], s=3, c='red')
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[1, start:end], s=3, c='green')
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[2, start:end], s=3, c='blue')
            ax[image_counter, 0].set(xlabel='intensité', ylabel='comptes')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = self.image_list[indexes[image_counter]]
            ax[image_counter, 0].set_title(f'histogramme RGB de {image_name}')

            # 2e histogramme
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[0, start:end], s=3, c='magenta')
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[1, start:end], s=3, c='purple')
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[2, start:end], s=3, c='cyan')
            ax[image_counter, 1].set(xlabel='intensité', ylabel='comptes')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = self.image_list[indexes[image_counter]]
            ax[image_counter, 1].set_title(f'histogramme LAB de {image_name}')

            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[0, start:end], s=3, c='brown')
            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[1, start:end], s=3, c='blue')
            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[2, start:end], s=3, c='red')
            ax[image_counter, 2].set(xlabel='intensité', ylabel='comptes')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = self.image_list[indexes[image_counter]]
            ax[image_counter, 2].set_title(f'histogramme HSV de {image_name}')
