_authors_ = ['1633198', '1636283', '1636526']
_group_ = 'DJ.10'

import numpy as np
import Kmeans as km
import KNN as kn
# from Kmeans import *
# from KNN import *

from utils_data import *
from utils_data import read_dataset, visualize_k_means, visualize_retrieval, Plot3DCloud
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)


    # You can start coding your functions here

    # FUNCIONS ANÀLISI QUALITATIU
    ## Retrieval_by_color
    """
    def get_color_predictions(images, max_k):
        preds = []

        for ix, input in enumerate(cropped_images):
            kms = km.KMeans(input, 1, {"km_init": 'first'})
            kms.find_bestK(max_k)
            kms.fit()
            preds.append(km.get_colors(np.array(kms.centroids)))

        return preds
    """
    def Retrieval_by_color(llista_imatges, etiquetes,
                           pregunta_colors):  # pregunta_colors és o un string o una llista d'strings. Retorna totes les imatges que contenen les etiquetes de la pregunta que fem
        llista1 = []

        for i in range(len(llista_imatges)):
            etiq = etiquetes[i]
            if all(j in etiq for j in pregunta_colors):
                llista1.append(llista_imatges[i])

        return llista1

        # Millorable afegint un paràmetre que sigui el percentatge de cada color i retorni les imatges ordenades


    ## Retrieval_by_shape

    def Retrieval_by_shape(llista_imatges, etiquetes,
                           pregunta_shape):  # pregunta_shape és un string definint la forma de la roba. Retorna totes les imatges que contenen l'etiqueta de la pregunta que fem.
        llista2 = []

        for i in range(len(llista_imatges)):
            etiq = etiquetes[i]
            if all(j in etiq for j in pregunta_shape):
                llista2.append(llista_imatges[i])

        return llista2


    # Millorable afegint un paràmetre que contignui  el percentatge de K-neighbours amb l'etiqueta que busquem i retorni les imatges ordenades

    ## Retrieval_combined

    def Retrieval_combined(llista_imatges, etiquetes_color, etiquetes_shape, pregunta_colors,
                           pregunta_shape):  # Retorna les imatges que coincideixin amb les dues preguntes ("Red Flip Flops")
        llista3 = []

        for i in range(len(llista_imatges)):
            etiqColor = etiquetes_color[i]
            etiqShape = etiquetes_shape[i]
            if all(x in etiqColor for x in pregunta_colors):
                if all(j in etiqShape for j in pregunta_shape):
                    llista3.append(llista_imatges[i])

        return llista3


    def get_color_accuracy(kmeans_labels, color_labels):


        """ #NUESTRO (0.5)
        suma = 0
        for x,y in zip(kmeans_labels, color_labels):
                if (x == y):
                suma = suma+1
        percentatge = suma / len(kmeans_labels) * 100
        print(percentatge)
        """
        """ #CON ESTO 65
        print("El color accuracy és:")
        aux = 0.0
        for i in range(len(kmeans_labels)):
            pred = set(kmeans_labels[i])
            aux += 1 - len(pred - set(color_labels[i])) / len(pred)
        return aux / len(kmeans_labels) * 100
        """

        suma = 0.0
        for i in range(len(kmeans_labels)):
            etiqKmeans = set(kmeans_labels[i])
            suma += 1 - len(etiqKmeans - set(color_labels[i])) / len(etiqKmeans)
        return suma / len(kmeans_labels) * 100

    def get_shape_accuracy(knn_labels, shape_labels):
        matching_labels = [x == y for x, y in zip(knn_labels, shape_labels)]
        percentatge = sum(matching_labels) / len(knn_labels) * 100
        print(percentatge)


    def kmean_statistics(element_kmeans, kmax=4):
        statistics = []

        for k in range(2, kmax + 1):
            element_kmeans.fit()
            wcd = element_kmeans.withinClassDistance()
            iterations = element_kmeans.num_iter

            statistics.append({'K': k, 'WCD': wcd, 'Iterations': iterations})

            print(f"K={k}: WCD={wcd}, Iterations={iterations}")


    # A PARTIR DE AQUI HAY QUE PASAR LAS LABELS DE DESPUES DE APLICAR KNN Y KMEANS
    # preguntaColor = ['Orange']
    # auxImgs = train_imgs[:50]
    # auxColorLabels = train_color_labels[:50]

    # preguntaShape = ['Dresses']
    # auxShapeLabels = train_class_labels[:50]

    # Utilitzem els retrievals
    # matchColor = Retrieval_by_color(auxImgs, auxColorLabels, preguntaColor)
    # matchShape = Retrieval_by_shape(auxImgs, auxShapeLabels, preguntaShape)
    # matchCombined = Retrieval_combined(auxImgs, auxColorLabels, auxShapeLabels, preguntaColor, preguntaShape)

    # Visualitzem els resultats
    # visualize_retrieval(np.array(matchColor), len(matchColor), title='Retrieval by color Test')
    # visualize_retrieval(np.array(matchShape), len(matchShape), title='Retrieval by shape Test')
    # visualize_retrieval(np.array(matchCombined), len(matchCombined), title='Retrieval COMBINED Test')

    # Retrieval by shape
    knn = kn.KNN(train_imgs, train_class_labels)
    totsFormes = knn.predict(imgs, 2)

    shapes = Retrieval_by_shape(imgs, totsFormes, 'Sandals')
    visualize_retrieval(np.array(shapes), len(shapes), title = 'Retrieval by shape KNN')

    totsColors = []
    for i in imgs:
        km1 = km.KMeans(i, 2, options={'km_init': 'first'})
        #km1.find_bestK(10)
        km1.fit()
        colors = km.get_colors(np.array(km1.centroids))
        totsColors.append(colors)
    """
    # Retrieval by color
    totsColors = []
    for i in imgs:
        kmm = km.KMeans(i, options={'km_init': 'first'})
        kmm.find_bestK(10)
        kmm.fit()
        color = km.get_colors(kmm.centroids)
        totsColors.append(color)
    """
    colours = Retrieval_by_color(imgs, totsColors, ['Blue'])
    visualize_retrieval(np.array(colours), len(colours), title = 'Retrieval by color KMEANS')

    # Retrieval combined amb KMEANS i KNN
    totsImages = Retrieval_combined(imgs, totsColors, totsFormes, ['Red'], 'Sandals')
    visualize_retrieval(np.array(totsImages), len(totsImages), title = 'Retrieval Combined amb KNN i KMEANS')

    # Fer també les millores de classificació

    # Visualitzar get_color_accuracy
    #kmeans_labels = ['red', 'blue', 'green', 'green', 'yellow']
    #color_labels = ['red', 'blue', 'green', 'green', 'red']
    #get_color_accuracy(kmeans_labels, color_labels)

    # Visualitzar get_shape_accuracy
    # knn_labels = [1, 2, 3, 4, 5]
    # shape_labels = [1, 2, 3, 5, 6]
    # get_shape_accuracy(knn_labels, shape_labels)

    #Get Shape Accuracy utilitzant KNN per conseguiur forma etiquetes
    #totsFormes Calculat al apartat del retrieval shape
    percentShape=get_shape_accuracy(totsFormes, class_labels)

    """
    etiqKmeans = []  # guardem el color utilitzant el kmeans
    for i in enumerate(cropped_images):  # anem recorrent totes les imatges que hi han una per una
        km1 = km.KMeans(i, 2, options={'km_init': 'first'})
        km1.find_bestK(10)
        km1.fit()
        etiqKmeans.append(km.get_colors(np.array(km1.centroids)))

    percentCol = get_color_accuracy(etiqKmeans, color_labels)

    # mirar
    """

    etiqKmeans2 = []

    for i, input in enumerate(cropped_images):
        kms = km.KMeans(input, {"km_init": 'first'})
        kms.find_bestK(7)
        kms.fit()
        etiqKmeans2.append(km.get_colors(np.array(kms.centroids)))

    #predicted_color_labels = get_color_predictions(imgs, 7)
    #preds = Retrieval_by_color(imgs, predicted_color_labels, "Black")
    #print(get_color_accuracy(predicted_color_labels, color_labels))

    preds = Retrieval_by_color(imgs, etiqKmeans2, "Black")
    print(get_color_accuracy(etiqKmeans2, color_labels))

    """
    # elkmeans = km.KMeans(test_imgs)
    # elkmeans.fit()
    # kmeans_labels = km.get_colors(elkmeans.centroids)
    # get_color_accuracy(kmeans_labels, color_labels)
    """