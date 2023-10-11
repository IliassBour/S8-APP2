import numpy as np
from numpy import linalg as LA

#matrice de covariance
covariance = np.array([[2,1,0],[1,2,0],[0,0,7]])

#valeur propre, vecteur propre
eigenvalues, eigenvectors = LA.eig(covariance)

print(covariance)
print(eigenvalues)
print(eigenvectors)

