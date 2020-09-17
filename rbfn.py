import numpy as np

class RBFN(object):

    def __init__(self, hidden_shape, sigma=1.0):
        """ Red Neuronal de Función de Base Radial
        # Arguments
            input_shape: Dimensión de los datos de entrada
            e.g. scalar functions have should have input_dimension = 1
            hidden_shape: Número de neuronas ocultas de función de base radial (centroides).
        """
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        return np.exp(-self.sigma*np.linalg.norm(center-data_point)**2)

    def _calculate_interpolation_matrix(self, X):
        """ Calcula la matriz de interpolación usando una función de activación.
        # Argumentos.
            X: Datos de entrenamiento.
        # Input shape.
            (num_data_samples, input_shape)
        # Retorna.
            G: Matriz de interpolación.
        """
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(
                        center, data_point)
        return G

    def _select_centers(self, X):
        random_args = np.random.choice(len(X), self.hidden_shape)
        centers = X[random_args]
        return centers

    def fit(self, X, Y):
        """ Ajusta los pesos usando regresión lineal.
        # Argumentos
            X: Muestras de entrenamiento.
            Y: Objetivos.
        # Input shape.
            X: (num_data_samples, input_shape)
            Y: (num_data_samples, input_shape)
        """
        self.centers = self._select_centers(X)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        """
        # Argumentos
            X: Datos de prueba.
        # Input shape.
            (num_test_samples, input_shape)
        """
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions