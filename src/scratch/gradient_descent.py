from scratch.linear_algebra import Vector, dot

#Distancia al cuadrado
def sum_of_squares(v: Vector) -> float:
    """Computes the sum of squared elements in v"""
    return dot(v, v)

#importa el tipo Callable
from typing import Callable

#Calcula la derivada de f con x.
def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x + h) - f(x)) / h

#Vamos a comprobar que es una aproximación correcta, usando una x**2
def square(x: float) -> float:
    return x * x
#La derivada de x**2
def derivative(x: float) -> float:
    return 2 * x

#Devuelve una lista en la que cada valor es la derivada parcial en esa dirección
def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.0001):
    return [partial_difference_quotient(f, v, i, h)
            for i in range(len(v))]

import random
from scratch.linear_algebra import distance, add, scalar_multiply

#Calcula el siguiente vector despues de aplicar el gradiente sobre la posición inicial
def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

#el gradiente del cuadrado
def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]

# x ranges from -50 to 49, y is always 20 * x + 5
inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

#Evalua en un punto (x,y) el error, y el gradiente - el vector en el que el error se minimiza. El error depende de theta, y el gradiente nos apunta en la direccion en la que thera debe cambiarse
def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept    # The prediction of the model.
    error = (predicted - y)              # error is (predicted - actual)
    squared_error = error ** 2           # We'll minimize squared error
    grad = [2 * error * x, 2 * error]    # using its gradient.
    return grad

from typing import TypeVar, List, Iterator

#Definimos el minibatchs como un genérico
T = TypeVar('T')  # this allows us to type "generic" functions
#Creamos el minibatch que usaremos para adiestrar
def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[List[T]]:
    """Generates `batch_size`-sized minibatches from the dataset"""
    # Start indexes 0, batch_size, 2 * batch_size, ...
    #Indentifica los indices del dataset en los que tendra que empezar cada batch 
    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    #Desordenamos estos indices, de modo que sacaremos cada minibatch en un orden aleatorio
    if shuffle: random.shuffle(batch_starts)  # shuffle the batches

    #Es un iterator, que devuelve con cada next un minibatch. Cada minibatch tiene los items en el mismo orden en el que estaban en el dataset
    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]

def main():
    #Comparamos la derivada estimada vs la real
    xs = range(-10, 11)
    actuals = [derivative(x) for x in xs]
    estimates = [difference_quotient(square, x, h=0.001) for x in xs]
    
    # Pintamos ambos resultados
    import matplotlib.pyplot as plt
    plt.title("Actual Derivatives vs. Estimates")
    plt.plot(xs, actuals, 'rx', label='Actual')       # red  x
    plt.plot(xs, estimates, 'b+', label='Estimate')   # blue +
    #Colocamos la leyenda
    plt.legend(loc=9)
    plt.show()
    
    plt.close()
    
    #Calcula la derivada parcial de f con respecto a i en un espacio vectorial
    def partial_difference_quotient(f: Callable[[Vector], float],
                                    v: Vector,
                                    i: int,
                                    h: float) -> float:
        """Returns the i-th partial difference quotient of f at v"""
        #Calcula v+delta
        w = [v_j + (h if j == i else 0)    # add h to just the ith element of v
             for j, v_j in enumerate(v)]
        #Calcula la derivada
        return (f(w) - f(v)) / h
    
    # 1. Usando el método del Gradiente vamos a buscar el mínimo
    # Elige un punto al azar en el espacio
    v = [random.uniform(-10, 10) for i in range(3)]
    
    for epoch in range(1000):
        grad = sum_of_squares_gradient(v)    # el gradiente de la funcion v**2
        v = gradient_step(v, grad, -0.01)    # calcula el siguiente punto
        print(epoch, v)
    
    assert distance(v, [0, 0, 0]) < 0.001    # Calcula la distanci con respecto al origen (que es el minimo)
    
    
    # 1. Usamos Gradient Descent 
    # En inputs tenemos el dataset. En theta tenemos los parametros que queremos encontrar
    # Calculamos el gradiente sobre toda la muestra, sobre cada elemento del dataset, y calculamos una medida
    # Con el gradiente asi calculado, identificamos el nuevo valor de theta
    # Para evitar la inestabilidad de la búsqueda, que la solución converja, estamos calculando el gradiente sobre todos los datos del dataset

    from scratch.linear_algebra import vector_mean
    
    # QUeremos estimar theta. Elegimos un valor al azar de theta
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    
    learning_rate = 0.001
    
    #Empezamos el loop de entrenamiento
    for epoch in range(5000):
        # Compute the mean of the gradients
        # En inputs tenemos el espacio de valores (x,y). En cada punto vamos a calcular el gradiente. Usamos el valor medio de todos los gradientes 
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
        # Con el gradiente calculamos el nuevo valor de theta
        theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
    
    #Veamos si theta ha convergido a la solución correcta
    slope, intercept = theta
    assert 19.9 < slope < 20.1,   "slope should be about 20"
    assert 4.9 < intercept < 5.1, "intercept should be about 5"
    
    # 2. Minibatch gradient descent example
    # En inputs tenemos el dataset. En theta tenemos los parametros que queremos encontrar
    # Calculamos el gradiente sobre una parte de la muestra, el minibath, y calculamos una medida
    # Con el gradiente asi calculado, identificamos el nuevo valor de theta
    # Estamos Para evitar la inestabilidad de la búsqueda, que la solución converja, estamos calculando el sobre un subconjunto del dataset
    
    # Otra vez elegimos theta al azar
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    
    for epoch in range(1000):
        #Tom un minibatch
        for batch in minibatches(inputs, batch_size=20):
            #Entrenamos con el minibatch, en lugar de con todo el dataset
            grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
    
    slope, intercept = theta
    assert 19.9 < slope < 20.1,   "slope should be about 20"
    assert 4.9 < intercept < 5.1, "intercept should be about 5"
    
    
    # 3. Stochastic Gradient Descent
    # En este caso, parecido al anterior cuando el tamaño del minibatch es 1
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    
    for epoch in range(100):
        for x, y in inputs:
            grad = linear_gradient(x, y, theta)
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
    
    slope, intercept = theta
    assert 19.9 < slope < 20.1,   "slope should be about 20"
    assert 4.9 < intercept < 5.1, "intercept should be about 5"
    
if __name__ == "__main__": main()