from sklearn.linear_model import LogisticRegression
import numpy as np

"""
Variables o Features:

"accelerometer",
	"gyroscope",
	"magnetometer",
	"humidity"
"""

#Creamos la matriz que contiene los datos de las variables (accelerometer, gyroscope, magnetometer, humidity)
feature_matrix = np.random.rand(300,4)
#Creamos el arreglo que contiene los labels
predict_label = (np.random.rand(300) > 0.5).astype(int)

#Checamos los datos contenidos en el predict_label, 1 mantenimiento, 0 no necesita
print(predict_label[0:20])

#Creamos el modelo usando un modelo de clasificacion
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(feature_matrix, predict_label)

#Hacemos una pequena prediccion
prueba = np.random.rand(1, 4)
print(clf.predict(prueba))

print(prueba)
"""
Recordemos si nos marca 1  necesita mantenimiento, de lo contrario 0 no necesita
"""
#Checamos el contenido de nuestro clasificador
print("Los pesos, son 4 ya que nuestros features o variables eran 4")
print(clf.coef_)
print("El interceptor")
print(clf.intercept_)

# Nota: esos pesos e interceptor son los que se usaran en el servidor para hacer la prediccion