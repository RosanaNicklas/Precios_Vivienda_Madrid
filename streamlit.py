import pickle
from numpy import asarray
#import xgboost as xgb
from xgboost import XGBRegressor
model = XGBRegressor()
# fit model
model.fit(X,y)

# Guardar el modelo entrenado
nombre_archivo = 'model_entrenado.pkl'
with open(nombre_archivo, 'wb') as archivo:
    pickle.dump(model, archivo)

print(f"Modelo guardado en {nombre_archivo}")
