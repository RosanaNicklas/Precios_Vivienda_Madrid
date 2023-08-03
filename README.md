# Precio_Vivienda_Madrid
Análisis del precio de las viviendas en Madrid con Python y sus librerias.


![craiyon_201845_Pisos_madrid](https://user-images.githubusercontent.com/98030137/234825461-1fc45eb3-0a09-484e-ac84-1282f7209a42.png)
![craiyon_202822_python](https://user-images.githubusercontent.com/98030137/234827222-c68ed33e-7320-4206-a680-491f4e8defd9.png)
![craiyon_201506_venta_pisos_madrid](https://user-images.githubusercontent.com/98030137/234825470-cb87e709-ca88-4820-8bc6-b726e7cc6b51.png)


Al realizar una predicción con machine learning sobre los precios de vivienda en Madrid, el objetivo es desarrollar un modelo que pueda aprender patrones y relaciones entre las características de las propiedades y sus respectivos precios en el área de Madrid. Una vez entrenado, este modelo se podrá utilizar para hacer predicciones precisas sobre el precio de una vivienda desconocida, basándose en sus características.

El proceso general para realizar esta predicción podría ser el siguiente:

1. **Obtención y preparación de los datos:** Se recolectarían datos históricos de ventas o alquileres de viviendas en Madrid. Estos datos deberían incluir información sobre las características de cada propiedad (como tamaño, número de habitaciones, ubicación, comodidades, etc.) y su precio de venta o alquiler. Luego, se realizaría un proceso de limpieza y preprocesamiento para asegurar que los datos sean consistentes y adecuados para el modelo.

2. **División de los datos:** El conjunto de datos se dividiría en dos partes: un conjunto de entrenamiento y un conjunto de prueba. El conjunto de entrenamiento se utilizará para entrenar el modelo y aprender las relaciones entre las características y los precios. El conjunto de prueba se utilizará para evaluar el rendimiento del modelo en datos no vistos.

3. **Selección del algoritmo de machine learning:** Se seleccionaría un algoritmo adecuado para el problema de predicción de precios de vivienda. Los algoritmos comunes para este tipo de tareas son regresión lineal, regresión de árbol de decisión, bosques aleatorios, XGBoost, entre otros.

4. **Entrenamiento del modelo:** Se alimentaría el conjunto de entrenamiento al algoritmo seleccionado, y este aprendería a hacer predicciones basadas en las características de las viviendas. Durante el entrenamiento, el modelo ajustará sus parámetros para minimizar el error entre las predicciones y los valores reales de los precios.

5. **Validación y ajuste de hiperparámetros:** Es importante realizar una validación cruzada y ajuste de hiperparámetros para evitar el sobreajuste y encontrar la configuración óptima del modelo.

6. **Evaluación del modelo:** Una vez entrenado, el modelo se evaluará utilizando el conjunto de prueba para medir su rendimiento en datos no vistos. Se analizarán métricas como el error medio absoluto, el error cuadrático medio o el coeficiente de determinación (R²) para evaluar qué tan bien se ajusta el modelo a los datos de prueba.

7. **Predicciones:** Con el modelo entrenado y evaluado, estará listo para realizar predicciones sobre precios de viviendas desconocidas en Madrid. Se proporcionarían las características de una propiedad (por ejemplo, tamaño, número de habitaciones, ubicación, etc.) y el modelo calculará su precio estimado.

8. **Despliegue y monitoreo:** Finalmente, el modelo entrenado se pondría en producción y se utilizaría para hacer predicciones en tiempo real. Además, se monitorearía su rendimiento y se podría reentrenar periódicamente con nuevos datos para mantener su precisión a lo largo del tiempo.

Es importante tener en cuenta que el rendimiento y precisión del modelo dependerán en gran medida de la calidad de los datos y de las características seleccionadas. También es esencial mantener una evaluación constante para asegurar que el modelo se mantenga relevante y preciso con el tiempo, ya que los precios de las viviendas y las preferencias del mercado pueden cambiar.

El dataset que tienes contiene una serie de características (features) que describen diferentes aspectos de propiedades inmobiliarias. A continuación, te explicaré cada una de las características:

1. 'título': El título del anuncio o publicación de la propiedad.

2. 'subtítulo': Un subtítulo adicional que complementa la descripción de la propiedad.

3. 'm2_construidos': La cantidad de metros cuadrados construidos de la propiedad.

4. 'm2_útiles': La cantidad de metros cuadrados útiles de la propiedad (área habitable).

5. 'n_habitaciones': El número de habitaciones (dormitorios) que tiene la propiedad.

6. 'n_baños': El número de baños que tiene la propiedad.

7. 'n_pisos': El número de pisos (plantas) que tiene la propiedad.

8. 'm2_asignación': La cantidad de metros cuadrados asignados a la propiedad, que podría referirse a áreas comunes en un edificio.

9. 'latitud': La coordenada de latitud geográfica de la ubicación de la propiedad.

10. 'longitud': La coordenada de longitud geográfica de la ubicación de la propiedad.

11. 'raw_address': La dirección sin procesar de la propiedad.

12. 'is_exact_address_hidden': Un valor booleano que indica si la dirección exacta de la propiedad está oculta o no.

13. 'street_name': El nombre de la calle en la que se encuentra la propiedad.

14. 'calle_número': El número de la calle donde se encuentra la propiedad.

15. 'portal': Número de portal o entrada a la propiedad (en edificios).

16. 'piso': El número de piso en el que se encuentra la propiedad.

17. 'es_piso_debajo': Un valor booleano que indica si la propiedad está en un piso debajo del nivel de la calle.

18. 'puerta': El número de puerta o apartamento en el que se encuentra la propiedad.

19. 'neighborhood_id': Identificador único del barrio o vecindario en el que se encuentra la propiedad.

20. 'operación': El tipo de operación, que podría ser "venta" o "alquiler".

21. 'rent_price': El precio de alquiler de la propiedad.

22. 'rent_price_by_area': El precio de alquiler dividido por el área útil de la propiedad.

23. 'is_rent_price_known': Un valor booleano que indica si el precio de alquiler es conocido o no.

24. 'buy_price': El precio de venta de la propiedad.

25. 'buy_price_by_area': El precio de venta dividido por el área útil de la propiedad.

26. 'is_buy_price_known': Un valor booleano que indica si el precio de venta es conocido o no.

27. 'house_type_id': Identificador único del tipo de propiedad (casa, piso, chalet, etc.).

28. 'is_renewal_needed': Un valor booleano que indica si la propiedad necesita renovación o no.

29. 'es_nuevo_desarrollo': Un valor booleano que indica si la propiedad es de nuevo desarrollo o no.

30. 'año_construido': El año de construcción de la propiedad.

31. 'tiene_calefacción_central': Un valor booleano que indica si la propiedad tiene calefacción central.

32. 'tiene_calefacción_individual': Un valor booleano que indica si la propiedad tiene calefacción individual.

33. 'se_admiten_mascotas': Un valor booleano que indica si se admiten mascotas en la propiedad.

34. 'tiene_ac': Un valor booleano que indica si la propiedad tiene aire acondicionado.

35. 'tiene_armarios_equipados': Un valor booleano que indica si la propiedad tiene armarios equipados.

36. 'tiene_ascensor': Un valor booleano que indica si la propiedad tiene ascensor.

37. 'es_exterior': Un valor booleano que indica si la propiedad es exterior.

38. 'tiene_jardín': Un valor booleano que indica si la propiedad tiene jardín.

39. 'tiene_piscina': Un valor booleano que indica si la propiedad tiene piscina.

40. 'tiene_terraza': Un valor booleano que indica si la propiedad tiene terraza.

41. 'tiene_balcón': Un valor booleano que indica si la propiedad tiene balcón.

42. 'tiene_trastero': Un valor booleano que indica si la propiedad tiene trastero.

43. 'está_amueblado': Un valor booleano que indica si la propiedad está amueblada.

44. 'está_cocina_equipada': Un valor booleano que indica si la propiedad tiene la cocina equipada.

45. 'es_accesible': Un valor booleano que indica si la propiedad es accesible para personas con discapacidad.

46. 'has_green_zones': Un valor booleano que indica si la propiedad cuenta con zonas verdes cercanas.

47. 'energy_certificate': Información sobre el certificado energético de la propiedad.

48. 'has_parking': Un valor booleano que indica si la propiedad tiene aparcamiento.

49. 'tiene_estacionamiento_privado': Un valor booleano que indica si la propiedad tiene estacionamiento privado.

50. 'tiene_estacionamiento_publico': Un valor booleano que indica si la propiedad tiene estacionamiento público cercano.

51. 'is_parking_included_in_price': Un valor booleano que indica si el aparcamiento está incluido en el precio de la propiedad.

52. 'parking_price': El precio del aparcamiento si no está incluido en el precio de la propiedad.

53. 'is_orientation_north': Un valor booleano que indica si la propiedad tiene orientación al norte.

54. 'es_orientación_oeste': Un valor booleano que indica si la propiedad tiene orientación al oeste.

55. 'es_orientación_sur': Un valor booleano que indica si la propiedad tiene orientación al sur.

56. 'es_orientación_este': Un valor booleano que indica si la propiedad tiene orientación al este.

Cada una de estas características puede ser relevante para el análisis y predicción de precios o características de las propiedades inmobiliarias. Dependiendo de tus objetivos, deberás seleccionar las características más importantes y realizar el preprocesamiento adecuado para utilizarlas en tus modelos de machine learning.
