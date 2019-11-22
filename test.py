from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

cd = pd.read_csv("NNdata.csv")

categoricalVars = cd[["internetservice", "contract", "paymentmethod"]]

exit(0)
for categoical_var in categoical_vars :   
  model = Sequential()
  no_of_unique_cat  = df_train[categorical_var].nunique()
  embedding_size = min(np.ceil((no_of_unique_cat)/2), 50 )
  embedding_size = int(embedding_size)
  vocab  = no_of_unique_cat+1
  model.add( Embedding(vocab ,embedding_size, input_length = 1 ))
  model.add(Reshape(target_shape=(embedding_size,)))
  models.append( model )

model_rest = Sequential()
model_rest.add(Dense(16, input_dim= 1 ))
models.append(model_rest)

full_model.add(Merge(models, mode='concat'))