import numpy as np
import pandas as pd
from rdkit import Chem
import sys
from sklearn import datasets
from matplotlib import pyplot as plt
from tensorflow import square

from src.d04_ensemble_selection.MolDistances import get_distances
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import time
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from tensorflow.keras import regularizers, constraints, initializers, metrics, utils, optimizers
from tensorflow.keras.constraints import min_max_norm
from keras.constraints import maxnorm, nonneg
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf

plt.style.use('seaborn-white')

t = time.time()

mol = Chem.MolFromPDBFile("/home/kkajo/Workspace/Conformers/CsA/CsA_chcl3_noebounds_numconf5400.pdb", removeHs=False)
# mol = Chem.MolFromPDBFile("/home/kkajo/Workspace/Conformers/CsA/CsA_chcl3_noebounds_numconf294.pdb", removeHs=False)
noe_df = pd.read_csv("/home/kkajo/Workspace/Git/MacrocycleConfGenNOE/src/CsA/CsA_chcl3_noebounds.csv", sep="\s",
                     comment="#")

y_true, X = get_distances(mol, noe_df)
y_true = y_true.reshape(-1, 1)

# Initialize variables
m, n = X.shape

# define base mode
def baseline_model(input_shape=100):
    # create model
    model = Sequential()  # initializers.Constant(1 / input_shape)
    model.add(Dense(1, activation='linear', use_bias=False,
                    input_shape=(input_shape,),
                    kernel_regularizer=regularizers.l1(0),
                    kernel_constraint=constraints.NonNeg()))  # regularizers.l1(1e-40)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=metrics.MeanSquaredError())
    return model


class pos_normal(constraints.Constraint):
    """Enforces positivity of weights and that the sum is 1"""

    def __init__(self, ref_value):
        self.ref_value = ref_value

    def __call__(self, w):
        #w = tf.where(w < 0, 0, w)
        w = tf.nn.relu(w)
        w = w / (tf.reduce_sum(w) + sys.float_info.epsilon)
        return w

    def get_config(self):
        return {'ref_value': self.ref_value}


def asymmetric_loss(y_act, y_pred):
    """Asymmetric squared loss. if predicted value is smaller than actual,
    reduce loss trough mult with factor < 1"""
    factor = 1
    dy = y_act - y_pred
    # if predicted value is smaller than actual, reduce loss trough mult with factor < 1
    dy = tf.where(tf.math.greater(dy, tf.zeros_like(dy)), factor * dy, dy)
    # test = dy.shape[1]
    return tf.math.square(dy)/dy.shape[1]


model = Sequential([
    Dense(units=1, input_shape=(n,), use_bias=False, kernel_constraint=pos_normal(n),
          kernel_regularizer=regularizers.l1(1e6))
])

opt = optimizers.Adam(learning_rate=1e-5) # std 1e-3

# model = baseline_model(n)
#model.compile(loss='mean_squared_error', optimizer='adam')
model.compile(loss=asymmetric_loss, optimizer=opt)
model.summary()
# utils.plot_model(model, "my_first_model.png", show_shapes=True)
history = model.fit(X, y_true, batch_size=m, epochs=2000, shuffle=True)

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.ylim((0, 6))
plt.show()

y_pred = model.predict(X)
dy = y_true - y_pred
# y_pred = sc.inverse_transform(y_pred_scaled)

print(f"Sum of dy: {np.sum(dy)}")
weights = model.get_weights()
weights = np.stack(weights).reshape(-1, n).T
print(np.sum(weights))

sorted = np.sort(weights, axis=0)
print(sorted[-15:])
neg_count = len(list(filter(lambda x: (x <= 0), weights)))
pos_count = len(list(filter(lambda x: (x > 1e-3), weights)))
print("Positive numbers : ", pos_count)
print("Negative numbers : ", neg_count)


runtime = time.time() - t
print("Took {:.0f} s.".format(runtime))
