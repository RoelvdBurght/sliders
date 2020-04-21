import keras
import numpy as np 
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Model
import matplotlib.pyplot as plt


class lstm_model():
	"""
	Model class which does all the modelling 8-)
	"""
	def __init__(self):
		pass


	def read_in_data(self, file_location):
		self.X_train = pickle.load(open(file_location + 'X_train.pickle', 'rb'))
		self.X_test = pickle.load(open(file_location + 'X_test.pickle', 'rb'))
		self.X_val = pickle.load(open(file_location + 'X_val.pickle', 'rb'))
		self.y_train = pickle.load(open(file_location + 'y_train.pickle', 'rb'))
		self.y_test = pickle.load(open(file_location + 'y_test.pickle', 'rb'))
		self.y_val = pickle.load(open(file_location + 'y_val.pickle', 'rb'))

	def scale_data(self, type):
		if type == 'standard':
			self.scaler = StandardScaler()
		elif type == 'minmax':
			self.scaler = MinMaxScaler()
			scaler = StandardScaler()

		# Scale the data. Reshapes are neccesary because scaler only take 2D arrays
		self.X_train = self.scaler.fit_transform(self.X_train.reshape(-1, self.X_train.shape[-1])).reshape(self.X_train.shape)
		self.X_test = self.scaler.transform(self.X_test.reshape(-1, self.X_test.shape[-1])).reshape(self.X_test.shape)
		self.X_val = self.scaler.transform(self.X_val.reshape(-1, self.X_val.shape[-1])).reshape(self.X_val.shape)

	def define_model(self, loss_func='mse', optimizer='adam'):
		timesteps, n_features = self.X_train.shape[1], self.X_train.shape[2]
		inp = Input(shape=(timesteps, n_features))
		lstm_layer = LSTM(32, return_sequences=True)(inp)
		lstm_layer2 = LSTM(16)(lstm_layer)
		dense_layer = Dense(1)(lstm_layer2)
		self.model = Model(inputs=inp, outputs=dense_layer)
		
		self.model.summary()
		self.model.compile(optimizer=optimizer, loss=loss_func)

	def train(self, epochs=100):
		self.model.fit(self.X_train, self.y_train, epochs=epochs, verbose=1)

	def evaluate(self):
		print(self.model.evaluate(x=self.X_val, y=self.y_val))

	def predict_test(self):
		predictions = self.model.predict(self.X_test)
		fig, ax = plt.subplots()
		ax.scatter(predictions, self.y_test)
		ax.set_xlim(0, 10)
		ax.set_ylim(0, 10)
		plt.show()

lstm = lstm_model()
lstm.read_in_data('../data/')
lstm.scale_data('standard')
lstm.define_model()
lstm.train()
lstm.evaluate()
lstm.predict_test()