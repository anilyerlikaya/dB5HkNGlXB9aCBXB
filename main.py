from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn import svm, metrics
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pandas.api.types import is_string_dtype
import numpy as np

# csv file must be in same folder with this python file
file_name = "term-deposit-marketing-2020.csv"

class ML_Solution():
	"""
	Machine Learning class for training a KNN model with 5-fold cross validation

	Attributes:
		model (KNN model) representing the KNN model that trained and tested with data
		X (array) represents data for training and testing
		y (array) represents label of data
	"""

	def __init__(self, model):
		self.model = model
		self.X = None
		self.Y = None

	def read_csv_file(self, fileName):
		"""
		Function to read csv file and split label from data

		Args: 
			fileName (string): name of the csv file 
		
		Returns:
			None
		"""

		data = pd.read_csv(fileName, sep=',', header=0)
		self.column_names = data.columns.tolist()

		for i in range(data.shape[1]):
			if is_string_dtype(data[self.column_names[i]]):
				data[self.column_names[i]] = self.string_to_int(data[self.column_names[i]])

		data = data.to_numpy()

		self.X = self.normalize(data[:, 0:data.shape[1]-1]) 
		self.y = data[:, data.shape[1]-1]

		return 

	def normalize(self, data):
		"""
		Function to normalize data

		Args:
			data (array): numerical data

		Returns:
			data (array): normalized data
		"""

		scalar = MinMaxScaler(feature_range=(0, 1))
		normalized_data = scalar.fit_transform(data)

		return normalized_data

	def string_to_int(self, data):
		"""
		Function to change non-numerical values into numerical value

		Args:
			data (array): string data

		Return:
			data (array): numerical data
		"""

		counter = 0
		int_dict = dict()

		for i in range(data.shape[0]):
			value = int_dict.get(data[i], -1)
			if value == -1:
				int_dict.update({data[i]: counter})
				data[i] = counter
				counter += 1
			else:
				data[i] = value

		return pd.to_numeric(data)

	def cross_validate(self):
		"""
		Function to train model using 5-fold cross validation

		Args:
			None

		Returns:
			avg_score (float): average score of model
		"""

		scores = []
		cv = KFold(n_splits=5, random_state=29, shuffle=False)

		for train_index, test_index in cv.split(self.X):
			X_train, X_test, y_train, y_test = self.X[train_index], self.X[test_index], self.y[train_index], self.y[test_index]

			self.train_model(X_train, y_train)
			score = self.test_model(X_test, y_test)
			scores.append(score)

		print()
		for i in range(len(scores)):
			print("{}. score: {}".format((i+1), scores[i]))

		return sum(scores) / len(scores)

	def train_model(self, train_data, train_label):
		"""
		Function to train model with training data and label

		Args: 
			train_data (array): data for train model
			train_label (array): label of training data 
		
		Returns:
			None
		"""

		self.model.fit(train_data, train_label)

		return


	def test_model(self, test_data, test_label):
		"""
		Function to test model's performance on testing data

		Args: 
			test_data (array): data for test model's performance
			test_label (array): label of testing data 
		
		Returns:
			score (int): test score of model
		"""

		score = self.model.score(test_data, test_label)
		
		return score

	def most_important_feature(self):
		"""
		Function to find the most importance feature in training data

		Args:
			None

		Returns:
			feature_name (string): name of most important feature
			feature_score (float): score of most important feature
		"""

		importances = self.model.feature_importances_
		indices = np.argsort(importances)[::-1]

		# shows the importance of each feature with descending order
		"""print("feature ranking:")
		for i in range(len(importances)):
			print("{} importance is {}".format(self.column_names[indices[i]], importances[indices[i]]))"""

		feature_name = self.column_names[indices[0]]
		feature_score = importances[indices[0]]

		return feature_name, feature_score 

	def potantial_customers(self):
		"""
		Function for estimating potential customers using trainid model and K Means Clustiring.
		First, function predicts the customers that subscribe a term deposit, then it keeps them in a list.
		Then it clusters this data in 2 clusters for clustering potential customers. Function returns potential
		customers' dataset index.

		Args: 
			None

		Returns:
			customer_list (list): ids of potertial customers
		"""

		pot_customers = []
		customer_index = []

		predictions = self.model.predict(self.X)

		for i in range(len(predictions)):
			if predictions[i] == 1:
				pot_customers.append(self.X[i])
				customer_index.append(i + 2)  # +2 because first item in dataset starts from 2


		customer_data = np.array(pot_customers)
		kmeans = KMeans(n_clusters=2, random_state=0).fit(customer_data)

		cl_pred = kmeans.predict(customer_data)
		customer_list = []

		for i in range(len(cl_pred)):
			if cl_pred[i] == 1:
				customer_list.append(customer_index[i])

		return customer_list


def start_train():
	"""
	Funtion to initialize class instance and start training.
	This function also finds avg_score of model, potential customer list, 
	and most important feature on dataset.

	Args:
		None

	Returns:
		None
	"""

	ml_solver = ML_Solution(RandomForestClassifier(n_estimators=50, max_features = 'sqrt', random_state=1))
	ml_solver.read_csv_file(file_name)
	
	print("\nmodel training...")
	avg_score = ml_solver.cross_validate()
	print("average score:", avg_score)
	print("model training is finished.")

	print("\nfinding potential customers...")
	customer_list = ml_solver.potantial_customers()

	# shows all potential customers' id (there are 1052 potential customers in the list)
	"""print("potential customers are:")
	for i in range(len(customer_list)):
		print("{}. customer's id: {}".format((i+1), customer_list[i]))"""
	print("potential customers are found.")
	
	print("\nfinding the most important feature in the dataset...")
	feature_name, feature_score = ml_solver.most_important_feature()
	print("most important feature is {}, and its importance score is %{:.2f}".format(feature_name, (feature_score * 100)))

	return


if __name__ == "__main__":
	start_train()	

	print("\ngoodbye")

