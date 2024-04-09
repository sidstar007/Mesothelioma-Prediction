# +-------------------------------------------------------------------+
# |  Author: Siddhant Sudesh Chalke, Roll No. 21BCS118, IIIT Dharwad  |
# +-------------------------------------------------------------------+

# Using Gradient Boosting for predicting Mesothelioma

from sklearn.metrics import accuracy_score

class DecisionTree:
    def __init__(self, max_depth = None):
        self.max_depth = max_depth
    
    def get_entropy(self, y):
        classes, counts = np.unique(y, return_counts = True)
        probablities = counts / len(y)
        entropy = -np.sum(probablities * np.log2(probablities))

        return entropy
    
    # Splitting data based on feature and threshold
    def split_data(self, x, y, feature_index, threshold):
        left = x[:, feature_index] <= threshold
        right = ~left

        return x[left], x[right], y[left], y[right]
    
    # Finding the best feature and threshold to split the data
    def get_best_split(self, x, y):
        best_gain = 0
        best_feature_index = None
        best_threshold = None
        n_samples, n_features = x.shape
        base_entropy = self.get_entropy(y)

        for feature_index in range(n_features):
            thresholds = np.unique(x[:, feature_index])
            for threshold in thresholds:
                x_left, x_right, y_left, y_right = self.split_data(x, y, feature_index, threshold)

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                left_entropy = self.get_entropy(y_left)
                right_entropy = self.get_entropy(y_right)

                information_gain = base_entropy - ((len(y_left) / n_samples) * left_entropy + (len(y_right) / n_samples) * right_entropy)

                if information_gain > best_gain:
                    best_gain = information_gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold
    
    # Recursively building the decision tree
    def build_tree(self, x, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)
        
        best_feature_index, best_threshold = self.get_best_split(x, y)
        if best_feature_index is None:
            return np.mean(y)
        
        x_left, x_right, y_left, y_right = self.split_data(x, y, best_feature_index, best_threshold)

        left_subtree = self.build_tree(x_left, y_left, depth + 1)
        right_subtree = self. build_tree(x_right, y_right, depth + 1)

        return {
            'feature_index' : best_feature_index,
            'threshold' : best_threshold,
            'left' : left_subtree,
            'right' : right_subtree
        }
    
    # Fitting the decision tree to the data
    def fit(self, x, y):
        self.tree = self.build_tree(x, y, depth = 0)

    # Predicting for a single datapoint
    def single_prediction(self, sample, tree):
        if isinstance(tree, dict):
            if sample[tree['feature_index']] <= tree['threshold']:
                return self.single_prediction(sample, tree['left'])
            else:
                return self.single_prediction(sample, tree['right'])
        else:
            return tree
        
    # Predicting class of multiple samples
    def predict(self, x):
        return np.array([self.single_prediction(sample, self.tree) for sample in x])


# Gradient Boosting
class GradientBoostingClassifer:
    def __init__(self, n_estimators = 50, learning_rate = 0.01, max_depth = 3, threshold_probability = 0.75):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.threshold_probability = threshold_probability
        self.estimators = []
    
    # Function to return all the parameters of the Gradient Boosting Classifier
    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'threshold_probability': self.threshold_probability
        }

    # Function to set parameters of the Gradient Boosting Classifier
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    # Sigmoid function to get probablity from numerical predictions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def residuals(self, y, y_pred):
        return y - self.sigmoid(y_pred)
    
    # Fitting Gradient boosting classifier to data
    def fit(self, x, y):
        n_samples = len(x)
        y_pred = np.zeros(n_samples)

        for i in range(self.n_estimators):
            residual = self.residuals(y, y_pred)
            tree = DecisionTree(max_depth = self.max_depth)
            tree.fit(x, residual)
            self.estimators.append(tree)

            y_pred = y_pred + self.learning_rate * tree.predict(x)
    
    # Predict function to classify based on given features
    def predict(self, x):
        y_pred = np.zeros(len(x))
        for tree in self.estimators:
            y_pred = y_pred + self.learning_rate * tree.predict(x)
        
        y_res = []
        for y in self.sigmoid(y_pred):
            if y >= self.threshold_probability:
                y_res.append(1)
            else:
                y_res.append(0)
        
        return y_res
    
    # Function to return accuracy score of the the Gradient Boosting Classifier
    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)