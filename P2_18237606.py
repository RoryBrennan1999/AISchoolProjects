# project2_18237606.py
# AI Project 2 CE4041
# Dr Colin Flanagan
# Rory Brennan 18237606

# Import relevant packages/libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Scan in training/test files
try:
    training_file = open("adaboost-train-21.txt", 'r')
    testing_file = open("adaboost-test-21.txt", 'r')
except IOError:
    print("Error. File (or files) not found.")
    sys.exit()

# Weak Linear Classifier as basic weak learner (polarity is +1)
class WeakLinearClassifier:
    def __init__(self):
        self.polarity = 1
        self.threshold = None
        self.alpha = None
        self.diff_vect = None

    def predict(self, X):
    
        # Perform predictions based on training data (and previous classifier)
        X = self.dot_product(X)
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        
        # Check WeakLinearClassifier polarity and compute predictions
        if self.polarity == 1:
            predictions[X < self.threshold] = -1
        else:
            predictions[X > self.threshold] = -1

        return predictions
    
    # Train model by calculating difference vector,
    # computing dot product of it and training inputs
    def fit(self, X, y, weight):
    
        # Find positive side mean
        X_pos = X[y == 1]
        w_pos = weight[y == 1]
        x_mean = np.average(X_pos[:,0],weights = w_pos)
        y_mean = np.average(X_pos[:,1],weights = w_pos)
        pos_mean = np.array([x_mean, y_mean])

        # Find negative side mean
        X_neg = X[y == -1]
        w_neg = weight[y == -1]
        x_mean = np.average(X_neg[:,0],weights = w_neg)
        y_mean = np.average(X_neg[:,1],weights = w_neg)
        neg_mean = np.array([x_mean, y_mean])
        
        # Compute difference
        dif_vect = pos_mean-neg_mean
        self.diff_vect = dif_vect
        
        # Return new weighted X
        X = self.dot_product(X)
        return X
    
    # Helper function calculate dot product 
    def dot_product(self, X):
        dot_array = np.array([])
        for i in X:
            dotted = np.dot(self.diff_vect, i)
            dot_array = np.append(dot_array,dotted)

        return dot_array


# AdaBoost Algorithm
# Pseudocode:
# Given: A set of training input X and associated training features/targets
# Initialize: The first weights distribution/list w(i) = 1/m (where m is number of training features)
# For i = 1.....N
#   Train a weak learner H using distribution w(i) -> H(i)(Xi) is an element of {-1,+1}
#   Set alpha = 0.5 * log( (1 - eps(i) ) / eps(i) ) where eps(i) is the weighted misclassification error of H(i)
#   Create a new weights distribution w(i+1):
#       Set w(i+1,j) = w(i,j) * { 1 / ( 2 * eps(i) ) if H(i)(Xi) != t(j)
#                               { 1 / ( 2 * ( 1 - eps(i) ) ) if H(i)(Xi) == t(j)
# Return strong(er) classifier H = sgn( sum from i = 1 to N for alpha(i) * H(i)(X))
class AdaBoost:
    def __init__(self, no_of_clfs=20): # Default number of classifiers = 20
        self.no_of_clfs = no_of_clfs
        self.classifiers = []

    def fit(self, X, y):
        
        # np.shape returns the corresponding array dimensions (as a tuple)
        no_of_features = X.shape[0]

        # Initialize weights to 1/N
        # np.full returns a new array of given shape and type, filled with given fill value.
        w = np.full(no_of_features, (1 / no_of_features))
        
        # Initialize classifier list
        self.classifiers = list()

        # Iterate through classifiers
        for _ in range(self.no_of_clfs):
            
            # Set each classifier as a Weak Linear classifier
            current_classifier = WeakLinearClassifier()
            
            # Train classifier and return normal
            X_normal = current_classifier.fit(X, y, w)
            
            # Set minimum error to a very high value
            min_error = float("inf")

            ## Returns unique elements of X normal to thresholds list
            thresholds = np.unique(X_normal)
            
            # greedy search to find best threshold with feature = 1
            for threshold in thresholds:
            
                # predict with polarity 1
                p = 1
                
                # Create predictions list filled with ones (1s)
                predictions = np.ones(no_of_features)
                
                # Change prediction to -1 if sample is < threshold
                predictions[X_normal < threshold] = -1
                
                # Error = sum of weights of misclassified samples
                misclassified = w[y != predictions]
                error = sum(misclassified)
                
                # Flip error if greater than 0.5 (can do this due to error being a log function)
                if error > 0.5:
                    error = 1 - error
                    p = -1

                # Store the best performing classifier split
                if error < min_error:
                    current_classifier.polarity = p
                    current_classifier.threshold = threshold
                    min_error = error
                    

            # Calculate alpha using new min error
            current_classifier.alpha = 0.5 * np.log((1.0 - min_error) / (min_error))

            # Calculate predictions and update weights
            predictions = current_classifier.predict(X)
            w *= np.exp(-current_classifier.alpha * y * predictions)
            
            # Normalize to one
            w /= np.sum(w)

            # Save classifier
            self.classifiers.append(current_classifier)

    def predict(self, X):
    
        # Compute predictions for AdaBoost
        classifier_preds = [classifier.alpha * classifier.predict(X) for classifier in self.classifiers]
        
        y_pred = np.sum(classifier_preds, axis=0)
        
        # np.sign returns an element-wise indication of the sign of a number
        y_pred = np.sign(y_pred)

        return y_pred

# Helper function to plot classification accuracy at each iteration
def plot_class_acc(X, y, classifier, N):
    
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    fig.set_facecolor('white')

    pad = 1
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    sizes = np.ones(shape=X.shape[0]) * 100

    X_pos = X[y == 1]
    sizes_pos = sizes[y == 1]
    ax.scatter(*X_pos.T, s=sizes_pos, marker='+', color='red')

    X_neg = X[y == -1]
    sizes_neg = sizes[y == -1]
    ax.scatter(*X_neg.T, s=sizes_neg, marker='.', c='blue')

    if classifier:
        plot_step = 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))
        
        C = np.c_[xx.ravel(), yy.ravel()]
        Z = classifier.predict(C)
        Z = Z.reshape(xx.shape)
        
        # If all predictions are positive class, adjust color map acordingly
        if list(np.unique(Z)) == [1]:
            fill_colors = ['r']
        else:
            fill_colors = ['b', 'r']

        ax.contourf(xx, yy, Z, colors=fill_colors, alpha=0.2)
    

    ax.set_xlim(x_min+0.5, x_max-0.5)
    ax.set_ylim(y_min+0.5, y_max-0.5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    title = "No of classifiers:" + str(N)
    plt.title(title)
    plt.show()
    

# Testing classifier(s)
if __name__ == "__main__":
    
    # Check if predictions match true values
    # and compute accuracy over total sum of targets
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    # Load in input data/targets as numpy datasets
    X_train = np.loadtxt(training_file, usecols=(0,1))
    X_test = np.loadtxt(testing_file, usecols=(0,1))  
    y_train = np.loadtxt("adaboost-train-21.txt", usecols=2)
    y_test = np.loadtxt("adaboost-test-21.txt", usecols=2)
    
    # Close input files
    training_file.close()
    testing_file.close()
    
    # Empty lists to be populated by for loop
    testing_accuracy = []
    training_accuracy = []

    # Adaboost classification building from 1 to 10 weak classifiers
    for N in range(1, 31):
    
        # Begin boosting
        strong_clf = AdaBoost(N)
        strong_clf.fit(X_train, y_train)
        y_train_pred = strong_clf.predict(X_train)
        y_pred = strong_clf.predict(X_test)
        
        # Print out classfier accuracy
        print("--- No of Classifiers: ",N, "----")
        train_acc = accuracy(y_train, y_train_pred)
        test_acc = accuracy(y_test, y_pred)
        testing_accuracy.append(test_acc)
        training_accuracy.append(train_acc)
        print("Testing Accuracy:", test_acc)
        print("Training Accuracy:", train_acc, "\n")
        
        # Draw plot every step
        if (N % 5) == 0:
            plot_class_acc(X_train, y_train, strong_clf, N)
            
    # Output figures for training and testing accuracy per number of classifiers
    plt.figure('Testing Accuracy per number of classifiers',figsize = (8,8))  
    plt.plot(training_accuracy,label = 'training')
    plt.plot(testing_accuracy,label = 'testing') 
    plt.grid(True)
    plt.legend()
    plt.xlabel('No of Classifiers')
    plt.ylabel('Testing Accuracy')
    plt.show()