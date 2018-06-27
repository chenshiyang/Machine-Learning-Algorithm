from randomforest.ID3Tree import ID3Tree


class RandomForest(object):
    """A random forest object with the default of using a C45Tree 
    for each of the trees in the random forest. To train the forest
    create an instance of it then call train on a TraingData object"""
    def __init__(self, data, numberOfTrees=100):
        '''
        Initialize the random forest. 
        Each tree has a bag of the data associated with it.
        '''
        self.data = data    #The data that the trees will be trained on
        self.numberOfTrees = numberOfTrees
        self.forest = []

        for i in range(numberOfTrees):
            bag = data.getBag()
            self.forest.append(ID3Tree(bag))

    def train(self):
        '''
        Train the random forest trees.
        '''
        for tree in self.forest:
            tree.fit()

    def classify(self, sample):
        '''
        Classify a data sample by polling the trees.
        Using vote to get the predicted label
        '''

        #Create an empty dictionary
        votes = {}
        #Tally the votes, for each tree classify the sample
        for tree in self.forest:
            label = tree.classify(sample)
            if label in votes:
                votes[label] += 1
            else:
                votes[label] = 1

        bestLabel = None
        mostTimes = 0
        #Find the label with the most votes
        for key in votes:
            if votes[key] > mostTimes:
                bestLabel = key
                mostTimes = votes[key]

        #Return the most popular label
        return bestLabel      