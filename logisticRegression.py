import numpy as np

def sigmoid(x):
    return 1./(1 + np.exp(- x))

def crossEntropyLoss(targets, features, w):
    predictions = np.array([sigmoid(np.dot(w, feature)) for feature in features])
    s1 = np.sum(np.dot(targets, np.log(predictions)))
    s2 = np.sum(np.dot(np.ones(targets.shape) - targets , np.log(np.ones(predictions.shape) - predictions)))
    return - (s1 + s2)

class PolyBasis(object):
  def __init__(self, size = 5):
    self.size = size # the number of basis functions we use

  def calFeatureVect(self, x):
    feature  = [len(x)]
    [feature.append(sum(np.power(x,n))) for n in range(1, self.size + 1)]
    return np.array(feature)

  def calFeatureVects(self, data):
      features = []
      [features.append(self.calFeatureVect(data[i, :])) for i in range(data.shape[0])]
      features = np.array(features)
      return np.array(features)

class TriogonometricBasis(object):
    def __init__(self, size = 5):
        self.size = size # the number of basis functions we use
    
    def calFeatureVect(self, x):
        feature  = [len(x)]
        [feature.append(np.sign(np.sin(n * np.prod(x)))) for n in range(1, self.size + 1)]
        return np.array(feature)
    
    def calFeatureVects(self, data):
          features = []
          [features.append(self.calFeatureVect(data[i, :])) for i in range(data.shape[0])]
          features = np.array(features)
          return np.array(features)
    
class GradDescent:

    def __init__(self, lr):
        self.lr = lr

    def calGrad(self, w, targets, features):
        s = np.zeros(len(features[0]))
        for i in range(len(targets)):
            temp = sigmoid(np.dot(w, features[i])) - targets[i]
            s += temp * features[i]
        return s * 1./len(targets)
    
    def initializeWeights(self, M):
        return np.random.normal(size = M)

    def gradDes(self, targets, features, tor, max_interations):
        w = self.initializeWeights(len(features[0]))
        loss = crossEntropyLoss(targets, features, w)
        interations = 0
        while((loss > tor)): 
            grad = self.calGrad(w, targets, features)
            w = w - self.lr * grad
            loss = crossEntropyLoss(targets, features, w)
            interations+=1
            if interations > max_interations:
                break
        #end while
        return w        

class LogisticReg(object):
    def __init__(self, basis, optimizer):
        self.basis = basis
        self.optimizer = optimizer
        self.w = np.random.normal(size = basis.size)
    
    def fit(self, data, targets, tor = 1e-12, max_interations = 1000):
        features = self.basis.calFeatureVects(data)
        self.w =  optimizer.gradDes(targets, features, tor, max_interations)
        return
    
    def predict(self, tests):
        features = self.basis.calFeatureVects(tests)
        return np.array([int(round(sigmoid(np.dot(self.w, feature)))) for feature in features])


