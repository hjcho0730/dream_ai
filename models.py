from utils import *

def getModels(modelName):
    from sklearn.svm import SVC, LinearSVC
    from sklearn.linear_model import SGDClassifier
    
    models = {
        #SVC
        "SVC-linear" : {"class": SVC,
            "prarameters" : {
                "kernel":'linear'
            },
        },
        "SVC-rbf" : {"class": SVC,
            "prarameters" : {
                "kernel":'rbf'
            },
        },
        "SVC-poly" : {"class": SVC,
            "prarameters" : {
                "kernel":'poly'
            },
        },
        "SVC-sigmoid" : {"class": SVC,
            "prarameters" : {
                "kernel":'sigmoid'
            },
        },
        #LinearSVC
        "LinearSVC" : {"class": LinearSVC,
            "prarameters" : {},
        },
        #SGDClassifier
        "SGDClassifier" : {"class": SGDClassifier,
            "prarameters" : {
                "loss": 'hinge'
            },
        },
    }
    model = models[modelName]
    return model["class"](**model["prarameters"])
