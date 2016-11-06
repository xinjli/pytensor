from classify.fisher_classifier import *

def demoFisherClassifier():

    # following dataset is about acceptance desicion of applicants
    # explaining variables are scores of examination and interview
    # target variable is whether the specific applicant are successful
    # because the data is few, we did not split it into training, dev, test data

    acceptance = [[[68,65],1],[[85,80],1],
                  [[50,95],1],[[54,70],1],
                  [[66,75],1],[[35,55],0],
                  [[56,65],0],[[25,75],0],
                  [[43,50],0],[[70,40],0]]


    # number of explaining variables are 2
    fc = FisherClassifier(2)
    fc.train(acceptance)
    fc.test(acceptance)


if __name__ == "__main__":

    demoFisherClassifier()