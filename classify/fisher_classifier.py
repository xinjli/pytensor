#/usr/bin/python
# -*- coding:utf-8 -*-
#
# Fisher's classifier (Linear Discriminant Analysis)
# 
#  Description:
#
#   Following program is to train and classify numerical data by Fisher's linear method.
#   Current version only deal with two group classification.
#   It can be easily extended to deal multiple classification

import numpy as np


class FisherClassifier:

    """
    Fisher's classifier (Linear Discriminant Analysis)

    Following program is to train and classify numerical data by Fisher's linear method.
    Current version only deal with two group classification.
    It can be easily extended to deal multiple classification.

    """

    def __init__(self, v):

        # dimension
        self.v = v

        # mean for each group
        self.m1  = np.zeros(v)
        self.m2  = np.zeros(v)
        self.m   = np.zeros(v)

        # variance in each group
        self.s1  = np.zeros([v,v])
        self.s2  = np.zeros([v,v])

        # variance covariance matrix
        self.s   = np.zeros([v,v])

        # inverse of the previous matrix
        self.sinv= np.zeros([v,v])

        # trained coefficient
        self.coef= np.zeros(v)


    # dataset should be set in the following format
    # dataset = [[[v1,v2,v3], group1],
    #            [[v1,v2,v3], group2],
    #            ...]

    def train(self, dataset):
        """
        :param dataset:
        dataset should be set in the following format
        dataset = [[[v1,v2,v3], group1],
                   [[v1,v2,v3], group2],
                   ...]

        :return:
        """

        # separate data into each group
        g1 = []
        g2 = []

        for data in dataset:
            if data[1] == 1:
                g1.append(data[0])
            else:
                g2.append(data[0])

        n1 = len(g1)
        n2 = len(g2)

        # calculate variance covariance matrix
            
        self.s1 = np.cov(np.array(g1).T)
        self.s2 = np.cov(np.array(g2).T)

        self.s  = ((n1-1)*self.s1 + (n2-1)*self.s2)*1.0/(n1+n2-2)
        self.sinv = np.linalg.pinv(self.s)

        # calculate means for each group
        
        self.m1 = np.mean(np.array(g1), axis=0)
        self.m2 = np.mean(np.array(g2), axis=0)

        self.m  = (self.m1+self.m2)/2.0

        
        # caculate coefficient
        self.coef = np.dot(self.sinv, (self.m1-self.m2))


    # dataset should be set in the following format
    # dataset = [[v1,v2,v3], [v1,v2,v3]]
    #

    def predict(self, dataset, printlog=0):

        result = []
        
        for i, data in enumerate(dataset):
            r = np.sum(self.coef*(np.array(data)-self.m))

            if(r>=0):
                result.append(1)
                if(printlog==1):
                    print(1)
            else:
                result.append(0)
                if(printlog==1):
                    print(0)
        
        return result


    # dataset should be set in the same format as training dataset
    #
    
    def test(self, dataset, printlog=0):
        
        input_dataset  = [data[0] for data in dataset]
        predict_result = self.predict(input_dataset)

        cnt_correct = 0
        cnt_wrong   = 0

        for i in range(len(predict_result)):
            if(predict_result[i] == dataset[i][1]):
                cnt_correct += 1
                if(printlog==1):
                    print("Correct")
            else:
                cnt_wrong += 1
                if(printlog==1):
                    print("Wrong")
        

        print("Result:")
        print("Correct Case: "+str(cnt_correct))
        print("Wrong   Case: "+str(cnt_wrong))


def demoFisherClassifier():
    
    # following dataset is about acceptance desicion of applicants
    # explaining variables are scores of examination and interview 
    # target variable is whether the specific applicant are successful

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
