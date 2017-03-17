import numpy as np

class KMeansCluster:

    # lst should be reformed like [[1.0,2.0],[3.0,4.0]]
    def __init__(self, lst=[], k=2):

        if lst != [] :
            self.setdata(lst, k)

    def setdata(self, lst, k=None):

        self.lst  = np.array(lst)
        self.n    = len(lst)
        self.dim  = len(lst[0])
        self.nd   = [0]*self.n
        self.kc   = []

        self.err  = 0.0
        self.perr = 0.0

        if k != None:
            self.k = k


        # set center of i cluster to random choosed point
        for i in range(self.k):
            self.kc.append(lst[np.random.randint(self.n)])

        # change its type into numpy array
        self.kc   = np.array(self.kc)


    def train(self, iterations):

        for ii in range(iterations):

            flag = 0

            # expectation phases
            for i in range(self.n):
                min_distance = 100000000
                for j in range(self.k):
                    ij_distance = np.linalg.norm(self.lst[i]-self.kc[j])
                    if ij_distance < min_distance:
                        min_distance = ij_distance
                        self.nd[i] = j
                        flag = 1

            if(flag==0):
                return


            # maximization phases
            self.kcount = [0]*self.k
            self.ksum   = [[0.0]*self.dim]*self.k

            for i in range(self.n):
                self.kcount[self.nd[i]] += 1
                self.ksum[self.nd[i]] += self.lst[i]

            for i in range(self.k):
                self.kc[i] = self.ksum[i] / self.kcount[i]


            # calc error
            self.err = 0.0

            for i in range(self.n):
                self.err += np.linalg.norm(self.lst[i]-self.kc[self.nd[i]])

            print("error %.6lf" % self.err)

            # check convergence

            if np.abs(self.err - self.perr) < 0.0001:
                print("Convergence")
                break
            else:
                self.perr = self.err


    def predict(self, lst):

        rlst = [0]*len(lst)

        # find the center for each item in the list
        for i, item in enumerate(lst):
            min_distance = 100000000
            for j in range(self.k):
                ij_distance = np.linalg.norm(np.array(item) - self.kc[j])
                if ij_distance < min_distance:
                    min_distance = ij_distance
                    rlst[i] = j

        return rlst