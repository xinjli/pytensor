import numpy as np
import logging



class Node:
    def __init__(self, name, pos_cnt, neg_cnt):
        self.name = name
        self.children = []

        self.all_cnt = pos_cnt + neg_cnt

        self.pos_cnt = pos_cnt
        self.neg_cnt = neg_cnt

        self.pos_prob = pos_cnt * 1.0 / self.all_cnt
        self.neg_prob = neg_cnt * 1.0 / self.all_cnt

    def entropy(self):
        return -self.pos_prob * np.log(self.pos_prob) - self.neg_prob * np.log(self.neg_prob)


def merge_node(fst_node, snd_node):

    node_name = "("+fst_node.name + ", " +snd_node.name+")"
    new_node = Node(node_name, fst_node.pos_cnt + snd_node.pos_cnt, fst_node.neg_cnt+snd_node.neg_cnt)
    new_node.children = [fst_node, snd_node]

    return new_node


class TreeCluster:

    def __init__(self):

        self.node_lst = []

    def information_loss(self, fst_node, snd_node):
        """
        Compute information loss

        :param fst_node:
        :param snd_node:
        :return:
        """
        merged_node = merge_node(fst_node, snd_node)
        loss = merged_node.all_cnt*merged_node.entropy() - fst_node.all_cnt*fst_node.entropy() - snd_node.all_cnt*snd_node.entropy()
        return loss


    def train(self, node_lst):

        self.node_lst = node_lst.copy()

        step = 1

        while(len(self.node_lst) > 1):

            least_loss = 1000000000000.0
            least_fst_node = None
            least_snd_node = None

            for fst_node in self.node_lst:
                for snd_node in self.node_lst:

                    if fst_node == snd_node:
                        continue

                    loss = self.information_loss(fst_node, snd_node)

                    if loss < least_loss:
                        least_fst_node = fst_node
                        least_snd_node = snd_node
                        least_loss = loss


            new_node = merge_node(least_fst_node, least_snd_node)

            self.node_lst.remove(least_fst_node)
            self.node_lst.remove(least_snd_node)
            self.node_lst.append(new_node)

            print("Step " + str(step) + ": Merge "+least_fst_node.name + " and "+least_snd_node.name)
            print("Step " + str(step) + ": Information Loss "+str(least_loss))
            print("Step " + str(step) + ": Distribution ["+str(new_node.pos_prob)+", "+str(new_node.neg_prob))

            step += 1



if __name__ == '__main__':

    node_lst = []
    values = [[4, 12],
              [1, 7],
              [12, 20],
              [5, 3],
              [14, 2],
              [3, 1]]

    for i, val in enumerate(values):

        name = "T"+str(i+1)
        node = Node(name, val[0], val[1])

        node_lst.append(node)


    cluster = TreeCluster()

    cluster.train(node_lst)