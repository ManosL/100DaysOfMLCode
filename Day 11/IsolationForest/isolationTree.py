import numpy as np
import random
import math

euler_constant = 0.5772156649

class IsolationTree:
    class Node:
        class Test:
            def __init__(self, attr_index, val):
                self.__attr_index = attr_index
                self.__val        = val
            
            def test_pass(self, sample):
                return sample[self.__attr_index] < self.__val
            
        def __init__(self, attr_index=None, val=None, size=None, left_child=None, right_child=None):
            self.test        = self.Test(attr_index, val)
            self.size        = size
            self.left_child  = left_child
            self.right_child = right_child

    def __init__(self, X, height_limit = None):
        self.__root    = self.__create(X, height_limit)

    def __create(self, X, height_limit=None):
        assert len(X) > 0

        # Checking base cases
        if len(X) == 1:
            return self.Node(size=1) # It's an external node
        
        all_equal = True
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                if not np.array_equal(X[i], X[j]):
                    all_equal = False
                    break
            
            if not all_equal:
                break
        
        if all_equal or (height_limit == 0):
            return self.Node(size=len(X))   # It's an external node

        # Now, doing the split and afterwards the recursive call

        # Finding the splitting attribute, its max and min value
        # through the samples and the splitting value
        attributes_num   = X.shape[1]

        split_attr_index = random.randint(0, attributes_num - 1)

        attr_max_val     = X[:, split_attr_index].max()
        attr_min_val     = X[:, split_attr_index].min()

        # In case we take a feature that won't actually split the data
        while attr_max_val == attr_min_val:
            if split_attr_index == attributes_num - 1:
                split_attr_index = 0
            else:
                split_attr_index += 1

            attr_max_val     = X[:, split_attr_index].max()
            attr_min_val     = X[:, split_attr_index].min()

        # WE want to ensure that a split will happen
        split_val        = random.uniform(attr_min_val, attr_max_val)

        # Determining the samples that will go to the left and
        # right child of the node
        X_left  = X[X[:, split_attr_index] <  split_val]
        X_right = X[X[:, split_attr_index] >= split_val]

        assert (len(X_left) + len(X_right)) == len(X)
        assert len(X_left)  > 0
        assert len(X_right) > 0

        new_height_limit = None if height_limit == None else height_limit - 1

        # Doing the recursive calls
        left_child  = self.__create(X_left,  new_height_limit)
        right_child = self.__create(X_right, new_height_limit)

        return self.Node(split_attr_index, split_val, None, left_child, right_child)

    def pathLength(self, sample):
        return self.__pathLength(sample, self.__root, 0)

    def __pathLength(self, sample, curr_node, curr_height):
        if (curr_node.left_child == None) and (curr_node.right_child == None):
            if curr_node.size == 1:
                return curr_height

            c_num  = 2 * (math.log(curr_node.size - 1) + euler_constant)
            c_num -= 2 * ((curr_node.size - 1) / curr_node.size)

            return curr_height + c_num

        assert (curr_node.left_child != None) and (curr_node.right_child != None)

        if curr_node.test.test_pass(sample):
            return self.__pathLength(sample, curr_node.left_child,  curr_height + 1)
        else:
            return self.__pathLength(sample, curr_node.right_child, curr_height + 1)
