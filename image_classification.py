import problem_unittests as tests
import numpy as np

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    return x/255

tests.test_normalize(normalize)


def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    x = np.array(x)
    number_of_categories = 10
    # number_of_categories = x.max()+1
    a = np.zeros((x.shape[0], number_of_categories), dtype=np.bool)
    a[np.arange(x.shape[0]), x] = True
    return a

tests.test_one_hot_encode(one_hot_encode)