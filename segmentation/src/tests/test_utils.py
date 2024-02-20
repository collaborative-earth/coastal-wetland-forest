from src.utils import chain_list_tf_npiter, exclude_isolated_pixels
from collections import Counter
import numpy as np

def test_chain():
    g1 = (y for y in range(10))
    g2 = (y for y in range(12))
    x = [g1,g2]
    g = chain_list_tf_npiter(*x)
    c = Counter(g)
    for i in range(10): 
        assert c[i] == 2
    assert c[10] == 1
    assert c[11] == 1


def test_exclude_isolated_pixels():
    test_arr = np.array([[0,0,0,1],[1,1,0,1],[0,1,1,0],[0,1,1,0]])
    arr,_ = exclude_isolated_pixels(test_arr, 2)
    assert np.array_equal(arr, np.array([[0,0,0,0],[1,1,0,0],[0,1,1,0],[0,1,1,0]]))
    