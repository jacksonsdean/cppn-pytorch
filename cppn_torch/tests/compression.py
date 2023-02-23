import unittest
import matplotlib.pyplot as plt

from cppn_torch import ImageCPPN

class TestCompression(unittest.TestCase):
    def test_compression(self):
        cppn = ImageCPPN()
        before = cppn.get_image()
        cppn.compress("test.cppn")
        del cppn
        
        loaded = ImageCPPN()
        loaded.decompress("test.cppn")
        after = loaded.get_image()
        
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(before)
        ax[1].imshow(after)
        ax[0].set_title("Before")
        ax[1].set_title("After")
        plt.show()
        
        assert (before == after).all()
        
        
if __name__ == "__main__":
    unittest.main()
        
        
        
        
        







  