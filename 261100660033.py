import numpy as np
import unittest

def pair_score(x,z,w):
    if not isinstance(w,(tuple,list)) or len(w)!= 4:
        raise ValueError("w must contain 4 elements and should be a tuple")
    
    A,a,B,b=w
    
    dx=6
    dz=10
    dh=4
    
    if not isinstance(x,np.ndarray) or x.ndim!=2 or x.shape[0]!=dx or x.shape[1]==0:
        raise ValueError("x must be a 6xN matrix")
    
    if not isinstance(z,np.ndarray) or z.ndim!=2 or z.shape[0]!=dz or z.shape[1]==0:
        raise ValueError("z must be a 10xN matrix")
    
    if  A.shape!=(dh,dx) or a.shape!=(dh,) or B.shape!=(dh,dz) or b.shape!=(dh,):
        raise ValueError("Invalid dimensions")

    P = A @ x + a.reshape(dh,1)
    R = B @ z + b.reshape(dh,1)
    
    S= (R.T @ P) / (np.sqrt(dh))
    
    return S
    
    

class TestPairScore(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(6, 5)
        self.z = np.random.rand(10, 5)
        self.w = (np.random.rand(4, 6), np.random.rand(4), np.random.rand(4, 10), np.random.rand(4))
        
        
    def test_output(self):
        S = pair_score(self.x, self.z, self.w)
        self.assertEqual(S.shape, (5, 5))
        
    def test_invalid_w(self):
        with self.assertRaises(ValueError):
            pair_score(self.x, self.z, (np.random.rand(4, 6), np.random.rand(4), np.random.rand(4, 10)))
            
    def test_invalid_x(self):
        with self.assertRaises(ValueError):
            pair_score(np.random.rand(5, 5), self.z, self.w)
            
    def test_invalid_z(self):
        with self.assertRaises(ValueError):
            pair_score(self.x, np.random.rand(9, 5), self.w)
            
    
    def test_bias_change(self):
        S1 = pair_score(self.x, self.z, self.w)

        A, a, B, b = self.w
        delta = np.array([1., 0., 0., 0.])

        new_w = (A, a, B, b + delta)
        S2 = pair_score(self.x, self.z, new_w)

        P = A @ self.x + a.reshape(4, 1)

        expected = (
            np.ones((5, 1)) @ (delta.reshape(1, 4) @ P)
        ) / np.sqrt(4)

        np.testing.assert_allclose(S2 - S1, expected)

            
if __name__ == '__main__':
    unittest.main()
        
        
    
    
    

   