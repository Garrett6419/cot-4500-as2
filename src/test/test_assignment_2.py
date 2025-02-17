import unittest
import sys
import os

# Adjust the path to point to the 'main' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../main')))

from assignment_2 import (
    neville_method,
    newtons_forward,
    approximate_f_using_newton,
    divided_difference,
    cubic_spline
)

class TestAssignment2(unittest.TestCase):

    def test_neville_method(self):
        # Test Neville's method for polynomial interpolation
        x_values = [3.6, 3.8, 3.9]
        y_values = [1.675, 1.436, 1.318]
        result = neville_method(x_values, y_values, 3.7)
        self.assertAlmostEqual(result, 1.5549999999999995, places=5)  # Answer

    def test_newtons_forward(self):
        # Test Newton's forward method for polynomial approximation
        x_values = [7.2, 7.4, 7.5, 7.6]
        y_values = [23.5492, 25.3913, 26.8224, 27.4589]
        table = newtons_forward(x_values, y_values)
        self.assertEqual(table.shape, (4, 4))  # Just making sure the table is the right size

    def test_approximate_f_using_newton(self):
        # Test approximation of f(7.3) using Newton's forward method
        x_values = [7.2, 7.4, 7.5, 7.6]
        y_values = [23.5492, 25.3913, 26.8224, 27.4589]
        table = newtons_forward(x_values, y_values)
        result = approximate_f_using_newton(table, x_values, 7.3)
        self.assertAlmostEqual(result, 24.497649999999997, places=5)  # Answer

    def test_divided_difference(self):
        # Test divided difference method for Hermite polynomial approximation
        x_values = [3.6, 3.8, 3.9]
        y_values = [1.675, 1.436, 1.318]
        derivatives = [-1.195, -1.188, -1.182]
        table = divided_difference(x_values, y_values, derivatives)
        self.assertEqual(table.shape, (3, 3))  # same table size check

    def test_cubic_spline(self):
        # Test cubic spline interpolation
        x_values = [2, 5, 8, 10]  
        y_values = [3, 5, 7, 9]  
        A, b, c = cubic_spline(x_values, y_values)
        self.assertEqual(A.shape, (4, 4))  # Make sure all these are the correct size
        self.assertEqual(len(b), 4)  
        self.assertEqual(len(c), 4)  

    def test_cubic_spline_invalid(self):
        # Test cubic spline with invalid input (duplicate x-values)
        x_values = [2, 3, 5, 5, 8, 10]  # Duplicate x-value
        y_values = [3, 5, 5, 7, 9, 10]
        with self.assertRaises(ValueError):
            cubic_spline(x_values, y_values)

if __name__ == '__main__':
    unittest.main()