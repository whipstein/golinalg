package eig

import (
	"fmt"

	"github.com/whipstein/golinalg/mat"
)

// dlafts tests the result vector against the threshold value to
//    see which tests for this matrix _type failed to pass the threshold.
//    Output is to the file given by unit IOUNIT.
func dlafts(_type string, m, n, imat, ntests int, result *mat.Vector, iseed []int, thresh float64, ie int) (err error) {
	var k int

	if m == n {
		//     Output for square matrices:
		for k = 1; k <= ntests; k++ {
			if result.Get(k-1) >= thresh {
				//           If this is the first test to fail, call DLAHD2
				//           to print a header to the data file.
				if ie == 0 {
					dlahd2(_type)
				}
				err = fmt.Errorf("result.Get(k-1) >= thresh")
				ie++
				if result.Get(k-1) < 10000.0 {
					fmt.Printf(" Matrix order=%5d, _type=%2d, seed=%4d, result %3d is%8.2f\n", n, imat, iseed, k, result.Get(k-1))
				} else {
					fmt.Printf(" Matrix order=%5d, _type=%2d, seed=%4d, result %3d is%10.3E\n", n, imat, iseed, k, result.Get(k-1))
				}
			}
		}
	} else {
		//     Output for rectangular matrices
		for k = 1; k <= ntests; k++ {
			if result.Get(k-1) >= thresh {
				//              If this is the first test to fail, call DLAHD2
				//              to print a header to the data file.
				if ie == 0 {
					dlahd2(_type)
				}
				err = fmt.Errorf("result.Get(k-1) >= thresh")
				ie++
				if result.Get(k-1) < 10000.0 {
					fmt.Printf(" %5d x%5d matrix, _type=%2d, seed=%4d: result %3d is%8.2f\n", m, n, imat, iseed, k, result.Get(k-1))
				} else {
					fmt.Printf(" %5d x%5d matrix, _type=%2d, seed=%4d: result %3d is%10.3E\n", m, n, imat, iseed, k, result.Get(k-1))
				}
			}
		}

	}

	return
}
