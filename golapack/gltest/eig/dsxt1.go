package eig

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dsxt1 computes the difference between a set of eigenvalues.
// The result is returned as the function value.
//
// IJOB = 1:   Computes   max { min | D1(i)-D2(j) | }
//                         i     j
//
// IJOB = 2:   Computes   max { min | D1(i)-D2(j) | /
//                         i     j
//                              ( ABSTOL + |D1(i)|*ULP ) }
func Dsxt1(ijob *int, d1 *mat.Vector, n1 *int, d2 *mat.Vector, n2 *int, abstol, ulp, unfl *float64) (dsxt1Return float64) {
	var temp1, temp2, zero float64
	var i, j int

	zero = 0.0

	temp1 = zero

	j = 1
	for i = 1; i <= (*n1); i++ {
	label10:
		;
		if d2.Get(j-1) < d1.Get(i-1) && j < (*n2) {
			j = j + 1
			goto label10
		}
		if j == 1 {
			temp2 = math.Abs(d2.Get(j-1) - d1.Get(i-1))
			if (*ijob) == 2 {
				temp2 = temp2 / maxf64(*unfl, (*abstol)+(*ulp)*math.Abs(d1.Get(i-1)))
			}
		} else {
			temp2 = minf64(math.Abs(d2.Get(j-1)-d1.Get(i-1)), math.Abs(d1.Get(i-1)-d2.Get(j-1-1)))
			if (*ijob) == 2 {
				temp2 = temp2 / maxf64(*unfl, (*abstol)+(*ulp)*math.Abs(d1.Get(i-1)))
			}
		}
		temp1 = maxf64(temp1, temp2)
	}

	dsxt1Return = temp1
	return
}
