package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlarzt forms the triangular factor T of a real block reflector
// H of order > n, which is defined as a product of k elementary
// reflectors.
//
// If DIRECT = 'F', H = H(1) H(2) . . . H(k) and T is upper triangular;
//
// If DIRECT = 'B', H = H(k) . . . H(2) H(1) and T is lower triangular.
//
// If STOREV = 'C', the vector which defines the elementary reflector
// H(i) is stored in the i-th column of the array V, and
//
//    H  =  I - V * T * V**T
//
// If STOREV = 'R', the vector which defines the elementary reflector
// H(i) is stored in the i-th row of the array V, and
//
//    H  =  I - V**T * T * V
//
// Currently, only STOREV = 'R' and DIRECT = 'B' are supported.
func Dlarzt(direct, storev byte, n, k int, v *mat.Matrix, tau *mat.Vector, t *mat.Matrix) (err error) {
	var zero float64
	var i, j int

	zero = 0.0

	//     Check for currently supported options
	if direct != 'B' {
		err = fmt.Errorf("direct != 'B': direct='%c'", direct)
	} else if storev != 'R' {
		err = fmt.Errorf("storev != 'R': storev='%c'", storev)
	}
	if err != nil {
		gltest.Xerbla2("Dlarzt", err)
		return
	}

	for i = k; i >= 1; i-- {
		if tau.Get(i-1) == zero {
			//           H(i)  =  I
			for j = i; j <= k; j++ {
				t.Set(j-1, i-1, zero)
			}
		} else {
			//           general case
			if i < k {
				//              T(i+1:k,i) = - tau(i) * V(i+1:k,1:n) * V(i,1:n)**T
				if err = goblas.Dgemv(NoTrans, k-i, n, -tau.Get(i-1), v.Off(i, 0), v.Vector(i-1, 0), zero, t.Vector(i, i-1, 1)); err != nil {
					panic(err)
				}

				//              T(i+1:k,i) = T(i+1:k,i+1:k) * T(i+1:k,i)
				if err = goblas.Dtrmv(Lower, NoTrans, NonUnit, k-i, t.Off(i, i), t.Vector(i, i-1, 1)); err != nil {
					panic(err)
				}
			}
			t.Set(i-1, i-1, tau.Get(i-1))
		}
	}

	return
}
