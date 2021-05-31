package golapack

import (
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
func Dlarzt(direct, storev byte, n, k *int, v *mat.Matrix, ldv *int, tau *mat.Vector, t *mat.Matrix, ldt *int) {
	var zero float64
	var i, info, j int

	zero = 0.0

	//     Check for currently supported options
	info = 0
	if direct != 'B' {
		info = -1
	} else if storev != 'R' {
		info = -2
	}
	if info != 0 {
		gltest.Xerbla([]byte("DLARZT"), -info)
		return
	}

	for i = (*k); i >= 1; i-- {
		if tau.Get(i-1) == zero {
			//           H(i)  =  I
			for j = i; j <= (*k); j++ {
				t.Set(j-1, i-1, zero)
			}
		} else {
			//           general case
			if i < (*k) {
				//              T(i+1:k,i) = - tau(i) * V(i+1:k,1:n) * V(i,1:n)**T
				goblas.Dgemv(NoTrans, toPtr((*k)-i), n, toPtrf64(-tau.Get(i-1)), v.Off(i+1-1, 0), ldv, v.Vector(i-1, 0), ldv, &zero, t.Vector(i+1-1, i-1), toPtr(1))

				//              T(i+1:k,i) = T(i+1:k,i+1:k) * T(i+1:k,i)
				goblas.Dtrmv(Lower, NoTrans, NonUnit, toPtr((*k)-i), t.Off(i+1-1, i+1-1), ldt, t.Vector(i+1-1, i-1), toPtr(1))
			}
			t.Set(i-1, i-1, tau.Get(i-1))
		}
	}
}
