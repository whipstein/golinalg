package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zlaipd sets the imaginary part of the diagonal elements of a complex
// matrix A to a large value.  This is used to test LAPACK routines for
// complex Hermitian matrices, which are not supposed to access or use
// the imaginary parts of the diagonals.
func zlaipd(n int, a *mat.CVector, inda, vinda int) {
	var bignum float64
	var i, ia, ixa int

	bignum = golapack.Dlamch(Epsilon) / golapack.Dlamch(SafeMinimum)
	ia = 1
	ixa = inda
	for i = 1; i <= n; i++ {
		a.Set(ia-1, complex(a.GetRe(ia-1), bignum))
		ia = ia + ixa
		ixa = ixa + vinda
	}
}
