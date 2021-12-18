package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlauu2 computes the product U * U**T or L**T * L, where the triangular
// factor U or L is stored in the upper or lower triangular part of
// the array A.
//
// If UPLO = 'U' or 'u' then the upper triangle of the result is stored,
// overwriting the factor U in A.
// If UPLO = 'L' or 'l' then the lower triangle of the result is stored,
// overwriting the factor L in A.
//
// This is the unblocked form of the algorithm, calling Level 2 BLAS.
func Dlauu2(uplo mat.MatUplo, n int, a *mat.Matrix) (err error) {
	var upper bool
	var aii, one float64
	var i int

	one = 1.0

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("")
	} else if n < 0 {
		err = fmt.Errorf("")
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("")
	}
	if err != nil {
		gltest.Xerbla2("Dlauu2", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if upper {
		//        Compute the product U * U**T.
		for i = 1; i <= n; i++ {
			aii = a.Get(i-1, i-1)
			if i < n {
				a.Set(i-1, i-1, a.Off(i-1, i-1).Vector().Dot(n-i+1, a.Off(i-1, i-1).Vector(), a.Rows, a.Rows))
				err = a.Off(0, i-1).Vector().Gemv(NoTrans, i-1, n-i, one, a.Off(0, i), a.Off(i-1, i).Vector(), a.Rows, aii, 1)
			} else {
				a.Off(0, i-1).Vector().Scal(i, aii, 1)
			}
		}

	} else {
		//        Compute the product L**T * L.
		for i = 1; i <= n; i++ {
			aii = a.Get(i-1, i-1)
			if i < n {
				a.Set(i-1, i-1, a.Off(i-1, i-1).Vector().Dot(n-i+1, a.Off(i-1, i-1).Vector(), 1, 1))
				err = a.Off(i-1, 0).Vector().Gemv(Trans, n-i, i-1, one, a.Off(i, 0), a.Off(i, i-1).Vector(), 1, aii, a.Rows)
			} else {
				a.Off(i-1, 0).Vector().Scal(i, aii, a.Rows)
			}
		}
	}

	return
}
