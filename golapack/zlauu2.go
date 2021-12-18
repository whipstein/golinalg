package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlauu2 computes the product U * U**H or L**H * L, where the triangular
// factor U or L is stored in the upper or lower triangular part of
// the array A.
//
// If UPLO = 'U' or 'u' then the upper triangle of the result is stored,
// overwriting the factor U in A.
// If UPLO = 'L' or 'l' then the lower triangle of the result is stored,
// overwriting the factor L in A.
//
// This is the unblocked form of the algorithm, calling Level 2 BLAS.
func Zlauu2(uplo mat.MatUplo, n int, a *mat.CMatrix) (err error) {
	var upper bool
	var one complex128
	var aii float64
	var i int

	one = (1.0 + 0.0*1i)

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zlauu2", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if upper {
		//        Compute the product U * U**H.
		for i = 1; i <= n; i++ {
			aii = real(a.Get(i-1, i-1))
			if i < n {
				a.SetRe(i-1, i-1, aii*aii+real(a.Off(i-1, i).CVector().Dotc(n-i, a.Off(i-1, i).CVector(), a.Rows, a.Rows)))
				Zlacgv(n-i, a.Off(i-1, i).CVector(), a.Rows)
				if err = a.Off(0, i-1).CVector().Gemv(NoTrans, i-1, n-i, one, a.Off(0, i), a.Off(i-1, i).CVector(), a.Rows, complex(aii, 0), 1); err != nil {
					panic(err)
				}
				Zlacgv(n-i, a.Off(i-1, i).CVector(), a.Rows)
			} else {
				a.Off(0, i-1).CVector().Dscal(i, aii, 1)
			}
		}

	} else {
		//        Compute the product L**H * L.
		for i = 1; i <= n; i++ {
			aii = real(a.Get(i-1, i-1))
			if i < n {
				a.SetRe(i-1, i-1, aii*aii+real(a.Off(i, i-1).CVector().Dotc(n-i, a.Off(i, i-1).CVector(), 1, 1)))
				Zlacgv(i-1, a.Off(i-1, 0).CVector(), a.Rows)
				if err = a.Off(i-1, 0).CVector().Gemv(ConjTrans, n-i, i-1, one, a.Off(i, 0), a.Off(i, i-1).CVector(), 1, complex(aii, 0), a.Rows); err != nil {
					panic(err)
				}
				Zlacgv(i-1, a.Off(i-1, 0).CVector(), a.Rows)
			} else {
				a.Off(i-1, 0).CVector().Dscal(i, aii, a.Rows)
			}
		}
	}

	return
}
