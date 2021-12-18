package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztrti2 computes the inverse of a complex upper or lower triangular
// matrix.
//
// This is the Level 2 BLAS version of the algorithm.
func Ztrti2(uplo mat.MatUplo, diag mat.MatDiag, n int, a *mat.CMatrix) (err error) {
	var nounit, upper bool
	var ajj, one complex128
	var j int

	one = (1.0 + 0.0*1i)

	//     Test the input parameters.
	upper = uplo == Upper
	nounit = diag == NonUnit
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if !nounit && diag != Unit {
		err = fmt.Errorf("!nounit && diag != Unit: diag=%s", diag)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Ztrti2", err)
		return
	}

	if upper {
		//        Compute inverse of upper triangular matrix.
		for j = 1; j <= n; j++ {
			if nounit {
				a.Set(j-1, j-1, one/a.Get(j-1, j-1))
				ajj = -a.Get(j-1, j-1)
			} else {
				ajj = -one
			}

			//           Compute elements 1:j-1 of j-th column.
			if err = a.Off(0, j-1).CVector().Trmv(Upper, NoTrans, diag, j-1, a, 1); err != nil {
				panic(err)
			}
			a.Off(0, j-1).CVector().Scal(j-1, ajj, 1)
		}
	} else {
		//        Compute inverse of lower triangular matrix.
		for j = n; j >= 1; j-- {
			if nounit {
				a.Set(j-1, j-1, one/a.Get(j-1, j-1))
				ajj = -a.Get(j-1, j-1)
			} else {
				ajj = -one
			}
			if j < n {
				//              Compute elements j+1:n of j-th column.
				if err = a.Off(j, j-1).CVector().Trmv(Lower, NoTrans, diag, n-j, a.Off(j, j), 1); err != nil {
					panic(err)
				}
				a.Off(j, j-1).CVector().Scal(n-j, ajj, 1)
			}
		}
	}

	return
}
