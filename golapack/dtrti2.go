package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtrti2 computes the inverse of a real upper or lower triangular
// matrix.
//
// This is the Level 2 BLAS version of the algorithm.
func Dtrti2(uplo mat.MatUplo, diag mat.MatDiag, n int, a *mat.Matrix) (err error) {
	var nounit, upper bool
	var ajj, one float64
	var j int

	one = 1.0

	//     Test the input parameters.
	upper = uplo == Upper
	nounit = diag == NonUnit
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if !diag.IsValid() {
		err = fmt.Errorf("!diag.IsValid(): diag=%s", diag)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dtrti2", err)
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
			err = a.Off(0, j-1).Vector().Trmv(Upper, NoTrans, diag, j-1, a, 1)
			a.Off(0, j-1).Vector().Scal(j-1, ajj, 1)
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
				err = a.Off(j, j-1).Vector().Trmv(Lower, NoTrans, diag, n-j, a.Off(j, j), 1)
				a.Off(j, j-1).Vector().Scal(n-j, ajj, 1)
			}
		}
	}

	return
}
