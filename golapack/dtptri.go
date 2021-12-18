package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtptri computes the inverse of a real upper or lower triangular
// matrix A stored in packed format.
func Dtptri(uplo mat.MatUplo, diag mat.MatDiag, n int, ap *mat.Vector) (info int, err error) {
	var nounit, upper bool
	var ajj, one, zero float64
	var j, jc, jclast, jj int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	upper = uplo == Upper
	nounit = diag == NonUnit
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if !nounit && diag != Unit {
		err = fmt.Errorf("!nounit && diag != Unit: diag=%s", diag)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	}
	if err != nil {
		gltest.Xerbla2("Dtptri", err)
		return
	}

	//     Check for singularity if non-unit.
	if nounit {
		if upper {
			jj = 0
			for info = 1; info <= n; info++ {
				jj = jj + info
				if ap.Get(jj-1) == zero {
					return
				}
			}
		} else {
			jj = 1
			for info = 1; info <= n; info++ {
				if ap.Get(jj-1) == zero {
					return
				}
				jj = jj + n - info + 1
			}
		}
		info = 0
	}

	if upper {
		//        Compute inverse of upper triangular matrix.
		jc = 1
		for j = 1; j <= n; j++ {
			if nounit {
				ap.Set(jc+j-1-1, one/ap.Get(jc+j-1-1))
				ajj = -ap.Get(jc + j - 1 - 1)
			} else {
				ajj = -one
			}

			//           Compute elements 1:j-1 of j-th column.
			err = ap.Off(jc-1).Tpmv(Upper, NoTrans, diag, j-1, ap, 1)
			ap.Off(jc-1).Scal(j-1, ajj, 1)
			jc = jc + j
		}

	} else {
		//        Compute inverse of lower triangular matrix.
		jc = n * (n + 1) / 2
		for j = n; j >= 1; j-- {
			if nounit {
				ap.Set(jc-1, one/ap.Get(jc-1))
				ajj = -ap.Get(jc - 1)
			} else {
				ajj = -one
			}
			if j < n {
				//              Compute elements j+1:n of j-th column.
				err = ap.Off(jc).Tpmv(Lower, NoTrans, diag, n-j, ap.Off(jclast-1), 1)
				ap.Off(jc).Scal(n-j, ajj, 1)
			}
			jclast = jc
			jc = jc - n + j - 2
		}
	}

	return
}
