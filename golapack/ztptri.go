package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztptri computes the inverse of a complex upper or lower triangular
// matrix A stored in packed format.
func Ztptri(uplo, diag byte, n *int, ap *mat.CVector, info *int) {
	var nounit, upper bool
	var ajj, one, zero complex128
	var j, jc, jclast, jj int
	var err error
	_ = err

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	nounit = diag == 'N'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if !nounit && diag != 'U' {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTPTRI"), -(*info))
		return
	}

	//     Check for singularity if non-unit.
	if nounit {
		if upper {
			jj = 0
			for (*info) = 1; (*info) <= (*n); (*info)++ {
				jj = jj + (*info)
				if ap.Get(jj-1) == zero {
					return
				}
			}
		} else {
			jj = 1
			for (*info) = 1; (*info) <= (*n); (*info)++ {
				if ap.Get(jj-1) == zero {
					return
				}
				jj = jj + (*n) - (*info) + 1
			}
		}
		(*info) = 0
	}

	if upper {
		//        Compute inverse of upper triangular matrix.
		jc = 1
		for j = 1; j <= (*n); j++ {
			if nounit {
				ap.Set(jc+j-1-1, one/ap.Get(jc+j-1-1))
				ajj = -ap.Get(jc + j - 1 - 1)
			} else {
				ajj = -one
			}

			//           Compute elements 1:j-1 of j-th column.
			err = goblas.Ztpmv(Upper, NoTrans, mat.DiagByte(diag), j-1, ap, ap.Off(jc-1, 1))
			goblas.Zscal(j-1, ajj, ap.Off(jc-1, 1))
			jc = jc + j
		}

	} else {
		//        Compute inverse of lower triangular matrix.
		jc = (*n) * ((*n) + 1) / 2
		for j = (*n); j >= 1; j-- {
			if nounit {
				ap.Set(jc-1, one/ap.Get(jc-1))
				ajj = -ap.Get(jc - 1)
			} else {
				ajj = -one
			}
			if j < (*n) {
				//              Compute elements j+1:n of j-th column.
				err = goblas.Ztpmv(Lower, NoTrans, mat.DiagByte(diag), (*n)-j, ap.Off(jclast-1), ap.Off(jc, 1))
				goblas.Zscal((*n)-j, ajj, ap.Off(jc, 1))
			}
			jclast = jc
			jc = jc - (*n) + j - 2
		}
	}
}
