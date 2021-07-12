package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztrti2 computes the inverse of a complex upper or lower triangular
// matrix.
//
// This is the Level 2 BLAS version of the algorithm.
func Ztrti2(uplo, diag byte, n *int, a *mat.CMatrix, lda, info *int) {
	var nounit, upper bool
	var ajj, one complex128
	var j int
	var err error
	_ = err

	one = (1.0 + 0.0*1i)

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
	} else if (*lda) < max(1, *n) {
		(*info) = -5
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTRTI2"), -(*info))
		return
	}

	if upper {
		//        Compute inverse of upper triangular matrix.
		for j = 1; j <= (*n); j++ {
			if nounit {
				a.Set(j-1, j-1, one/a.Get(j-1, j-1))
				ajj = -a.Get(j-1, j-1)
			} else {
				ajj = -one
			}

			//           Compute elements 1:j-1 of j-th column.
			err = goblas.Ztrmv(Upper, NoTrans, mat.DiagByte(diag), j-1, a, a.CVector(0, j-1, 1))
			goblas.Zscal(j-1, ajj, a.CVector(0, j-1, 1))
		}
	} else {
		//        Compute inverse of lower triangular matrix.
		for j = (*n); j >= 1; j-- {
			if nounit {
				a.Set(j-1, j-1, one/a.Get(j-1, j-1))
				ajj = -a.Get(j-1, j-1)
			} else {
				ajj = -one
			}
			if j < (*n) {
				//              Compute elements j+1:n of j-th column.
				err = goblas.Ztrmv(Lower, NoTrans, mat.DiagByte(diag), (*n)-j, a.Off(j, j), a.CVector(j, j-1, 1))
				goblas.Zscal((*n)-j, ajj, a.CVector(j, j-1, 1))
			}
		}
	}
}
