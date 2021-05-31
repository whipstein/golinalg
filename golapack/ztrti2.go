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
	} else if (*lda) < maxint(1, *n) {
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
			goblas.Ztrmv(Upper, NoTrans, mat.DiagByte(diag), toPtr(j-1), a, lda, a.CVector(0, j-1), func() *int { y := 1; return &y }())
			goblas.Zscal(toPtr(j-1), &ajj, a.CVector(0, j-1), func() *int { y := 1; return &y }())
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
				goblas.Ztrmv(Lower, NoTrans, mat.DiagByte(diag), toPtr((*n)-j), a.Off(j+1-1, j+1-1), lda, a.CVector(j+1-1, j-1), func() *int { y := 1; return &y }())
				goblas.Zscal(toPtr((*n)-j), &ajj, a.CVector(j+1-1, j-1), func() *int { y := 1; return &y }())
			}
		}
	}
}
