package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtrti2 computes the inverse of a real upper or lower triangular
// matrix.
//
// This is the Level 2 BLAS version of the algorithm.
func Dtrti2(uplo, diag byte, n *int, a *mat.Matrix, lda *int, info *int) {
	var nounit, upper bool
	var ajj, one float64
	var j int
	var err error
	_ = err

	one = 1.0

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
		gltest.Xerbla([]byte("DTRTI2"), -(*info))
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
			err = goblas.Dtrmv(mat.Upper, mat.NoTrans, mat.DiagByte(diag), j-1, a, a.Vector(0, j-1, 1))
			goblas.Dscal(j-1, ajj, a.Vector(0, j-1, 1))
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
				err = goblas.Dtrmv(mat.Lower, mat.NoTrans, mat.DiagByte(diag), (*n)-j, a.Off(j, j), a.Vector(j, j-1, 1))
				goblas.Dscal((*n)-j, ajj, a.Vector(j, j-1, 1))
			}
		}
	}
}
