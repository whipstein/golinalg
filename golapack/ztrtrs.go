package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztrtrs solves a triangular system of the form
//
//    A * X = B,  A**T * X = B,  or  A**H * X = B,
//
// where A is a triangular matrix of order N, and B is an N-by-NRHS
// matrix.  A check is made to verify that A is nonsingular.
func Ztrtrs(uplo, trans, diag byte, n, nrhs *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb, info *int) {
	var nounit bool
	var one, zero complex128
	var err error
	_ = err

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Test the input parameters.
	(*info) = 0
	nounit = diag == 'N'
	if uplo != 'U' && uplo != 'L' {
		(*info) = -1
	} else if trans != 'N' && trans != 'T' && trans != 'C' {
		(*info) = -2
	} else if !nounit && diag != 'U' {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*nrhs) < 0 {
		(*info) = -5
	} else if (*lda) < max(1, *n) {
		(*info) = -7
	} else if (*ldb) < max(1, *n) {
		(*info) = -9
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTRTRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Check for singularity.
	if nounit {
		for (*info) = 1; (*info) <= (*n); (*info)++ {
			if a.Get((*info)-1, (*info)-1) == zero {
				return
			}
		}
	}
	(*info) = 0

	//     Solve A * x = b,  A**T * x = b,  or  A**H * x = b.
	err = goblas.Ztrsm(Left, mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), *n, *nrhs, one, a, b)
}
