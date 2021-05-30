package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dtrtrs solves a triangular system of the form
//
//    A * X = B  or  A**T * X = B,
//
// where A is a triangular matrix of order N, and B is an N-by-NRHS
// matrix.  A check is made to verify that A is nonsingular.
func Dtrtrs(uplo, trans, diag byte, n, nrhs *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb, info *int) {
	var nounit bool
	var one, zero float64

	zero = 0.0
	one = 1.0

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
	} else if (*lda) < maxint(1, *n) {
		(*info) = -7
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -9
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DTRTRS"), -(*info))
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

	//     Solve A * x = b  or  A**T * x = b.
	goblas.Dtrsm(Left, mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), n, nrhs, &one, a, lda, b, ldb)
}