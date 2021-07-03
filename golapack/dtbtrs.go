package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtbtrs solves a triangular system of the form
//
//    A * X = B  or  A**T * X = B,
//
// where A is a triangular band matrix of order N, and B is an
// N-by NRHS matrix.  A check is made to verify that A is nonsingular.
func Dtbtrs(uplo, trans, diag byte, n, kd, nrhs *int, ab *mat.Matrix, ldab *int, b *mat.Matrix, ldb, info *int) {
	var nounit, upper bool
	var zero float64
	var j int
	var err error
	_ = err

	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	nounit = diag == 'N'
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if trans != 'N' && trans != 'T' && trans != 'C' {
		(*info) = -2
	} else if !nounit && diag != 'U' {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*kd) < 0 {
		(*info) = -5
	} else if (*nrhs) < 0 {
		(*info) = -6
	} else if (*ldab) < (*kd)+1 {
		(*info) = -8
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DTBTRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Check for singularity.
	if nounit {
		if upper {
			for (*info) = 1; (*info) <= (*n); (*info)++ {
				if ab.Get((*kd)+1-1, (*info)-1) == zero {
					return
				}
			}
		} else {
			for (*info) = 1; (*info) <= (*n); (*info)++ {
				if ab.Get(0, (*info)-1) == zero {
					return
				}
			}
		}
	}
	(*info) = 0

	//     Solve A * X = B  or  A**T * X = B.
	for j = 1; j <= (*nrhs); j++ {
		err = goblas.Dtbsv(mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), *n, *kd, ab, *ldab, b.Vector(0, j-1), 1)
	}
}
