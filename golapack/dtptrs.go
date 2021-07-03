package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtptrs solves a triangular system of the form
//
//    A * X = B  or  A**T * X = B,
//
// where A is a triangular matrix of order N stored in packed format,
// and B is an N-by-NRHS matrix.  A check is made to verify that A is
// nonsingular.
func Dtptrs(uplo, trans, diag byte, n, nrhs *int, ap *mat.Vector, b *mat.Matrix, ldb, info *int) {
	var nounit, upper bool
	var zero float64
	var j, jc int
	var err error
	_ = err

	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	nounit = diag == 'N'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if trans != 'N' && trans != 'T' && trans != 'C' {
		(*info) = -2
	} else if !nounit && diag != 'U' {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*nrhs) < 0 {
		(*info) = -5
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DTPTRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Check for singularity.
	if nounit {
		if upper {
			jc = 1
			for (*info) = 1; (*info) <= (*n); (*info)++ {
				if ap.Get(jc+(*info)-1-1) == zero {
					return
				}
				jc = jc + (*info)
			}
		} else {
			jc = 1
			for (*info) = 1; (*info) <= (*n); (*info)++ {
				if ap.Get(jc-1) == zero {
					return
				}
				jc = jc + (*n) - (*info) + 1
			}
		}
	}
	(*info) = 0

	//     Solve A * x = b  or  A**T * x = b.
	for j = 1; j <= (*nrhs); j++ {
		err = goblas.Dtpsv(mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), *n, ap, b.Vector(0, j-1), 1)
	}
}
