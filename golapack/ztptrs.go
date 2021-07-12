package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztptrs solves a triangular system of the form
//
//    A * X = B,  A**T * X = B,  or  A**H * X = B,
//
// where A is a triangular matrix of order N stored in packed format,
// and B is an N-by-NRHS matrix.  A check is made to verify that A is
// nonsingular.
func Ztptrs(uplo, trans, diag byte, n, nrhs *int, ap *mat.CVector, b *mat.CMatrix, ldb, info *int) {
	var nounit, upper bool
	var zero complex128
	var j, jc int
	var err error
	_ = err

	zero = (0.0 + 0.0*1i)

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
	} else if (*ldb) < max(1, *n) {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTPTRS"), -(*info))
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

	//     Solve  A * x = b,  A**T * x = b,  or  A**H * x = b.
	for j = 1; j <= (*nrhs); j++ {
		err = goblas.Ztpsv(mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), *n, ap, b.CVector(0, j-1, 1))
	}
}
