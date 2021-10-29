package golapack

import (
	"fmt"

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
func Dtptrs(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, nrhs int, ap *mat.Vector, b *mat.Matrix) (info int, err error) {
	var nounit, upper bool
	var zero float64
	var j, jc int

	zero = 0.0

	//     Test the input parameters.
	upper = uplo == Upper
	nounit = diag == NonUnit
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if trans != NoTrans && trans != Trans && trans != ConjTrans {
		err = fmt.Errorf("trans != NoTrans && trans != Trans && trans != ConjTrans: trans=%s", trans)
	} else if !nounit && diag != Unit {
		err = fmt.Errorf("!nounit && diag != Unit: diag=%s", diag)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dtptrs", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Check for singularity.
	if nounit {
		if upper {
			jc = 1
			for info = 1; info <= n; info++ {
				if ap.Get(jc+info-1-1) == zero {
					return
				}
				jc = jc + info
			}
		} else {
			jc = 1
			for info = 1; info <= n; info++ {
				if ap.Get(jc-1) == zero {
					return
				}
				jc = jc + n - info + 1
			}
		}
	}
	info = 0

	//     Solve A * x = b  or  A**T * x = b.
	for j = 1; j <= nrhs; j++ {
		err = goblas.Dtpsv(uplo, trans, diag, n, ap, b.Vector(0, j-1, 1))
	}

	return
}
