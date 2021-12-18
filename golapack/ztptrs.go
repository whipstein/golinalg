package golapack

import (
	"fmt"

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
func Ztptrs(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, nrhs int, ap *mat.CVector, b *mat.CMatrix) (info int, err error) {
	var nounit, upper bool
	var zero complex128
	var j, jc int

	zero = (0.0 + 0.0*1i)

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
		gltest.Xerbla2("Ztptrs", err)
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

	//     Solve  A * x = b,  A**T * x = b,  or  A**H * x = b.
	for j = 1; j <= nrhs; j++ {
		if err = b.Off(0, j-1).CVector().Tpsv(uplo, trans, diag, n, ap, 1); err != nil {
			panic(err)
		}
	}

	return
}
