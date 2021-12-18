package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtrtrs solves a triangular system of the form
//
//    A * X = B  or  A**T * X = B,
//
// where A is a triangular matrix of order N, and B is an N-by-NRHS
// matrix.  A check is made to verify that A is nonsingular.
func Dtrtrs(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, nrhs int, a, b *mat.Matrix) (info int, err error) {
	var nounit bool
	var one, zero float64

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	nounit = diag == NonUnit
	if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo != Upper && uplo != Lower: uplo=%s", uplo)
	} else if trans != NoTrans && trans != Trans && trans != ConjTrans {
		err = fmt.Errorf("trans != NoTrans && trans != Trans && trans != ConjTrans: trans=%s", trans)
	} else if !nounit && diag != Unit {
		err = fmt.Errorf("!nounit && diag != Unit: diag=%s", diag)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dtrtrs", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Check for singularity.
	if nounit {
		for info = 1; info <= n; info++ {
			if a.Get(info-1, info-1) == zero {
				return
			}
		}
	}
	info = 0

	//     Solve A * x = b  or  A**T * x = b.
	if err = b.Trsm(Left, uplo, trans, diag, n, nrhs, one, a); err != nil {
		panic(err)
	}

	return
}
