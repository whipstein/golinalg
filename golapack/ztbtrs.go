package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztbtrs solves a triangular system of the form
//
//    A * X = B,  A**T * X = B,  or  A**H * X = B,
//
// where A is a triangular band matrix of order N, and B is an
// N-by-NRHS matrix.  A check is made to verify that A is nonsingular.
func Ztbtrs(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, kd, nrhs int, ab, b *mat.CMatrix) (info int, err error) {
	var nounit, upper bool
	var zero complex128
	var j int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	nounit = diag == NonUnit
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if trans != NoTrans && trans != Trans && trans != ConjTrans {
		err = fmt.Errorf("trans != NoTrans && trans != Trans && trans != ConjTrans: trans=%s", trans)
	} else if !nounit && diag != Unit {
		err = fmt.Errorf("!nounit && diag != Unit: diag=%s", diag)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kd < 0 {
		err = fmt.Errorf("kd < 0: kd=%v", kd)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if ab.Rows < kd+1 {
		err = fmt.Errorf("ab.Rows < kd+1: ab.Rows=%v, kd=%v", ab.Rows, kd)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Ztbtrs", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Check for singularity.
	if nounit {
		if upper {
			for info = 1; info <= n; info++ {
				if ab.Get(kd, info-1) == zero {
					return
				}
			}
		} else {
			for info = 1; info <= n; info++ {
				if ab.Get(0, info-1) == zero {
					return
				}
			}
		}
	}
	info = 0

	//     Solve A * X = B,  A**T * X = B,  or  A**H * X = B.
	for j = 1; j <= nrhs; j++ {
		err = b.Off(0, j-1).CVector().Tbsv(uplo, trans, diag, n, kd, ab, 1)
	}

	return
}
