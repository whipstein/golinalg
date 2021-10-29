package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpttrs solves a tridiagonal system of the form
//    A * X = B
// using the factorization A = U**H *D* U or A = L*D*L**H computed by ZPTTRF.
// D is a diagonal matrix specified in the vector D, U (or L) is a unit
// bidiagonal matrix whose superdiagonal (subdiagonal) is specified in
// the vector E, and X and B are N by NRHS matrices.
func Zpttrs(uplo mat.MatUplo, n, nrhs int, d *mat.Vector, e *mat.CVector, b *mat.CMatrix) (err error) {
	var upper bool
	var iuplo, j, jb, nb int

	//     Test the input arguments.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zpttrs", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		return
	}

	//     Determine the number of right-hand sides to solve at a time.
	if nrhs == 1 {
		nb = 1
	} else {
		nb = max(1, Ilaenv(1, "Zpttrs", []byte{uplo.Byte()}, n, nrhs, -1, -1))
	}

	//     Decode UPLO
	if upper {
		iuplo = 1
	} else {
		iuplo = 0
	}

	if nb >= nrhs {
		Zptts2(iuplo, n, nrhs, d, e, b)
	} else {
		for j = 1; j <= nrhs; j += nb {
			jb = min(nrhs-j+1, nb)
			Zptts2(iuplo, n, jb, d, e, b.Off(0, j-1))
		}
	}

	return
}
