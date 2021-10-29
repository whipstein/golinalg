package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpttrs solves a tridiagonal system of the form
//    A * X = B
// using the L*D*L**T factorization of A computed by DPTTRF.  D is a
// diagonal matrix specified in the vector D, L is a unit bidiagonal
// matrix whose subdiagonal is specified in the vector E, and X and B
// are N by NRHS matrices.
func Dpttrs(n, nrhs int, d, e *mat.Vector, b *mat.Matrix) (err error) {
	var j, jb, nb int

	//     Test the input arguments.
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dpttrs", err)
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
		nb = max(1, Ilaenv(1, "Dpttrs", []byte{' '}, n, nrhs, -1, -1))
	}

	if nb >= nrhs {
		Dptts2(n, nrhs, d, e, b)
	} else {
		for j = 1; j <= nrhs; j += nb {
			jb = min(nrhs-j+1, nb)
			Dptts2(n, jb, d, e, b.Off(0, j-1))
		}
	}

	return
}
