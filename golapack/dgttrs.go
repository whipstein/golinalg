package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgttrs solves one of the systems of equations
//    A*X = B  or  A**T*X = B,
// with a tridiagonal matrix A using the LU factorization computed
// by DGTTRF.
func Dgttrs(trans mat.MatTrans, n, nrhs int, dl, d, du, du2 *mat.Vector, ipiv []int, b *mat.Matrix) error {
	var notran bool
	var itrans, j, jb, nb int
	var err error

	notran = trans == NoTrans
	if !trans.IsValid() {
		err = fmt.Errorf("!trans.IsValid(): trans=%s", trans)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if b.Rows < max(n, 1) {
		err = fmt.Errorf("b.Rows < max(n, 1): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dgttrs", err)
		return err
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		return err
	}

	//     Decode TRANS
	if notran {
		itrans = 0
	} else {
		itrans = 1
	}

	//     Determine the number of right-hand sides to solve at a time.
	if nrhs == 1 {
		nb = 1
	} else {
		nb = max(1, Ilaenv(1, "Dgttrs", []byte{trans.Byte()}, n, nrhs, -1, -1))
	}

	if nb >= nrhs {
		Dgtts2(itrans, n, nrhs, dl, d, du, du2, ipiv, b)
	} else {
		for j = 1; j <= nrhs; j += nb {
			jb = min(nrhs-j+1, nb)
			Dgtts2(itrans, n, jb, dl, d, du, du2, ipiv, b.Off(0, j-1))
		}
	}

	return err
}
