package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgttrs solves one of the systems of equations
//    A * X = B,  A**T * X = B,  or  A**H * X = B,
// with a tridiagonal matrix A using the LU factorization computed
// by ZGTTRF.
func Zgttrs(trans mat.MatTrans, n, nrhs int, dl, d, du, du2 *mat.CVector, ipiv *[]int, b *mat.CMatrix) (err error) {
	var notran bool
	var itrans, j, jb, nb int

	notran = trans == NoTrans
	if !notran && trans != Trans && trans != ConjTrans {
		err = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=%s", trans)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if b.Rows < max(n, 1) {
		err = fmt.Errorf("b.Rows < max(n, 1): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zgttrs", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		return
	}

	//     Decode TRANS
	if notran {
		itrans = 0
	} else if trans == Trans {
		itrans = 1
	} else {
		itrans = 2
	}

	//     Determine the number of right-hand sides to solve at a time.
	if nrhs == 1 {
		nb = 1
	} else {
		nb = max(1, Ilaenv(1, "Zgttrs", []byte{trans.Byte()}, n, nrhs, -1, -1))
	}

	if nb >= nrhs {
		Zgtts2(itrans, n, nrhs, dl, d, du, du2, ipiv, b)
	} else {
		for j = 1; j <= nrhs; j += nb {
			jb = min(nrhs-j+1, nb)
			Zgtts2(itrans, n, jb, dl, d, du, du2, ipiv, b.Off(0, j-1))
		}
	}

	return
}
