package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpteqr computes all eigenvalues and, optionally, eigenvectors of a
// symmetric positive definite tridiagonal matrix by first factoring the
// matrix using DPTTRF, and then calling DBDSQR to compute the singular
// values of the bidiagonal factor.
//
// This routine computes the eigenvalues of the positive definite
// tridiagonal matrix to high relative accuracy.  This means that if the
// eigenvalues range over many orders of magnitude in size, then the
// small eigenvalues and corresponding eigenvectors will be computed
// more accurately than, for example, with the standard QR method.
//
// The eigenvectors of a full or band symmetric positive definite matrix
// can also be found if DSYTRD, DSPTRD, or DSBTRD has been used to
// reduce this matrix to tridiagonal form. (The reduction to tridiagonal
// form, however, may preclude the possibility of obtaining high
// relative accuracy in the small eigenvalues of the original matrix, if
// these eigenvalues range over many orders of magnitude.)
func Dpteqr(compz byte, n int, d, e *mat.Vector, z *mat.Matrix, work *mat.Vector) (info int, err error) {
	var one, zero float64
	var i, icompz, nru int

	c := mf(1, 1, opts)
	vt := mf(1, 1, opts)

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	if compz == 'N' {
		icompz = 0
	} else if compz == 'V' {
		icompz = 1
	} else if compz == 'I' {
		icompz = 2
	} else {
		icompz = -1
	}
	if icompz < 0 {
		err = fmt.Errorf("icompz < 0: compz='%c'", compz)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if (z.Rows < 1) || (icompz > 0 && z.Rows < max(1, n)) {
		err = fmt.Errorf("(z.Rows < 1) || (icompz > 0 && z.Rows < max(1, n)): compz='%c', z.Rows=%v, n=%v", compz, z.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dpteqr", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if n == 1 {
		if icompz > 0 {
			z.Set(0, 0, one)
		}
		return
	}
	if icompz == 2 {
		Dlaset(Full, n, n, zero, one, z)
	}

	//     Call DPTTRF to factor the matrix.
	if info, err = Dpttrf(n, d, e); err != nil {
		panic(err)
	}
	if info != 0 {
		return
	}
	for i = 1; i <= n; i++ {
		d.Set(i-1, math.Sqrt(d.Get(i-1)))
	}
	for i = 1; i <= n-1; i++ {
		e.Set(i-1, e.Get(i-1)*d.Get(i-1))
	}

	//     Call DBDSQR to compute the singular values/vectors of the
	//     bidiagonal factor.
	if icompz > 0 {
		nru = n
	} else {
		nru = 0
	}
	if info, err = Dbdsqr(Lower, n, 0, nru, 0, d, e, vt, z, c, work); err != nil {
		panic(err)
	}

	//     Square the singular values.
	if info == 0 {
		for i = 1; i <= n; i++ {
			d.Set(i-1, d.Get(i-1)*d.Get(i-1))
		}
	} else {
		info = n + info
	}

	return
}
