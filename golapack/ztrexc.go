package golapack

import (
	"fmt"
	"math/cmplx"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztrexc reorders the Schur factorization of a complex matrix
// A = Q*T*Q**H, so that the diagonal element of T with row index IFST
// is moved to row ILST.
//
// The Schur form T is reordered by a unitary similarity transformation
// Z**H*T*Z, and optionally the matrix Q of Schur vectors is updated by
// postmultplying it with Z.
func Ztrexc(compq byte, n int, t, q *mat.CMatrix, ifst, ilst int) (err error) {
	var wantq bool
	var sn, t11, t22 complex128
	var cs float64
	var k, m1, m2, m3 int

	//     Decode and test the input parameters.
	wantq = compq == 'V'
	if compq != 'N' && !wantq {
		err = fmt.Errorf("compq != 'N' && !wantq: compq='%c'", compq)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if t.Rows < max(1, n) {
		err = fmt.Errorf("t.Rows < max(1, n): t.Rows=%v, n=%v", t.Rows, n)
	} else if q.Rows < 1 || (wantq && q.Rows < max(1, n)) {
		err = fmt.Errorf("q.Rows < 1 || (wantq && q.Rows < max(1, n)): compq='%c', q.Rows=%v, n=%v", compq, q.Rows, n)
	} else if (ifst < 1 || ifst > n) && (n > 0) {
		err = fmt.Errorf("(ifst < 1 || ifst > n) && (n > 0): ifst=%v, n=%v", ifst, n)
	} else if (ilst < 1 || ilst > n) && (n > 0) {
		err = fmt.Errorf("(ilst < 1 || ilst > n) && (n > 0): ilst=%v, n=%v", ilst, n)
	}
	if err != nil {
		gltest.Xerbla2("Ztrexc", err)
		return
	}

	//     Quick return if possible
	if n <= 1 || ifst == ilst {
		return
	}

	if ifst < ilst {
		//        Move the IFST-th diagonal element forward down the diagonal.
		m1 = 0
		m2 = -1
		m3 = 1
	} else {
		//        Move the IFST-th diagonal element backward up the diagonal.
		m1 = -1
		m2 = 0
		m3 = -1
	}

	for _, k = range genIter(ifst+m1, ilst+m2, m3) {
		//        Interchange the k-th and (k+1)-th diagonal elements.
		t11 = t.Get(k-1, k-1)
		t22 = t.Get(k, k)

		//        Determine the transformation to perform the interchange.
		cs, sn, _ = Zlartg(t.Get(k-1, k), t22-t11)

		//        Apply transformation to the matrix T.
		if k+2 <= n {
			Zrot(n-k-1, t.CVector(k-1, k+2-1), t.CVector(k, k+2-1), cs, sn)
		}
		Zrot(k-1, t.CVector(0, k-1, 1), t.CVector(0, k, 1), cs, cmplx.Conj(sn))

		t.Set(k-1, k-1, t22)
		t.Set(k, k, t11)

		if wantq {
			//           Accumulate transformation in the matrix Q.
			Zrot(n, q.CVector(0, k-1, 1), q.CVector(0, k, 1), cs, cmplx.Conj(sn))
		}

	}

	return
}
