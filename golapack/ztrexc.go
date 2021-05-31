package golapack

import (
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
func Ztrexc(compq byte, n *int, t *mat.CMatrix, ldt *int, q *mat.CMatrix, ldq, ifst, ilst, info *int) {
	var wantq bool
	var sn, t11, t22, temp complex128
	var cs float64
	var k, m1, m2, m3 int

	//     Decode and test the input parameters.
	(*info) = 0
	wantq = compq == 'V'
	if compq != 'N' && !wantq {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*ldt) < maxint(1, *n) {
		(*info) = -4
	} else if (*ldq) < 1 || (wantq && (*ldq) < maxint(1, *n)) {
		(*info) = -6
	} else if ((*ifst) < 1 || (*ifst) > (*n)) && ((*n) > 0) {
		(*info) = -7
	} else if ((*ilst) < 1 || (*ilst) > (*n)) && ((*n) > 0) {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTREXC"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) <= 1 || (*ifst) == (*ilst) {
		return
	}

	if (*ifst) < (*ilst) {
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

	for _, k = range genIter((*ifst)+m1, (*ilst)+m2, m3) {
		//        Interchange the k-th and (k+1)-th diagonal elements.
		t11 = t.Get(k-1, k-1)
		t22 = t.Get(k+1-1, k+1-1)

		//        Determine the transformation to perform the interchange.
		Zlartg(t.GetPtr(k-1, k+1-1), toPtrc128(t22-t11), &cs, &sn, &temp)

		//        Apply transformation to the matrix T.
		if k+2 <= (*n) {
			Zrot(toPtr((*n)-k-1), t.CVector(k-1, k+2-1), ldt, t.CVector(k+1-1, k+2-1), ldt, &cs, &sn)
		}
		Zrot(toPtr(k-1), t.CVector(0, k-1), func() *int { y := 1; return &y }(), t.CVector(0, k+1-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(sn)))

		t.Set(k-1, k-1, t22)
		t.Set(k+1-1, k+1-1, t11)

		if wantq {
			//           Accumulate transformation in the matrix Q.
			Zrot(n, q.CVector(0, k-1), func() *int { y := 1; return &y }(), q.CVector(0, k+1-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(sn)))
		}

	}
}
