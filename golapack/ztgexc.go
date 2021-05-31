package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztgexc reorders the generalized Schur decomposition of a complex
// matrix pair (A,B), using an unitary equivalence transformation
// (A, B) := Q * (A, B) * Z**H, so that the diagonal block of (A, B) with
// row index IFST is moved to row ILST.
//
// (A, B) must be in generalized Schur canonical form, that is, A and
// B are both upper triangular.
//
// Optionally, the matrices Q and Z of generalized Schur vectors are
// updated.
//
//        Q(in) * A(in) * Z(in)**H = Q(out) * A(out) * Z(out)**H
//        Q(in) * B(in) * Z(in)**H = Q(out) * B(out) * Z(out)**H
func Ztgexc(wantq, wantz bool, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, q *mat.CMatrix, ldq *int, z *mat.CMatrix, ldz, ifst, ilst, info *int) {
	var here int

	//     Decode and test input arguments.
	(*info) = 0
	if (*n) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -7
	} else if (*ldq) < 1 || wantq && ((*ldq) < maxint(1, *n)) {
		(*info) = -9
	} else if (*ldz) < 1 || wantz && ((*ldz) < maxint(1, *n)) {
		(*info) = -11
	} else if (*ifst) < 1 || (*ifst) > (*n) {
		(*info) = -12
	} else if (*ilst) < 1 || (*ilst) > (*n) {
		(*info) = -13
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTGEXC"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) <= 1 {
		return
	}
	if (*ifst) == (*ilst) {
		return
	}

	if (*ifst) < (*ilst) {

		here = (*ifst)

	label10:
		;

		//        Swap with next one below
		Ztgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, &here, info)
		if (*info) != 0 {
			(*ilst) = here
			return
		}
		here = here + 1
		if here < (*ilst) {
			goto label10
		}
		here = here - 1
	} else {
		here = (*ifst) - 1

	label20:
		;

		//        Swap with next one above
		Ztgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, &here, info)
		if (*info) != 0 {
			(*ilst) = here
			return
		}
		here = here - 1
		if here >= (*ilst) {
			goto label20
		}
		here = here + 1
	}
	(*ilst) = here
}
