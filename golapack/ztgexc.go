package golapack

import (
	"fmt"

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
func Ztgexc(wantq, wantz bool, n int, a, b, q, z *mat.CMatrix, ifst, ilst int) (ilstOut, info int, err error) {
	var here int

	ilstOut = ilst

	//     Decode and test input arguments.
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if q.Rows < 1 || wantq && (q.Rows < max(1, n)) {
		err = fmt.Errorf("q.Rows < 1 || wantq && (q.Rows < max(1, n)): wantq=%v, q.Rows=%v, n=%v", wantq, q.Rows, n)
	} else if z.Rows < 1 || wantz && (z.Rows < max(1, n)) {
		err = fmt.Errorf("z.Rows < 1 || wantz && (z.Rows < max(1, n)): wantz=%v, z.Rows=%v, n=%v", wantz, z.Rows, n)
	} else if ifst < 1 || ifst > n {
		err = fmt.Errorf("ifst < 1 || ifst > n: ifst=%v, n=%v", ifst, n)
	} else if ilstOut < 1 || ilstOut > n {
		err = fmt.Errorf("ilst < 1 || ilst > n: ilst=%v, n=%v", ilstOut, n)
	}
	if err != nil {
		gltest.Xerbla2("Ztgexc", err)
		return
	}

	//     Quick return if possible
	if n <= 1 {
		return
	}
	if ifst == ilstOut {
		return
	}

	if ifst < ilstOut {

		here = ifst

	label10:
		;

		//        Swap with next one below
		if info = Ztgex2(wantq, wantz, n, a, b, q, z, here); info != 0 {
			ilstOut = here
			return
		}
		here = here + 1
		if here < ilstOut {
			goto label10
		}
		here = here - 1
	} else {
		here = ifst - 1

	label20:
		;

		//        Swap with next one above
		if info = Ztgex2(wantq, wantz, n, a, b, q, z, here); info != 0 {
			ilstOut = here
			return
		}
		here = here - 1
		if here >= ilstOut {
			goto label20
		}
		here = here + 1
	}
	ilstOut = here

	return
}
