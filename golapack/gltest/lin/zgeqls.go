package lin

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zgeqls Solve the least squares problem
//     min || A*X - B ||
// using the QL factorization
//     A = Q*L
// computed by ZGEQLF.
func zgeqls(m, n, nrhs int, a *mat.CMatrix, tau *mat.CVector, b *mat.CMatrix, work *mat.CVector, lwork int) (err error) {
	var one complex128

	one = (1.0 + 0.0*1i)

	//     Test the input arguments.
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 || n > m {
		err = fmt.Errorf("n < 0 || n > m: n=%v, m=%v", n, m)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if b.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows < max(1, m): b.Rows=%v, m=%v", b.Rows, m)
	} else if lwork < 1 || lwork < nrhs && m > 0 && n > 0 {
		err = fmt.Errorf("lwork < 1 || lwork < nrhs && m > 0 && n > 0: lwork=%v, nrhs=%v, m=%v, n=%v", lwork, nrhs, m, n)
	}
	if err != nil {
		gltest.Xerbla2("zgeqls", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 || m == 0 {
		return
	}

	//     B := Q' * B
	if err = golapack.Zunmql(Left, ConjTrans, m, nrhs, n, a, tau, b, work, lwork); err != nil {
		panic(err)
	}

	//     Solve L*X = B(m-n+1:m,:)
	if err = goblas.Ztrsm(Left, Lower, NoTrans, NonUnit, n, nrhs, one, a.Off(m-n, 0), b.Off(m-n, 0)); err != nil {
		panic(err)
	}

	return
}
