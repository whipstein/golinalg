package lin

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zgeqrs Solve the least squares problem
//     min || A*X - B ||
// using the QR factorization
//     A = Q*R
// computed by ZGEQRF.
func zgeqrs(m, n, nrhs int, a *mat.CMatrix, tau *mat.CVector, b *mat.CMatrix, work *mat.CVector, lwork int) (err error) {
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
		gltest.Xerbla2("zgeqrs", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 || m == 0 {
		return
	}

	//     B := Q' * B
	if err = golapack.Zunmqr(Left, ConjTrans, m, nrhs, n, a, tau, b, work, lwork); err != nil {
		panic(err)
	}

	//     Solve R*X = B(1:n,:)
	if err = goblas.Ztrsm(Left, Upper, NoTrans, NonUnit, n, nrhs, one, a, b); err != nil {
		panic(err)
	}

	return
}
