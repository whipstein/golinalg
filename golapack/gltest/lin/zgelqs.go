package lin

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zgelqs Compute a minimum-norm solution
//     min || A*X - B ||
// using the LQ factorization
//     A = L*Q
// computed by ZGELQF.
func zgelqs(m, n, nrhs int, a *mat.CMatrix, tau *mat.CVector, b *mat.CMatrix, work *mat.CVector, lwork int) (err error) {
	var cone, czero complex128

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters.
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 || m > n {
		err = fmt.Errorf("n < 0 || m > n: m=%v, n=%v", m, n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if lwork < 1 || lwork < nrhs && m > 0 && n > 0 {
		err = fmt.Errorf("lwork < 1 || lwork < nrhs && m > 0 && n > 0: lwork=%v, nrhs=%v, m=%v, n=%v", lwork, nrhs, m, n)
	}
	if err != nil {
		gltest.Xerbla2("zgelqs", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 || m == 0 {
		return
	}

	//     Solve L*X = B(1:m,:)
	if err = goblas.Ztrsm(Left, Lower, NoTrans, NonUnit, m, nrhs, cone, a, b); err != nil {
		panic(err)
	}

	//     Set B(m+1:n,:) to zero
	if m < n {
		golapack.Zlaset(Full, n-m, nrhs, czero, czero, b.Off(m, 0))
	}

	//     B := Q' * B
	if err = golapack.Zunmlq(Left, ConjTrans, n, nrhs, m, a, tau, b, work, lwork); err != nil {
		panic(err)
	}

	return
}
