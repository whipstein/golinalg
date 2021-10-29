package lin

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dgelqs Compute a minimum-norm solution
//     min || A*X - B ||
// using the LQ factorization
//     A = L*Q
// computed by DGELQF.
func dgelqs(m, n, nrhs int, a *mat.Matrix, tau *mat.Vector, b *mat.Matrix, work *mat.Vector, lwork int) (err error) {
	var one, zero float64

	zero = 0.0
	one = 1.0

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
		gltest.Xerbla2("Dgelqs", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 || m == 0 {
		return
	}

	//     Solve L*X = B(1:m,:)
	if err = goblas.Dtrsm(Left, Lower, NoTrans, NonUnit, m, nrhs, one, a, b); err != nil {
		panic(err)
	}

	//     Set B(m+1:n,:) to zero
	if m < n {
		golapack.Dlaset(Full, n-m, nrhs, zero, zero, b.Off(m, 0))
	}

	//     B := Q' * B
	if err = golapack.Dormlq(Left, Trans, n, nrhs, m, a, tau, b, work, lwork); err != nil {
		panic(err)
	}

	return
}
