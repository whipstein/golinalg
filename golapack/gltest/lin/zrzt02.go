package lin

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zrzt02 returns
//      || I - Q'*Q || / ( M * eps)
// where the matrix Q is defined by the Householder transformations
// generated by ZTZRZF.
func zrzt02(m, n int, af *mat.CMatrix, tau, work *mat.CVector, lwork int) (zrzt02Return float64) {
	var one, zero float64
	var i int
	var err error

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	zrzt02Return = zero

	if lwork < n*n+n {
		err = fmt.Errorf("lwork < n*n+n: lwork=%v, n=%v", lwork, n)
		gltest.Xerbla2("zrzt02", err)
		return
	}

	//     Quick return if possible
	if m <= 0 || n <= 0 {
		return
	}

	//     Q := I
	golapack.Zlaset(Full, n, n, complex(zero, 0), complex(one, 0), work.CMatrix(n, opts))

	//     Q := P(1) * ... * P(m) * Q
	if err = golapack.Zunmrz(Left, NoTrans, n, n, m, n-m, af, tau, work.CMatrix(n, opts), work.Off(n*n), lwork-n*n); err != nil {
		panic(err)
	}

	//     Q := P(m)' * ... * P(1)' * Q
	if err = golapack.Zunmrz(Left, ConjTrans, n, n, m, n-m, af, tau, work.CMatrix(n, opts), work.Off(n*n), lwork-n*n); err != nil {
		panic(err)
	}

	//     Q := Q - I
	for i = 1; i <= n; i++ {
		work.Set((i-1)*n+i-1, work.Get((i-1)*n+i-1)-complex(one, 0))
	}

	zrzt02Return = golapack.Zlange('O', n, n, work.CMatrix(n, opts), rwork) / (golapack.Dlamch(Epsilon) * float64(max(m, n)))
	return
}
