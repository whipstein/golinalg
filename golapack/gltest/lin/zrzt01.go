package lin

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zrzt01 returns
//      || A - R*Q || / ( M * eps * ||A|| )
// for an upper trapezoidal A that was factored with ZTZRZF.
func zrzt01(m, n int, a, af *mat.CMatrix, tau, work *mat.CVector, lwork int) (zrzt01Return float64) {
	var norma, one, zero float64
	var i, j int
	var err error

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	zrzt01Return = zero

	if lwork < m*n+m {
		err = fmt.Errorf("lwork < m*n+m: lwork=%v, m=%v, n=%v", lwork, m, n)
		gltest.Xerbla2("zrzt01", err)
		return
	}

	//     Quick return if possible
	if m <= 0 || n <= 0 {
		return
	}

	norma = golapack.Zlange('O', m, n, a, rwork)

	//     Copy upper triangle R
	golapack.Zlaset(Full, m, n, complex(zero, 0), complex(zero, 0), work.CMatrix(m, opts))
	for j = 1; j <= m; j++ {
		for i = 1; i <= j; i++ {
			work.Set((j-1)*m+i-1, af.Get(i-1, j-1))
		}
	}

	//     R = R * P(1) * ... *P(m)
	if err = golapack.Zunmrz(Right, NoTrans, m, n, m, n-m, af, tau, work.CMatrix(m, opts), work.Off(m*n), lwork-m*n); err != nil {
		panic(err)
	}

	//     R = R - A
	for i = 1; i <= n; i++ {
		work.Off((i-1)*m).Axpy(m, complex(-one, 0), a.Off(0, i-1).CVector(), 1, 1)
	}

	zrzt01Return = golapack.Zlange('O', m, n, work.CMatrix(m, opts), rwork)

	zrzt01Return = zrzt01Return / (golapack.Dlamch(Epsilon) * float64(max(m, n)))
	if norma != zero {
		zrzt01Return = zrzt01Return / norma
	}

	return
}
