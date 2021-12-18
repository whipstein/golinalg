package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// drzt01 returns
//      || A - R*Q || / ( M * eps * ||A|| )
// for an upper trapezoidal A that was factored with DTZRZF.
func drzt01(m, n int, a, af *mat.Matrix, tau, work *mat.Vector, lwork int) (drzt01Return float64) {
	var norma, one, zero float64
	var i, j int
	var err error

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	drzt01Return = zero

	if lwork < m*n+m {
		gltest.Xerbla("drzt01", 8)
		return
	}

	//     Quick return if possible
	if m <= 0 || n <= 0 {
		return
	}

	norma = golapack.Dlange('O', m, n, a, rwork)

	//     Copy upper triangle R
	golapack.Dlaset(Full, m, n, zero, zero, work.Matrix(m, opts))
	for j = 1; j <= m; j++ {
		for i = 1; i <= j; i++ {
			work.Set((j-1)*m+i-1, af.Get(i-1, j-1))
		}
	}

	//     R = R * P(1) * ... *P(m)
	if err = golapack.Dormrz(Right, NoTrans, m, n, m, n-m, af, tau, work.Matrix(m, opts), work.Off(m*n), lwork-m*n); err != nil {
		panic(err)
	}

	//     R = R - A
	for i = 1; i <= n; i++ {
		work.Off((i-1)*m).Axpy(m, -one, a.Off(0, i-1).Vector(), 1, 1)
	}

	drzt01Return = golapack.Dlange('O', m, n, work.Matrix(m, opts), rwork)

	drzt01Return = drzt01Return / (golapack.Dlamch(Epsilon) * float64(max(m, n)))
	if norma != zero {
		drzt01Return = drzt01Return / norma
	}

	return
}
