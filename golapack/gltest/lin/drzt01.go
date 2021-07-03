package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Drzt01 returns
//      || A - R*Q || / ( M * eps * ||A|| )
// for an upper trapezoidal A that was factored with DTZRZF.
func Drzt01(m, n *int, a, af *mat.Matrix, lda *int, tau, work *mat.Vector, lwork *int) (drzt01Return float64) {
	var norma, one, zero float64
	var i, info, j int

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	drzt01Return = zero

	if (*lwork) < (*m)*(*n)+(*m) {
		gltest.Xerbla([]byte("DRZT01"), 8)
		return
	}

	//     Quick return if possible
	if (*m) <= 0 || (*n) <= 0 {
		return
	}

	norma = golapack.Dlange('O', m, n, a, lda, rwork)

	//     Copy upper triangle R
	golapack.Dlaset('F', m, n, &zero, &zero, work.Matrix(*m, opts), m)
	for j = 1; j <= (*m); j++ {
		for i = 1; i <= j; i++ {
			work.Set((j-1)*(*m)+i-1, af.Get(i-1, j-1))
		}
	}

	//     R = R * P(1) * ... *P(m)
	golapack.Dormrz('R', 'N', m, n, m, toPtr((*n)-(*m)), af, lda, tau, work.Matrix(*m, opts), m, work.Off((*m)*(*n)+1-1), toPtr((*lwork)-(*m)*(*n)), &info)

	//     R = R - A
	for i = 1; i <= (*n); i++ {
		goblas.Daxpy(*m, -one, a.Vector(0, i-1), 1, work.Off((i-1)*(*m)+1-1), 1)
	}

	drzt01Return = golapack.Dlange('O', m, n, work.Matrix(*m, opts), m, rwork)

	drzt01Return = drzt01Return / (golapack.Dlamch(Epsilon) * float64(maxint(*m, *n)))
	if norma != zero {
		drzt01Return = drzt01Return / norma
	}

	return
}
