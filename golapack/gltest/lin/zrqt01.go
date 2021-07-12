package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zrqt01 tests ZGERQF, which computes the RQ factorization of an m-by-n
// matrix A, and partially tests ZUNGRQ which forms the n-by-n
// orthogonal matrix Q.
//
// ZRQT01 compares R with A*Q', and checks that Q is orthogonal.
func Zrqt01(m, n *int, a, af, q, r *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork *int, rwork, result *mat.Vector) {
	var rogue complex128
	var anorm, eps, one, resid, zero float64
	var info, minmn int
	var err error
	_ = err

	zero = 0.0
	one = 1.0
	rogue = (-1.0e+10 + (-1.0e+10)*1i)
	srnamt := &gltest.Common.Srnamc.Srnamt

	minmn = min(*m, *n)
	eps = golapack.Dlamch(Epsilon)

	//     Copy the matrix A to the array AF.
	golapack.Zlacpy('F', m, n, a, lda, af, lda)

	//     Factorize the matrix A in the array AF.
	*srnamt = "ZGERQF"
	golapack.Zgerqf(m, n, af, lda, tau, work, lwork, &info)

	//     Copy details of Q
	golapack.Zlaset('F', n, n, &rogue, &rogue, q, lda)
	if (*m) <= (*n) {
		if (*m) > 0 && (*m) < (*n) {
			golapack.Zlacpy('F', m, toPtr((*n)-(*m)), af, lda, q.Off((*n)-(*m), 0), lda)
		}
		if (*m) > 1 {
			golapack.Zlacpy('L', toPtr((*m)-1), toPtr((*m)-1), af.Off(1, (*n)-(*m)), lda, q.Off((*n)-(*m)+2-1, (*n)-(*m)), lda)
		}
	} else {
		if (*n) > 1 {
			golapack.Zlacpy('L', toPtr((*n)-1), toPtr((*n)-1), af.Off((*m)-(*n)+2-1, 0), lda, q.Off(1, 0), lda)
		}
	}

	//     Generate the n-by-n matrix Q
	*srnamt = "ZUNGRQ"
	golapack.Zungrq(n, n, &minmn, q, lda, tau, work, lwork, &info)

	//     Copy R
	golapack.Zlaset('F', m, n, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), r, lda)
	if (*m) <= (*n) {
		if (*m) > 0 {
			golapack.Zlacpy('U', m, m, af.Off(0, (*n)-(*m)), lda, r.Off(0, (*n)-(*m)), lda)
		}
	} else {
		if (*m) > (*n) && (*n) > 0 {
			golapack.Zlacpy('F', toPtr((*m)-(*n)), n, af, lda, r, lda)
		}
		if (*n) > 0 {
			golapack.Zlacpy('U', n, n, af.Off((*m)-(*n), 0), lda, r.Off((*m)-(*n), 0), lda)
		}
	}

	//     Compute R - A*Q'
	err = goblas.Zgemm(NoTrans, ConjTrans, *m, *n, *n, complex(-one, 0), a, q, complex(one, 0), r)

	//     Compute norm( R - Q'*A ) / ( N * norm(A) * EPS ) .
	anorm = golapack.Zlange('1', m, n, a, lda, rwork)
	resid = golapack.Zlange('1', m, n, r, lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, *n)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q*Q'
	golapack.Zlaset('F', n, n, toPtrc128(complex(zero, 0)), toPtrc128(complex(one, 0)), r, lda)
	err = goblas.Zherk(Upper, NoTrans, *n, *n, -one, q, one, r)

	//     Compute norm( I - Q*Q' ) / ( N * EPS ) .
	resid = golapack.Zlansy('1', 'U', n, r, lda, rwork)

	result.Set(1, (resid/float64(max(1, *n)))/eps)
}
