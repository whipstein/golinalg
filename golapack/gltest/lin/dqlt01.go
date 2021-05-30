package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dqlt01 tests DGEQLF, which computes the QL factorization of an m-by-n
// matrix A, and partially tests DORGQL which forms the m-by-m
// orthogonal matrix Q.
//
// DQLT01 compares L with Q'*A, and checks that Q is orthogonal.
func Dqlt01(m, n *int, a, af, q, l *mat.Matrix, lda *int, tau, work *mat.Vector, lwork *int, rwork, result *mat.Vector) {
	var anorm, eps, one, resid, rogue, zero float64
	var info, minmn int
	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	minmn = minint(*m, *n)
	eps = golapack.Dlamch(Epsilon)

	//     Copy the matrix A to the array AF.
	golapack.Dlacpy('F', m, n, a, lda, af, lda)

	//     Factorize the matrix A in the array AF.
	*srnamt = "DGEQLF"
	golapack.Dgeqlf(m, n, af, lda, tau, work, lwork, &info)

	//     Copy details of Q
	golapack.Dlaset('F', m, m, &rogue, &rogue, q, lda)
	if (*m) >= (*n) {
		if (*n) < (*m) && (*n) > 0 {
			golapack.Dlacpy('F', toPtr((*m)-(*n)), n, af, lda, q.Off(0, (*m)-(*n)+1-1), lda)
		}
		if (*n) > 1 {
			golapack.Dlacpy('U', toPtr((*n)-1), toPtr((*n)-1), af.Off((*m)-(*n)+1-1, 1), lda, q.Off((*m)-(*n)+1-1, (*m)-(*n)+2-1), lda)
		}
	} else {
		if (*m) > 1 {
			golapack.Dlacpy('U', toPtr((*m)-1), toPtr((*m)-1), af.Off(0, (*n)-(*m)+2-1), lda, q.Off(0, 1), lda)
		}
	}

	//     Generate the m-by-m matrix Q
	*srnamt = "DORGQL"
	golapack.Dorgql(m, m, &minmn, q, lda, tau, work, lwork, &info)

	//     Copy L
	golapack.Dlaset('F', m, n, &zero, &zero, l, lda)
	if (*m) >= (*n) {
		if (*n) > 0 {
			golapack.Dlacpy('L', n, n, af.Off((*m)-(*n)+1-1, 0), lda, l.Off((*m)-(*n)+1-1, 0), lda)
		}
	} else {
		if (*n) > (*m) && (*m) > 0 {
			golapack.Dlacpy('F', m, toPtr((*n)-(*m)), af, lda, l, lda)
		}
		if (*m) > 0 {
			golapack.Dlacpy('L', m, m, af.Off(0, (*n)-(*m)+1-1), lda, l.Off(0, (*n)-(*m)+1-1), lda)
		}
	}

	//     Compute L - Q'*A
	goblas.Dgemm(mat.Trans, mat.NoTrans, m, n, m, toPtrf64(-one), q, lda, a, lda, &one, l, lda)

	//     Compute norm( L - Q'*A ) / ( M * norm(A) * EPS ) .
	anorm = golapack.Dlange('1', m, n, a, lda, rwork)
	resid = golapack.Dlange('1', m, n, l, lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(maxint(1, *m)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q'*Q
	golapack.Dlaset('F', m, m, &zero, &one, l, lda)
	goblas.Dsyrk(mat.Upper, mat.Trans, m, m, toPtrf64(-one), q, lda, &one, l, lda)

	//     Compute norm( I - Q'*Q ) / ( M * EPS ) .
	resid = golapack.Dlansy('1', 'U', m, l, lda, rwork)

	result.Set(1, (resid/float64(maxint(1, *m)))/eps)
}
