package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dlqt01 tests DGELQF, which computes the LQ factorization of an m-by-n
// matrix A, and partially tests DORGLQ which forms the n-by-n
// orthogonal matrix Q.
//
// DLQT01 compares L with A*Q', and checks that Q is orthogonal.
func Dlqt01(m *int, n *int, a, af, q, l *mat.Matrix, lda *int, tau, work *mat.Vector, lwork *int, rwork, result *mat.Vector) {
	var anorm, eps, one, resid, rogue, zero float64
	var info, minmn int

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	srnamt := &gltest.Common.Srnamc.Srnamt

	minmn = minint(*m, *n)
	eps = golapack.Dlamch(Epsilon)

	//     Copy the matrix A to the array AF.
	golapack.Dlacpy('F', m, n, a, lda, af, lda)

	//     Factorize the matrix A in the array AF.
	*srnamt = "DGELQF"
	golapack.Dgelqf(m, n, af, lda, tau, work, lwork, &info)

	//     Copy details of Q
	golapack.Dlaset('F', n, n, &rogue, &rogue, q, lda)
	if (*n) > 1 {
		golapack.Dlacpy('U', m, toPtr((*n)-1), af.Off(0, 1), lda, q.Off(0, 1), lda)
	}

	//     Generate the n-by-n matrix Q
	*srnamt = "DORGLQ"
	golapack.Dorglq(n, n, &minmn, q, lda, tau, work, lwork, &info)

	//     Copy L
	golapack.Dlaset('F', m, n, &zero, &zero, l, lda)
	golapack.Dlacpy('L', m, n, af, lda, l, lda)

	//     Compute L - A*Q'
	goblas.Dgemm(NoTrans, Trans, m, n, n, toPtrf64(-one), a, lda, q, lda, &one, l, lda)

	//     Compute norm( L - Q'*A ) / ( N * norm(A) * EPS ) .
	anorm = golapack.Dlange('1', m, n, a, lda, rwork)
	resid = golapack.Dlange('1', m, n, l, lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(maxint(1, *n)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q*Q'
	golapack.Dlaset('F', n, n, &zero, &one, l, lda)
	goblas.Dsyrk(Upper, NoTrans, n, n, toPtrf64(-one), q, lda, &one, l, lda)

	//     Compute norm( I - Q*Q' ) / ( N * EPS ) .
	resid = golapack.Dlansy('1', 'U', n, l, lda, rwork)

	result.Set(1, (resid/float64(maxint(1, *n)))/eps)
}