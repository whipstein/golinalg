package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Drqt01 tests DGERQF, which computes the RQ factorization of an m-by-n
// matrix A, and partially tests DORGRQ which forms the n-by-n
// orthogonal matrix Q.
//
// DRQT01 compares R with A*Q', and checks that Q is orthogonal.
func Drqt01(m, n *int, a, af, q, r *mat.Matrix, lda *int, tau, work *mat.Vector, lwork *int, rwork, result *mat.Vector) {
	var anorm, eps, one, resid, rogue, zero float64
	var info, minmn int
	var err error
	_ = err

	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	minmn = min(*m, *n)
	eps = golapack.Dlamch(Epsilon)

	//     Copy the matrix A to the array AF.
	golapack.Dlacpy('F', m, n, a, lda, af, lda)

	//     Factorize the matrix A in the array AF.
	*srnamt = "DGERQF"
	golapack.Dgerqf(m, n, af, lda, tau, work, lwork, &info)

	//     Copy details of Q
	golapack.Dlaset('F', n, n, &rogue, &rogue, q, lda)
	if (*m) <= (*n) {
		if (*m) > 0 && (*m) < (*n) {
			golapack.Dlacpy('F', m, toPtr((*n)-(*m)), af, lda, q.Off((*n)-(*m), 0), lda)
		}
		if (*m) > 1 {
			golapack.Dlacpy('L', toPtr((*m)-1), toPtr((*m)-1), af.Off(1, (*n)-(*m)), lda, q.Off((*n)-(*m)+2-1, (*n)-(*m)), lda)
		}
	} else {
		if (*n) > 1 {
			golapack.Dlacpy('L', toPtr((*n)-1), toPtr((*n)-1), af.Off((*m)-(*n)+2-1, 0), lda, q.Off(1, 0), lda)
		}
	}

	//     Generate the n-by-n matrix Q
	*srnamt = "DORGRQ"
	golapack.Dorgrq(n, n, &minmn, q, lda, tau, work, lwork, &info)

	//     Copy R
	golapack.Dlaset('F', m, n, &zero, &zero, r, lda)
	if (*m) <= (*n) {
		if (*m) > 0 {
			golapack.Dlacpy('U', m, m, af.Off(0, (*n)-(*m)), lda, r.Off(0, (*n)-(*m)), lda)
		}
	} else {
		if (*m) > (*n) && (*n) > 0 {
			golapack.Dlacpy('F', toPtr((*m)-(*n)), n, af, lda, r, lda)
		}
		if (*n) > 0 {
			golapack.Dlacpy('U', n, n, af.Off((*m)-(*n), 0), lda, r.Off((*m)-(*n), 0), lda)
		}
	}

	//     Compute R - A*Q'
	err = goblas.Dgemm(mat.NoTrans, mat.Trans, *m, *n, *n, -one, a, q, one, r)

	//     Compute norm( R - Q'*A ) / ( N * norm(A) * EPS ) .
	anorm = golapack.Dlange('1', m, n, a, lda, rwork)
	resid = golapack.Dlange('1', m, n, r, lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, *n)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q*Q'
	golapack.Dlaset('F', n, n, &zero, &one, r, lda)
	err = goblas.Dsyrk(mat.Upper, mat.NoTrans, *n, *n, -one, q, one, r)

	//     Compute norm( I - Q*Q' ) / ( N * EPS ) .
	resid = golapack.Dlansy('1', 'U', n, r, lda, rwork)

	result.Set(1, (resid/float64(max(1, *n)))/eps)
}
