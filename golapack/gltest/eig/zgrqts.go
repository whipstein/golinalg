package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zgrqts tests ZGGRQF, which computes the GRQ factorization of an
// M-by-N matrix A and a P-by-N matrix B: A = R*Q and B = Z*T*Q.
func Zgrqts(m, p, n *int, a, af, q, r *mat.CMatrix, lda *int, taua *mat.CVector, b, bf, z, t, bwk *mat.CMatrix, ldb *int, taub, work *mat.CVector, lwork *int, rwork, result *mat.Vector) {
	var cone, crogue, czero complex128
	var anorm, bnorm, one, resid, ulp, unfl, zero float64
	var info int
	var err error
	_ = err

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	crogue = (-1.0e+10 + 0.0*1i)

	ulp = golapack.Dlamch(Precision)
	unfl = golapack.Dlamch(SafeMinimum)

	//     Copy the matrix A to the array AF.
	golapack.Zlacpy('F', m, n, a, lda, af, lda)
	golapack.Zlacpy('F', p, n, b, ldb, bf, ldb)

	anorm = math.Max(golapack.Zlange('1', m, n, a, lda, rwork), unfl)
	bnorm = math.Max(golapack.Zlange('1', p, n, b, ldb, rwork), unfl)

	//     Factorize the matrices A and B in the arrays AF and BF.
	golapack.Zggrqf(m, p, n, af, lda, taua, bf, ldb, taub, work, lwork, &info)

	//     Generate the N-by-N matrix Q
	golapack.Zlaset('F', n, n, &crogue, &crogue, q, lda)
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
	golapack.Zungrq(n, n, toPtr(min(*m, *n)), q, lda, taua, work, lwork, &info)

	//     Generate the P-by-P matrix Z
	golapack.Zlaset('F', p, p, &crogue, &crogue, z, ldb)
	if (*p) > 1 {
		golapack.Zlacpy('L', toPtr((*p)-1), n, bf.Off(1, 0), ldb, z.Off(1, 0), ldb)
	}
	golapack.Zungqr(p, p, toPtr(min(*p, *n)), z, ldb, taub, work, lwork, &info)

	//     Copy R
	golapack.Zlaset('F', m, n, &czero, &czero, r, lda)
	if (*m) <= (*n) {
		golapack.Zlacpy('U', m, m, af.Off(0, (*n)-(*m)), lda, r.Off(0, (*n)-(*m)), lda)
	} else {
		golapack.Zlacpy('F', toPtr((*m)-(*n)), n, af, lda, r, lda)
		golapack.Zlacpy('U', n, n, af.Off((*m)-(*n), 0), lda, r.Off((*m)-(*n), 0), lda)
	}

	//     Copy T
	golapack.Zlaset('F', p, n, &czero, &czero, t, ldb)
	golapack.Zlacpy('U', p, n, bf, ldb, t, ldb)

	//     Compute R - A*Q'
	err = goblas.Zgemm(NoTrans, ConjTrans, *m, *n, *n, -cone, a, q, cone, r)

	//     Compute norm( R - A*Q' ) / ( max(M,N)*norm(A)*ULP ) .
	resid = golapack.Zlange('1', m, n, r, lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, *m, *n)))/anorm)/ulp)
	} else {
		result.Set(0, zero)
	}

	//     Compute T*Q - Z'*B
	err = goblas.Zgemm(ConjTrans, NoTrans, *p, *n, *p, cone, z, b, czero, bwk)
	err = goblas.Zgemm(NoTrans, NoTrans, *p, *n, *n, cone, t, q, -cone, bwk)

	//     Compute norm( T*Q - Z'*B ) / ( max(P,N)*norm(A)*ULP ) .
	resid = golapack.Zlange('1', p, n, bwk, ldb, rwork)
	if bnorm > zero {
		result.Set(1, ((resid/float64(max(1, *p, *m)))/bnorm)/ulp)
	} else {
		result.Set(1, zero)
	}

	//     Compute I - Q*Q'
	golapack.Zlaset('F', n, n, &czero, &cone, r, lda)
	err = goblas.Zherk(Upper, NoTrans, *n, *n, -one, q, one, r)

	//     Compute norm( I - Q'*Q ) / ( N * ULP ) .
	resid = golapack.Zlanhe('1', 'U', n, r, lda, rwork)
	result.Set(2, (resid/float64(max(1, *n)))/ulp)

	//     Compute I - Z'*Z
	golapack.Zlaset('F', p, p, &czero, &cone, t, ldb)
	err = goblas.Zherk(Upper, ConjTrans, *p, *p, -one, z, one, t)

	//     Compute norm( I - Z'*Z ) / ( P*ULP ) .
	resid = golapack.Zlanhe('1', 'U', p, t, ldb, rwork)
	result.Set(3, (resid/float64(max(1, *p)))/ulp)
}
