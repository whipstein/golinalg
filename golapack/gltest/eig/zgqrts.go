package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zgqrts tests ZGGQRF, which computes the GQR factorization of an
// N-by-M matrix A and a N-by-P matrix B: A = Q*R and B = Q*T*Z.
func Zgqrts(n, m, p *int, a, af, q, r *mat.CMatrix, lda *int, taua *mat.CVector, b, bf, z, t, bwk *mat.CMatrix, ldb *int, taub, work *mat.CVector, lwork *int, rwork, result *mat.Vector) {
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
	golapack.Zlacpy('F', n, m, a, lda, af, lda)
	golapack.Zlacpy('F', n, p, b, ldb, bf, ldb)

	anorm = math.Max(golapack.Zlange('1', n, m, a, lda, rwork), unfl)
	bnorm = math.Max(golapack.Zlange('1', n, p, b, ldb, rwork), unfl)

	//     Factorize the matrices A and B in the arrays AF and BF.
	golapack.Zggqrf(n, m, p, af, lda, taua, bf, ldb, taub, work, lwork, &info)

	//     Generate the N-by-N matrix Q
	golapack.Zlaset('F', n, n, &crogue, &crogue, q, lda)
	golapack.Zlacpy('L', toPtr((*n)-1), m, af.Off(1, 0), lda, q.Off(1, 0), lda)
	golapack.Zungqr(n, n, toPtr(min(*n, *m)), q, lda, taua, work, lwork, &info)

	//     Generate the P-by-P matrix Z
	golapack.Zlaset('F', p, p, &crogue, &crogue, z, ldb)
	if (*n) <= (*p) {
		if (*n) > 0 && (*n) < (*p) {
			golapack.Zlacpy('F', n, toPtr((*p)-(*n)), bf, ldb, z.Off((*p)-(*n), 0), ldb)
		}
		if (*n) > 1 {
			golapack.Zlacpy('L', toPtr((*n)-1), toPtr((*n)-1), bf.Off(1, (*p)-(*n)), ldb, z.Off((*p)-(*n)+2-1, (*p)-(*n)), ldb)
		}
	} else {
		if (*p) > 1 {
			golapack.Zlacpy('L', toPtr((*p)-1), toPtr((*p)-1), bf.Off((*n)-(*p)+2-1, 0), ldb, z.Off(1, 0), ldb)
		}
	}
	golapack.Zungrq(p, p, toPtr(min(*n, *p)), z, ldb, taub, work, lwork, &info)

	//     Copy R
	golapack.Zlaset('F', n, m, &czero, &czero, r, lda)
	golapack.Zlacpy('U', n, m, af, lda, r, lda)

	//     Copy T
	golapack.Zlaset('F', n, p, &czero, &czero, t, ldb)
	if (*n) <= (*p) {
		golapack.Zlacpy('U', n, n, bf.Off(0, (*p)-(*n)), ldb, t.Off(0, (*p)-(*n)), ldb)
	} else {
		golapack.Zlacpy('F', toPtr((*n)-(*p)), p, bf, ldb, t, ldb)
		golapack.Zlacpy('U', p, p, bf.Off((*n)-(*p), 0), ldb, t.Off((*n)-(*p), 0), ldb)
	}

	//     Compute R - Q'*A
	err = goblas.Zgemm(ConjTrans, NoTrans, *n, *m, *n, -cone, q, a, cone, r)

	//     Compute norm( R - Q'*A ) / ( max(M,N)*norm(A)*ULP ) .
	resid = golapack.Zlange('1', n, m, r, lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, *m, *n)))/anorm)/ulp)
	} else {
		result.Set(0, zero)
	}

	//     Compute T*Z - Q'*B
	err = goblas.Zgemm(NoTrans, NoTrans, *n, *p, *p, cone, t, z, czero, bwk)
	err = goblas.Zgemm(ConjTrans, NoTrans, *n, *p, *n, -cone, q, b, cone, bwk)

	//     Compute norm( T*Z - Q'*B ) / ( max(P,N)*norm(A)*ULP ) .
	resid = golapack.Zlange('1', n, p, bwk, ldb, rwork)
	if bnorm > zero {
		result.Set(1, ((resid/float64(max(1, *p, *n)))/bnorm)/ulp)
	} else {
		result.Set(1, zero)
	}

	//     Compute I - Q'*Q
	golapack.Zlaset('F', n, n, &czero, &cone, r, lda)
	err = goblas.Zherk(Upper, ConjTrans, *n, *n, -one, q, one, r)

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
