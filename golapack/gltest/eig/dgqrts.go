package eig

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
)

// Dgqrts tests DGGQRF, which computes the GQR factorization of an
// N-by-M matrix A and a N-by-P matrix B: A = Q*R and B = Q*T*Z.
func Dgqrts(n *int, m *int, p *int, a, af, q, r *mat.Matrix, lda *int, taua *mat.Vector, b, bf, z, t, bwk *mat.Matrix, ldb *int, taub, work *mat.Vector, lwork *int, rwork, result *mat.Vector) {
	var anorm, bnorm, one, resid, rogue, ulp, unfl, zero float64
	var info int

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	ulp = golapack.Dlamch(Precision)
	unfl = golapack.Dlamch(SafeMinimum)

	//     Copy the matrix A to the array AF.
	golapack.Dlacpy('F', n, m, a, lda, af, lda)
	golapack.Dlacpy('F', n, p, b, ldb, bf, ldb)

	anorm = maxf64(golapack.Dlange('1', n, m, a, lda, rwork), unfl)
	bnorm = maxf64(golapack.Dlange('1', n, p, b, ldb, rwork), unfl)

	//     Factorize the matrices A and B in the arrays AF and BF.
	golapack.Dggqrf(n, m, p, af, lda, taua, bf, ldb, taub, work, lwork, &info)

	//     Generate the N-by-N matrix Q
	golapack.Dlaset('F', n, n, &rogue, &rogue, q, lda)
	golapack.Dlacpy('L', toPtr((*n)-1), m, af.Off(1, 0), lda, q.Off(1, 0), lda)
	golapack.Dorgqr(n, n, toPtr(minint(*n, *m)), q, lda, taua, work, lwork, &info)

	//     Generate the P-by-P matrix Z
	golapack.Dlaset('F', p, p, &rogue, &rogue, z, ldb)
	if (*n) <= (*p) {
		if (*n) > 0 && (*n) < (*p) {
			golapack.Dlacpy('F', n, toPtr((*p)-(*n)), bf, ldb, z.Off((*p)-(*n)+1-1, 0), ldb)
		}
		if (*n) > 1 {
			golapack.Dlacpy('L', toPtr((*n)-1), toPtr((*n)-1), bf.Off(1, (*p)-(*n)+1-1), ldb, z.Off((*p)-(*n)+2-1, (*p)-(*n)+1-1), ldb)
		}
	} else {
		if (*p) > 1 {
			golapack.Dlacpy('L', toPtr((*p)-1), toPtr((*p)-1), bf.Off((*n)-(*p)+2-1, 0), ldb, z.Off(1, 0), ldb)
		}
	}
	golapack.Dorgrq(p, p, toPtr(minint(*n, *p)), z, ldb, taub, work, lwork, &info)

	//     Copy R
	golapack.Dlaset('F', n, m, &zero, &zero, r, lda)
	golapack.Dlacpy('U', n, m, af, lda, r, lda)

	//     Copy T
	golapack.Dlaset('F', n, p, &zero, &zero, t, ldb)
	if (*n) <= (*p) {
		golapack.Dlacpy('U', n, n, bf.Off(0, (*p)-(*n)+1-1), ldb, t.Off(0, (*p)-(*n)+1-1), ldb)
	} else {
		golapack.Dlacpy('F', toPtr((*n)-(*p)), p, bf, ldb, t, ldb)
		golapack.Dlacpy('U', p, p, bf.Off((*n)-(*p)+1-1, 0), ldb, t.Off((*n)-(*p)+1-1, 0), ldb)
	}

	//     Compute R - Q'*A
	goblas.Dgemm(Trans, NoTrans, n, m, n, toPtrf64(-one), q, lda, a, lda, &one, r, lda)

	//     Compute norm( R - Q'*A ) / ( MAX(M,N)*norm(A)*ULP ) .
	resid = golapack.Dlange('1', n, m, r, lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(maxint(1, *m, *n)))/anorm)/ulp)
	} else {
		result.Set(0, zero)
	}

	//     Compute T*Z - Q'*B
	goblas.Dgemm(NoTrans, NoTrans, n, p, p, &one, t, ldb, z, ldb, &zero, bwk, ldb)
	goblas.Dgemm(Trans, NoTrans, n, p, n, toPtrf64(-one), q, lda, b, ldb, &one, bwk, ldb)

	//     Compute norm( T*Z - Q'*B ) / ( MAX(P,N)*norm(A)*ULP ) .
	resid = golapack.Dlange('1', n, p, bwk, ldb, rwork)
	if bnorm > zero {
		result.Set(1, ((resid/float64(maxint(1, *p, *n)))/bnorm)/ulp)
	} else {
		result.Set(1, zero)
	}

	//     Compute I - Q'*Q
	golapack.Dlaset('F', n, n, &zero, &one, r, lda)
	goblas.Dsyrk(Upper, Trans, n, n, toPtrf64(-one), q, lda, &one, r, lda)

	//     Compute norm( I - Q'*Q ) / ( N * ULP ) .
	resid = golapack.Dlansy('1', 'U', n, r, lda, rwork)
	result.Set(2, (resid/float64(maxint(1, *n)))/ulp)

	//     Compute I - Z'*Z
	golapack.Dlaset('F', p, p, &zero, &one, t, ldb)
	goblas.Dsyrk(Upper, Trans, p, p, toPtrf64(-one), z, ldb, &one, t, ldb)

	//     Compute norm( I - Z'*Z ) / ( P*ULP ) .
	resid = golapack.Dlansy('1', 'U', p, t, ldb, rwork)
	result.Set(3, (resid/float64(maxint(1, *p)))/ulp)
}
