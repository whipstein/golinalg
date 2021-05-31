package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dgrqts tests DGGRQF, which computes the GRQ factorization of an
// M-by-N matrix A and a P-by-N matrix B: A = R*Q and B = Z*T*Q.
func Dgrqts(m *int, p *int, n *int, a, af, q, r *mat.Matrix, lda *int, taua *mat.Vector, b, bf, z, t, bwk *mat.Matrix, ldb *int, taub, work *mat.Vector, lwork *int, rwork, result *mat.Vector) {
	var anorm, bnorm, one, resid, rogue, ulp, unfl, zero float64
	var info int

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	ulp = golapack.Dlamch(Precision)
	unfl = golapack.Dlamch(SafeMinimum)

	//     Copy the matrix A to the array AF.
	golapack.Dlacpy('F', m, n, a, lda, af, lda)
	golapack.Dlacpy('F', p, n, b, ldb, bf, ldb)

	anorm = maxf64(golapack.Dlange('1', m, n, a, lda, rwork), unfl)
	bnorm = maxf64(golapack.Dlange('1', p, n, b, ldb, rwork), unfl)

	//     Factorize the matrices A and B in the arrays AF and BF.
	golapack.Dggrqf(m, p, n, af, lda, taua, bf, ldb, taub, work, lwork, &info)

	//     Generate the N-by-N matrix Q
	golapack.Dlaset('F', n, n, &rogue, &rogue, q, lda)
	if (*m) <= (*n) {
		if (*m) > 0 && (*m) < (*n) {
			golapack.Dlacpy('F', m, toPtr((*n)-(*m)), af, lda, q.Off((*n)-(*m)+1-1, 0), lda)
		}
		if (*m) > 1 {
			golapack.Dlacpy('L', toPtr((*m)-1), toPtr((*m)-1), af.Off(1, (*n)-(*m)+1-1), lda, q.Off((*n)-(*m)+2-1, (*n)-(*m)+1-1), lda)
		}
	} else {
		if (*n) > 1 {
			golapack.Dlacpy('L', toPtr((*n)-1), toPtr((*n)-1), af.Off((*m)-(*n)+2-1, 0), lda, q.Off(1, 0), lda)
		}
	}
	golapack.Dorgrq(n, n, toPtr(minint(*m, *n)), q, lda, taua, work, lwork, &info)

	//     Generate the P-by-P matrix Z
	golapack.Dlaset('F', p, p, &rogue, &rogue, z, ldb)
	if (*p) > 1 {
		golapack.Dlacpy('L', toPtr((*p)-1), n, bf.Off(1, 0), ldb, z.Off(1, 0), ldb)
	}
	golapack.Dorgqr(p, p, toPtr(minint(*p, *n)), z, ldb, taub, work, lwork, &info)

	//     Copy R
	golapack.Dlaset('F', m, n, &zero, &zero, r, lda)
	if (*m) <= (*n) {
		golapack.Dlacpy('U', m, m, af.Off(0, (*n)-(*m)+1-1), lda, r.Off(0, (*n)-(*m)+1-1), lda)
	} else {
		golapack.Dlacpy('F', toPtr((*m)-(*n)), n, af, lda, r, lda)
		golapack.Dlacpy('U', n, n, af.Off((*m)-(*n)+1-1, 0), lda, r.Off((*m)-(*n)+1-1, 0), lda)
	}

	//     Copy T
	golapack.Dlaset('F', p, n, &zero, &zero, t, ldb)
	golapack.Dlacpy('U', p, n, bf, ldb, t, ldb)

	//     Compute R - A*Q'
	goblas.Dgemm(NoTrans, Trans, m, n, n, toPtrf64(-one), a, lda, q, lda, &one, r, lda)

	//     Compute norm( R - A*Q' ) / ( maxf64(M,N)*norm(A)*ULP ) .
	resid = golapack.Dlange('1', m, n, r, lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(maxint(1, *m, *n)))/anorm)/ulp)
	} else {
		result.Set(0, zero)
	}

	//     Compute T*Q - Z'*B
	goblas.Dgemm(Trans, NoTrans, p, n, p, &one, z, ldb, b, ldb, &zero, bwk, ldb)
	goblas.Dgemm(NoTrans, NoTrans, p, n, n, &one, t, ldb, q, lda, toPtrf64(-one), bwk, ldb)

	//     Compute norm( T*Q - Z'*B ) / ( maxf64(P,N)*norm(A)*ULP ) .
	resid = golapack.Dlange('1', p, n, bwk, ldb, rwork)
	if bnorm > zero {
		result.Set(1, ((resid/float64(maxint(1, *p, *m)))/bnorm)/ulp)
	} else {
		result.Set(1, zero)
	}

	//     Compute I - Q*Q'
	golapack.Dlaset('F', n, n, &zero, &one, r, lda)
	goblas.Dsyrk(Upper, NoTrans, n, n, toPtrf64(-one), q, lda, &one, r, lda)

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
