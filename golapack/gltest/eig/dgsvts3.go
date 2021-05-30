package eig

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
)

// Dgsvts3 tests DGGSVD3, which computes the GSVD of an M-by-N matrix A
// and a P-by-N matrix B:
//              U'*A*Q = D1*R and V'*B*Q = D2*R.
func Dgsvts3(m, p, n *int, a, af *mat.Matrix, lda *int, b, bf *mat.Matrix, ldb *int, u *mat.Matrix, ldu *int, v *mat.Matrix, ldv *int, q *mat.Matrix, ldq *int, alpha, beta *mat.Vector, r *mat.Matrix, ldr *int, iwork *[]int, work *mat.Vector, lwork *int, rwork, result *mat.Vector) {
	var anorm, bnorm, one, resid, temp, ulp, ulpinv, unfl, zero float64
	var i, info, j, k, l int

	zero = 0.0
	one = 1.0

	ulp = golapack.Dlamch(Precision)
	ulpinv = one / ulp
	unfl = golapack.Dlamch(SafeMinimum)

	//     Copy the matrix A to the array AF.
	golapack.Dlacpy('F', m, n, a, lda, af, lda)
	golapack.Dlacpy('F', p, n, b, ldb, bf, ldb)

	anorm = maxf64(golapack.Dlange('1', m, n, a, lda, rwork), unfl)
	bnorm = maxf64(golapack.Dlange('1', p, n, b, ldb, rwork), unfl)

	//     Factorize the matrices A and B in the arrays AF and BF.
	golapack.Dggsvd3('U', 'V', 'Q', m, n, p, &k, &l, af, lda, bf, ldb, alpha, beta, u, ldu, v, ldv, q, ldq, work, lwork, iwork, &info)

	//     Copy R
	for i = 1; i <= minint(k+l, *m); i++ {
		for j = i; j <= k+l; j++ {
			r.Set(i-1, j-1, af.Get(i-1, (*n)-k-l+j-1))
		}
	}

	if (*m)-k-l < 0 {
		for i = (*m) + 1; i <= k+l; i++ {
			for j = i; j <= k+l; j++ {
				r.Set(i-1, j-1, bf.Get(i-k-1, (*n)-k-l+j-1))
			}
		}
	}

	//     Compute A:= U'*A*Q - D1*R
	goblas.Dgemm(NoTrans, NoTrans, m, n, n, &one, a, lda, q, ldq, &zero, work.Matrix(*lda, opts), lda)

	goblas.Dgemm(Trans, NoTrans, m, n, m, &one, u, ldu, work.Matrix(*lda, opts), lda, &zero, a, lda)

	for i = 1; i <= k; i++ {
		for j = i; j <= k+l; j++ {
			a.Set(i-1, (*n)-k-l+j-1, a.Get(i-1, (*n)-k-l+j-1)-r.Get(i-1, j-1))
		}
	}

	for i = k + 1; i <= minint(k+l, *m); i++ {
		for j = i; j <= k+l; j++ {
			a.Set(i-1, (*n)-k-l+j-1, a.Get(i-1, (*n)-k-l+j-1)-alpha.Get(i-1)*r.Get(i-1, j-1))
		}
	}

	//     Compute norm( U'*A*Q - D1*R ) / ( maxint(1,M,N)*norm(A)*ULP ) .
	resid = golapack.Dlange('1', m, n, a, lda, rwork)

	if anorm > zero {
		result.Set(0, ((resid/float64(maxint(1, *m, *n)))/anorm)/ulp)
	} else {
		result.Set(0, zero)
	}

	//     Compute B := V'*B*Q - D2*R
	goblas.Dgemm(NoTrans, NoTrans, p, n, n, &one, b, ldb, q, ldq, &zero, work.Matrix(*ldb, opts), ldb)

	goblas.Dgemm(Trans, NoTrans, p, n, p, &one, v, ldv, work.Matrix(*lda, opts), ldb, &zero, b, ldb)

	for i = 1; i <= l; i++ {
		for j = i; j <= l; j++ {
			b.Set(i-1, (*n)-l+j-1, b.Get(i-1, (*n)-l+j-1)-beta.Get(k+i-1)*r.Get(k+i-1, k+j-1))
		}
	}

	//     Compute norm( V'*B*Q - D2*R ) / ( maxint(P,N)*norm(B)*ULP ) .
	resid = golapack.Dlange('1', p, n, b, ldb, rwork)
	if bnorm > zero {
		result.Set(1, ((resid/float64(maxint(1, *p, *n)))/bnorm)/ulp)
	} else {
		result.Set(1, zero)
	}

	//     Compute I - U'*U
	golapack.Dlaset('F', m, m, &zero, &one, work.Matrix(*ldq, opts), ldq)
	goblas.Dsyrk(Upper, Trans, m, m, toPtrf64(-one), u, ldu, &one, work.Matrix(*ldu, opts), ldu)

	//     Compute norm( I - U'*U ) / ( M * ULP ) .
	resid = golapack.Dlansy('1', 'U', m, work.Matrix(*ldu, opts), ldu, rwork)
	result.Set(2, (resid/float64(maxint(1, *m)))/ulp)

	//     Compute I - V'*V
	golapack.Dlaset('F', p, p, &zero, &one, work.Matrix(*ldv, opts), ldv)
	goblas.Dsyrk(Upper, Trans, p, p, toPtrf64(-one), v, ldv, &one, work.Matrix(*ldv, opts), ldv)

	//     Compute norm( I - V'*V ) / ( P * ULP ) .
	resid = golapack.Dlansy('1', 'U', p, work.Matrix(*ldv, opts), ldv, rwork)
	result.Set(3, (resid/float64(maxint(1, *p)))/ulp)

	//     Compute I - Q'*Q
	golapack.Dlaset('F', n, n, &zero, &one, work.Matrix(*ldq, opts), ldq)
	goblas.Dsyrk(Upper, Trans, n, n, toPtrf64(-one), q, ldq, &one, work.Matrix(*ldq, opts), ldq)

	//     Compute norm( I - Q'*Q ) / ( N * ULP ) .
	resid = golapack.Dlansy('1', 'U', n, work.Matrix(*ldq, opts), ldq, rwork)
	result.Set(4, (resid/float64(maxint(1, *n)))/ulp)

	//     Check sorting
	goblas.Dcopy(n, alpha, func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
	for i = k + 1; i <= minint(k+l, *m); i++ {
		j = (*iwork)[i-1]
		if i != j {
			temp = work.Get(i - 1)
			work.Set(i-1, work.Get(j-1))
			work.Set(j-1, temp)
		}
	}

	result.Set(5, zero)
	for i = k + 1; i <= minint(k+l, *m)-1; i++ {
		if work.Get(i-1) < work.Get(i+1-1) {
			result.Set(5, ulpinv)
		}
	}
}
