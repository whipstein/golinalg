package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zgsvts3 tests ZGGSVD3, which computes the GSVD of an M-by-N matrix A
// and a P-by-N matrix B:
//              U'*A*Q = D1*R and V'*B*Q = D2*R.
func Zgsvts3(m, p, n *int, a, af *mat.CMatrix, lda *int, b, bf *mat.CMatrix, ldb *int, u *mat.CMatrix, ldu *int, v *mat.CMatrix, ldv *int, q *mat.CMatrix, ldq *int, alpha, beta *mat.Vector, r *mat.CMatrix, ldr *int, iwork *[]int, work *mat.CVector, lwork *int, rwork, result *mat.Vector) {
	var cone, czero complex128
	var anorm, bnorm, one, resid, temp, ulp, ulpinv, unfl, zero float64
	var i, info, j, k, l int
	var err error
	_ = err

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	ulp = golapack.Dlamch(Precision)
	ulpinv = one / ulp
	unfl = golapack.Dlamch(SafeMinimum)

	//     Copy the matrix A to the array AF.
	golapack.Zlacpy('F', m, n, a, lda, af, lda)
	golapack.Zlacpy('F', p, n, b, ldb, bf, ldb)

	anorm = maxf64(golapack.Zlange('1', m, n, a, lda, rwork), unfl)
	bnorm = maxf64(golapack.Zlange('1', p, n, b, ldb, rwork), unfl)

	//     Factorize the matrices A and B in the arrays AF and BF.
	golapack.Zggsvd3('U', 'V', 'Q', m, n, p, &k, &l, af, lda, bf, ldb, alpha, beta, u, ldu, v, ldv, q, ldq, work, lwork, rwork, iwork, &info)

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
	err = goblas.Zgemm(NoTrans, NoTrans, *m, *n, *n, cone, a, *lda, q, *ldq, czero, work.CMatrix(*lda, opts), *lda)

	err = goblas.Zgemm(ConjTrans, NoTrans, *m, *n, *m, cone, u, *ldu, work.CMatrix(*lda, opts), *lda, czero, a, *lda)

	for i = 1; i <= k; i++ {
		for j = i; j <= k+l; j++ {
			a.Set(i-1, (*n)-k-l+j-1, a.Get(i-1, (*n)-k-l+j-1)-r.Get(i-1, j-1))
		}
	}

	for i = k + 1; i <= minint(k+l, *m); i++ {
		for j = i; j <= k+l; j++ {
			a.Set(i-1, (*n)-k-l+j-1, a.Get(i-1, (*n)-k-l+j-1)-alpha.GetCmplx(i-1)*r.Get(i-1, j-1))
		}
	}

	//     Compute norm( U'*A*Q - D1*R ) / ( maxint(1,M,N)*norm(A)*ULP ) .
	resid = golapack.Zlange('1', m, n, a, lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(maxint(1, *m, *n)))/anorm)/ulp)
	} else {
		result.Set(0, zero)
	}

	//     Compute B := V'*B*Q - D2*R
	err = goblas.Zgemm(NoTrans, NoTrans, *p, *n, *n, cone, b, *ldb, q, *ldq, czero, work.CMatrix(*ldb, opts), *ldb)

	err = goblas.Zgemm(ConjTrans, NoTrans, *p, *n, *p, cone, v, *ldv, work.CMatrix(*ldb, opts), *ldb, czero, b, *ldb)

	for i = 1; i <= l; i++ {
		for j = i; j <= l; j++ {
			b.Set(i-1, (*n)-l+j-1, b.Get(i-1, (*n)-l+j-1)-beta.GetCmplx(k+i-1)*r.Get(k+i-1, k+j-1))
		}
	}

	//     Compute norm( V'*B*Q - D2*R ) / ( maxint(P,N)*norm(B)*ULP ) .
	resid = golapack.Zlange('1', p, n, b, ldb, rwork)
	if bnorm > zero {
		result.Set(1, ((resid/float64(maxint(1, *p, *n)))/bnorm)/ulp)
	} else {
		result.Set(1, zero)
	}

	//     Compute I - U'*U
	golapack.Zlaset('F', m, m, &czero, &cone, work.CMatrix(*ldq, opts), ldq)
	err = goblas.Zherk(Upper, ConjTrans, *m, *m, -one, u, *ldu, one, work.CMatrix(*ldu, opts), *ldu)

	//     Compute norm( I - U'*U ) / ( M * ULP ) .
	resid = golapack.Zlanhe('1', 'U', m, work.CMatrix(*ldu, opts), ldu, rwork)
	result.Set(2, (resid/float64(maxint(1, *m)))/ulp)

	//     Compute I - V'*V
	golapack.Zlaset('F', p, p, &czero, &cone, work.CMatrix(*ldv, opts), ldv)
	err = goblas.Zherk(Upper, ConjTrans, *p, *p, -one, v, *ldv, one, work.CMatrix(*ldv, opts), *ldv)

	//     Compute norm( I - V'*V ) / ( P * ULP ) .
	resid = golapack.Zlanhe('1', 'U', p, work.CMatrix(*ldv, opts), ldv, rwork)
	result.Set(3, (resid/float64(maxint(1, *p)))/ulp)

	//     Compute I - Q'*Q
	golapack.Zlaset('F', n, n, &czero, &cone, work.CMatrix(*ldq, opts), ldq)
	err = goblas.Zherk(Upper, ConjTrans, *n, *n, -one, q, *ldq, one, work.CMatrix(*ldq, opts), *ldq)

	//     Compute norm( I - Q'*Q ) / ( N * ULP ) .
	resid = golapack.Zlanhe('1', 'U', n, work.CMatrix(*ldq, opts), ldq, rwork)
	result.Set(4, (resid/float64(maxint(1, *n)))/ulp)

	//     Check sorting
	goblas.Dcopy(*n, alpha, 1, rwork, 1)
	for i = k + 1; i <= minint(k+l, *m); i++ {
		j = (*iwork)[i-1]
		if i != j {
			temp = rwork.Get(i - 1)
			rwork.Set(i-1, rwork.Get(j-1))
			rwork.Set(j-1, temp)
		}
	}

	result.Set(5, zero)
	for i = k + 1; i <= minint(k+l, *m)-1; i++ {
		if rwork.Get(i-1) < rwork.Get(i+1-1) {
			result.Set(5, ulpinv)
		}
	}
}
