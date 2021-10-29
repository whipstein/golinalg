package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dgsvts3 tests DGGSVD3, which computes the GSVD of an M-by-N matrix A
// and a P-by-N matrix B:
//              U'*A*Q = D1*R and V'*B*Q = D2*R.
func dgsvts3(m, p, n int, a, af, b, bf, u, v, q *mat.Matrix, alpha, beta *mat.Vector, r *mat.Matrix, iwork *[]int, work *mat.Vector, lwork int, rwork, result *mat.Vector) {
	var anorm, bnorm, one, resid, temp, ulp, ulpinv, unfl, zero float64
	var i, j, k, l int
	var err error

	zero = 0.0
	one = 1.0

	ulp = golapack.Dlamch(Precision)
	ulpinv = one / ulp
	unfl = golapack.Dlamch(SafeMinimum)

	//     Copy the matrix A to the array AF.
	golapack.Dlacpy(Full, m, n, a, af)
	golapack.Dlacpy(Full, p, n, b, bf)

	anorm = math.Max(golapack.Dlange('1', m, n, a, rwork), unfl)
	bnorm = math.Max(golapack.Dlange('1', p, n, b, rwork), unfl)

	//     Factorize the matrices A and B in the arrays AF and BF.
	if k, l, _, err = golapack.Dggsvd3('U', 'V', 'Q', m, n, p, af, bf, alpha, beta, u, v, q, work, lwork, iwork); err != nil {
		panic(err)
	}

	//     Copy R
	for i = 1; i <= min(k+l, m); i++ {
		for j = i; j <= k+l; j++ {
			r.Set(i-1, j-1, af.Get(i-1, n-k-l+j-1))
		}
	}

	if m-k-l < 0 {
		for i = m + 1; i <= k+l; i++ {
			for j = i; j <= k+l; j++ {
				r.Set(i-1, j-1, bf.Get(i-k-1, n-k-l+j-1))
			}
		}
	}

	//     Compute A:= U'*A*Q - D1*R
	if err = goblas.Dgemm(NoTrans, NoTrans, m, n, n, one, a, q, zero, work.Matrix(a.Rows, opts)); err != nil {
		panic(err)
	}

	if err = goblas.Dgemm(Trans, NoTrans, m, n, m, one, u, work.Matrix(a.Rows, opts), zero, a); err != nil {
		panic(err)
	}

	for i = 1; i <= k; i++ {
		for j = i; j <= k+l; j++ {
			a.Set(i-1, n-k-l+j-1, a.Get(i-1, n-k-l+j-1)-r.Get(i-1, j-1))
		}
	}

	for i = k + 1; i <= min(k+l, m); i++ {
		for j = i; j <= k+l; j++ {
			a.Set(i-1, n-k-l+j-1, a.Get(i-1, n-k-l+j-1)-alpha.Get(i-1)*r.Get(i-1, j-1))
		}
	}

	//     Compute norm( U'*A*Q - D1*R ) / ( max(1,M,N)*norm(A)*ULP ) .
	resid = golapack.Dlange('1', m, n, a, rwork)

	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, m, n)))/anorm)/ulp)
	} else {
		result.Set(0, zero)
	}

	//     Compute B := V'*B*Q - D2*R
	if err = goblas.Dgemm(NoTrans, NoTrans, p, n, n, one, b, q, zero, work.Matrix(b.Rows, opts)); err != nil {
		panic(err)
	}

	if err = goblas.Dgemm(Trans, NoTrans, p, n, p, one, v, work.Matrix(a.Rows, opts), zero, b); err != nil {
		panic(err)
	}

	for i = 1; i <= l; i++ {
		for j = i; j <= l; j++ {
			b.Set(i-1, n-l+j-1, b.Get(i-1, n-l+j-1)-beta.Get(k+i-1)*r.Get(k+i-1, k+j-1))
		}
	}

	//     Compute norm( V'*B*Q - D2*R ) / ( max(P,N)*norm(B)*ULP ) .
	resid = golapack.Dlange('1', p, n, b, rwork)
	if bnorm > zero {
		result.Set(1, ((resid/float64(max(1, p, n)))/bnorm)/ulp)
	} else {
		result.Set(1, zero)
	}

	//     Compute I - U'*U
	golapack.Dlaset(Full, m, m, zero, one, work.Matrix(q.Rows, opts))
	if err = goblas.Dsyrk(Upper, Trans, m, m, -one, u, one, work.Matrix(u.Rows, opts)); err != nil {
		panic(err)
	}

	//     Compute norm( I - U'*U ) / ( M * ULP ) .
	resid = golapack.Dlansy('1', Upper, m, work.Matrix(u.Rows, opts), rwork)
	result.Set(2, (resid/float64(max(1, m)))/ulp)

	//     Compute I - V'*V
	golapack.Dlaset(Full, p, p, zero, one, work.Matrix(v.Rows, opts))
	if err = goblas.Dsyrk(Upper, Trans, p, p, -one, v, one, work.Matrix(v.Rows, opts)); err != nil {
		panic(err)
	}

	//     Compute norm( I - V'*V ) / ( P * ULP ) .
	resid = golapack.Dlansy('1', Upper, p, work.Matrix(v.Rows, opts), rwork)
	result.Set(3, (resid/float64(max(1, p)))/ulp)

	//     Compute I - Q'*Q
	golapack.Dlaset(Full, n, n, zero, one, work.Matrix(q.Rows, opts))
	if err = goblas.Dsyrk(Upper, Trans, n, n, -one, q, one, work.Matrix(q.Rows, opts)); err != nil {
		panic(err)
	}

	//     Compute norm( I - Q'*Q ) / ( N * ULP ) .
	resid = golapack.Dlansy('1', Upper, n, work.Matrix(q.Rows, opts), rwork)
	result.Set(4, (resid/float64(max(1, n)))/ulp)

	//     Check sorting
	goblas.Dcopy(n, alpha.Off(0, 1), work.Off(0, 1))
	for i = k + 1; i <= min(k+l, m); i++ {
		j = (*iwork)[i-1]
		if i != j {
			temp = work.Get(i - 1)
			work.Set(i-1, work.Get(j-1))
			work.Set(j-1, temp)
		}
	}

	result.Set(5, zero)
	for i = k + 1; i <= min(k+l, m)-1; i++ {
		if work.Get(i-1) < work.Get(i) {
			result.Set(5, ulpinv)
		}
	}

	return
}
