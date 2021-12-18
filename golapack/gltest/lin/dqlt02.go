package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dqlt02 tests DORGQL, which generates an m-by-n matrix Q with
// orthonornmal columns that is defined as the product of k elementary
// reflectors.
//
// Given the QL factorization of an m-by-n matrix A, DQLT02 generates
// the orthogonal matrix Q defined by the factorization of the last k
// columns of A; it compares L(m-n+1:m,n-k+1:n) with
// Q(1:m,m-n+1:m)'*A(1:m,n-k+1:n), and checks that the columns of Q are
// orthonormal.
func dqlt02(m, n, k int, a, af, q, l *mat.Matrix, tau, work *mat.Vector, lwork int, rwork, result *mat.Vector) {
	var anorm, eps, one, resid, rogue, zero float64
	var err error

	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	if m == 0 || n == 0 || k == 0 {
		result.Set(0, zero)
		result.Set(1, zero)
		return
	}

	eps = golapack.Dlamch(Epsilon)

	//     Copy the last k columns of the factorization to the array Q
	golapack.Dlaset(Full, m, n, rogue, rogue, q)
	if k < m {
		golapack.Dlacpy(Full, m-k, k, af.Off(0, n-k), q.Off(0, n-k))
	}
	if k > 1 {
		golapack.Dlacpy(Upper, k-1, k-1, af.Off(m-k, n-k+2-1), q.Off(m-k, n-k+2-1))
	}

	//     Generate the last n columns of the matrix Q
	*srnamt = "Dorgql"
	if err = golapack.Dorgql(m, n, k, q, tau.Off(n-k), work, lwork); err != nil {
		panic(err)
	}

	//     Copy L(m-n+1:m,n-k+1:n)
	golapack.Dlaset(Full, n, k, zero, zero, l.Off(m-n, n-k))
	golapack.Dlacpy(Lower, k, k, af.Off(m-k, n-k), l.Off(m-k, n-k))

	//     Compute L(m-n+1:m,n-k+1:n) - Q(1:m,m-n+1:m)' * A(1:m,n-k+1:n)
	if err = l.Off(m-n, n-k).Gemm(Trans, NoTrans, n, k, m, -one, q, a.Off(0, n-k), one); err != nil {
		panic(err)
	}

	//     Compute norm( L - Q'*A ) / ( M * norm(A) * EPS ) .
	anorm = golapack.Dlange('1', m, k, a.Off(0, n-k), rwork)
	resid = golapack.Dlange('1', n, k, l.Off(m-n, n-k), rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, m)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q'*Q
	golapack.Dlaset(Full, n, n, zero, one, l)
	if err = l.Syrk(Upper, Trans, n, m, -one, q, one); err != nil {
		panic(err)
	}

	//     Compute norm( I - Q'*Q ) / ( M * EPS ) .
	resid = golapack.Dlansy('1', Upper, n, l, rwork)

	result.Set(1, (resid/float64(max(1, m)))/eps)
}
