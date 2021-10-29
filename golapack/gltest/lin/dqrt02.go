package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dqrt02 tests Dorgqr, which generates an m-by-n matrix Q with
// orthonornmal columns that is defined as the product of k elementary
// reflectors.
//
// Given the QR factorization of an m-by-n matrix A, DQRT02 generates
// the orthogonal matrix Q defined by the factorization of the first k
// columns of A; it compares R(1:n,1:k) with Q(1:m,1:n)'*A(1:m,1:k),
// and checks that the columns of Q are orthonormal.
func dqrt02(m, n, k int, a, af, q, r *mat.Matrix, tau, work *mat.Vector, lwork int, rwork, result *mat.Vector) {
	var anorm, eps, one, resid, rogue, zero float64
	var err error

	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	eps = golapack.Dlamch(Epsilon)

	//     Copy the first k columns of the factorization to the array Q
	golapack.Dlaset(Full, m, n, rogue, rogue, q)
	golapack.Dlacpy(Lower, m-1, k, af.Off(1, 0), q.Off(1, 0))

	//     Generate the first n columns of the matrix Q
	*srnamt = "Dorgqr"
	if err = golapack.Dorgqr(m, n, k, q, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy R(1:n,1:k)
	golapack.Dlaset(Full, n, k, zero, zero, r)
	golapack.Dlacpy(Upper, n, k, af, r)

	//     Compute R(1:n,1:k) - Q(1:m,1:n)' * A(1:m,1:k)
	if err = goblas.Dgemm(Trans, NoTrans, n, k, m, -one, q, a, one, r); err != nil {
		panic((err))
	}

	//     Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) .
	anorm = golapack.Dlange('1', m, k, a, rwork)
	resid = golapack.Dlange('1', n, k, r, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, m)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q'*Q
	golapack.Dlaset(Full, n, n, zero, one, r)
	if err = goblas.Dsyrk(Upper, Trans, n, m, -one, q, one, r); err != nil {
		panic(err)
	}

	//     Compute norm( I - Q'*Q ) / ( M * EPS ) .
	resid = golapack.Dlansy('1', Upper, n, r, rwork)

	result.Set(1, (resid/float64(max(1, m)))/eps)
}
