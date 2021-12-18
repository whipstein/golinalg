package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dlqt02 tests Dorglq, which generates an m-by-n matrix Q with
// orthonornmal rows that is defined as the product of k elementary
// reflectors.
//
// Given the LQ factorization of an m-by-n matrix A, DLQT02 generates
// the orthogonal matrix Q defined by the factorization of the first k
// rows of A; it compares L(1:k,1:m) with A(1:k,1:n)*Q(1:m,1:n)', and
// checks that the rows of Q are orthonormal.
func dlqt02(m, n, k int, a, af, q, l *mat.Matrix, tau, work *mat.Vector, lwork int, rwork, result *mat.Vector) {
	var anorm, eps, one, resid, rogue, zero float64
	var err error

	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	eps = golapack.Dlamch(Epsilon)

	//     Copy the first k rows of the factorization to the array Q
	golapack.Dlaset(Full, m, n, rogue, rogue, q)
	golapack.Dlacpy(Upper, k, n-1, af.Off(0, 1), q.Off(0, 1))

	//     Generate the first n columns of the matrix Q
	*srnamt = "Dorglq"
	if err = golapack.Dorglq(m, n, k, q, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy L(1:k,1:m)
	golapack.Dlaset(Full, k, m, zero, zero, l)
	golapack.Dlacpy(Lower, k, m, af, l)

	//     Compute L(1:k,1:m) - A(1:k,1:n) * Q(1:m,1:n)'
	err = l.Gemm(mat.NoTrans, mat.Trans, k, m, n, -one, a, q, one)

	//     Compute norm( L - A*Q' ) / ( N * norm(A) * EPS ) .
	anorm = golapack.Dlange('1', k, n, a, rwork)
	resid = golapack.Dlange('1', k, m, l, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, n)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q*Q'
	golapack.Dlaset(Full, m, m, zero, one, l)
	if err = l.Syrk(Upper, NoTrans, m, n, -one, q, one); err != nil {
		panic(err)
	}

	//     Compute norm( I - Q*Q' ) / ( N * EPS ) .
	resid = golapack.Dlansy('1', Upper, m, l, rwork)

	result.Set(1, (resid/float64(max(1, n)))/eps)
}
