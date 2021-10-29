package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zlqt02 tests Zunglq, which generates an m-by-n matrix Q with
// orthonornmal rows that is defined as the product of k elementary
// reflectors.
//
// Given the LQ factorization of an m-by-n matrix A, ZLQT02 generates
// the orthogonal matrix Q defined by the factorization of the first k
// rows of A; it compares L(1:k,1:m) with A(1:k,1:n)*Q(1:m,1:n)', and
// checks that the rows of Q are orthonormal.
func zlqt02(m, n, k int, a, af, q, l *mat.CMatrix, tau, work *mat.CVector, lwork int, rwork, result *mat.Vector) {
	var rogue complex128
	var anorm, eps, one, resid, zero float64
	var err error

	zero = 0.0
	one = 1.0
	rogue = (-1.0e+10 + (-1.0e+10)*1i)
	srnamt := &gltest.Common.Srnamc.Srnamt

	eps = golapack.Dlamch(Epsilon)

	//     Copy the first k rows of the factorization to the array Q
	golapack.Zlaset(Full, m, n, rogue, rogue, q)
	golapack.Zlacpy(Upper, k, n-1, af.Off(0, 1), q.Off(0, 1))

	//     Generate the first n columns of the matrix Q
	*srnamt = "Zunglq"
	if err = golapack.Zunglq(m, n, k, q, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy L(1:k,1:m)
	golapack.Zlaset(Full, k, m, complex(zero, 0), complex(zero, 0), l)
	golapack.Zlacpy(Lower, k, m, af, l)

	//     Compute L(1:k,1:m) - A(1:k,1:n) * Q(1:m,1:n)'
	if err = goblas.Zgemm(NoTrans, ConjTrans, k, m, n, complex(-one, 0), a, q, complex(one, 0), l); err != nil {
		panic(err)
	}

	//     Compute norm( L - A*Q' ) / ( N * norm(A) * EPS ) .
	anorm = golapack.Zlange('1', k, n, a, rwork)
	resid = golapack.Zlange('1', k, m, l, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, n)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q*Q'
	golapack.Zlaset(Full, m, m, complex(zero, 0), complex(one, 0), l)
	if err = goblas.Zherk(Upper, NoTrans, m, n, -one, q, one, l); err != nil {
		panic(err)
	}

	//     Compute norm( I - Q*Q' ) / ( N * EPS ) .
	resid = golapack.Zlansy('1', Upper, m, l, rwork)

	result.Set(1, (resid/float64(max(1, n)))/eps)
}
