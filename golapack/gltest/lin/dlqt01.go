package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dlqt01 tests Dgelqf, which computes the LQ factorization of an m-by-n
// matrix A, and partially tests Dorglq which forms the n-by-n
// orthogonal matrix Q.
//
// DLQT01 compares L with A*Q', and checks that Q is orthogonal.
func dlqt01(m, n int, a, af, q, l *mat.Matrix, tau, work *mat.Vector, lwork int, rwork, result *mat.Vector) {
	var anorm, eps, one, resid, rogue, zero float64
	var minmn int
	var err error

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	srnamt := &gltest.Common.Srnamc.Srnamt

	minmn = min(m, n)
	eps = golapack.Dlamch(Epsilon)

	//     Copy the matrix A to the array AF.
	golapack.Dlacpy(Full, m, n, a, af)

	//     Factorize the matrix A in the array AF.
	*srnamt = "Dgelqf"
	if err = golapack.Dgelqf(m, n, af, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy details of Q
	golapack.Dlaset(Full, n, n, rogue, rogue, q)
	if n > 1 {
		golapack.Dlacpy(Upper, m, n-1, af.Off(0, 1), q.Off(0, 1))
	}

	//     Generate the n-by-n matrix Q
	*srnamt = "Dorglq"
	if err = golapack.Dorglq(n, n, minmn, q, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy L
	golapack.Dlaset(Full, m, n, zero, zero, l)
	golapack.Dlacpy(Lower, m, n, af, l)

	//     Compute L - A*Q'
	err = l.Gemm(NoTrans, Trans, m, n, n, -one, a, q, one)

	//     Compute norm( L - Q'*A ) / ( N * norm(A) * EPS ) .
	anorm = golapack.Dlange('1', m, n, a, rwork)
	resid = golapack.Dlange('1', m, n, l, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, n)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q*Q'
	golapack.Dlaset(Full, n, n, zero, one, l)
	if err = l.Syrk(Upper, NoTrans, n, n, -one, q, one); err != nil {
		panic(err)
	}

	//     Compute norm( I - Q*Q' ) / ( N * EPS ) .
	resid = golapack.Dlansy('1', Upper, n, l, rwork)

	result.Set(1, (resid/float64(max(1, n)))/eps)
}
