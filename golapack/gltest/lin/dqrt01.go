package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dqrt01 tests Dgeqrf, which computes the QR factorization of an m-by-n
// matrix A, and partially tests Dorgqr which forms the m-by-m
// orthogonal matrix Q.
//
// DQRT01 compares R with Q'*A, and checks that Q is orthogonal.
func dqrt01(m, n int, a, af, q, r *mat.Matrix, tau, work *mat.Vector, lwork int, rwork, result *mat.Vector) {
	var anorm, eps, one, resid, rogue, zero float64
	var minmn int
	var err error

	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	minmn = min(m, n)
	eps = golapack.Dlamch(Epsilon)

	//     Copy the matrix A to the array AF.
	golapack.Dlacpy(Full, m, n, a, af)

	//     Factorize the matrix A in the array AF.
	*srnamt = "Dgeqrf"
	if err = golapack.Dgeqrf(m, n, af, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy details of Q
	golapack.Dlaset(Full, m, m, rogue, rogue, q)
	golapack.Dlacpy(Lower, m-1, n, af.Off(1, 0), q.Off(1, 0))

	//     Generate the m-by-m matrix Q
	*srnamt = "Dorgqr"
	if err = golapack.Dorgqr(m, m, minmn, q, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy R
	golapack.Dlaset(Full, m, n, zero, zero, r)
	golapack.Dlacpy(Upper, m, n, af, r)

	//     Compute R - Q'*A
	if err = r.Gemm(Trans, NoTrans, m, n, m, -one, q, a, one); err != nil {
		panic(err)
	}

	//     Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) .
	anorm = golapack.Dlange('1', m, n, a, rwork)
	resid = golapack.Dlange('1', m, n, r, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, m)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q'*Q
	golapack.Dlaset(Full, m, m, zero, one, r)
	err = r.Syrk(Upper, Trans, m, m, -one, q, one)

	//     Compute norm( I - Q'*Q ) / ( M * EPS ) .
	resid = golapack.Dlansy('1', Upper, m, r, rwork)

	result.Set(1, (resid/float64(max(1, m)))/eps)
}
