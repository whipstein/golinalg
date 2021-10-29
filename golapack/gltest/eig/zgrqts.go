package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zgrqts tests ZGGRQF, which computes the GRQ factorization of an
// M-by-N matrix A and a P-by-N matrix B: A = R*Q and B = Z*T*Q.
func zgrqts(m, p, n int, a, af, q, r *mat.CMatrix, taua *mat.CVector, b, bf, z, t, bwk *mat.CMatrix, taub, work *mat.CVector, lwork int, rwork, result *mat.Vector) {
	var cone, crogue, czero complex128
	var anorm, bnorm, one, resid, ulp, unfl, zero float64
	var err error

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	crogue = (-1.0e+10 + 0.0*1i)

	ulp = golapack.Dlamch(Precision)
	unfl = golapack.Dlamch(SafeMinimum)

	//     Copy the matrix A to the array AF.
	golapack.Zlacpy(Full, m, n, a, af)
	golapack.Zlacpy(Full, p, n, b, bf)

	anorm = math.Max(golapack.Zlange('1', m, n, a, rwork), unfl)
	bnorm = math.Max(golapack.Zlange('1', p, n, b, rwork), unfl)

	//     Factorize the matrices A and B in the arrays AF and BF.
	if err = golapack.Zggrqf(m, p, n, af, taua, bf, taub, work, lwork); err != nil {
		panic(err)
	}

	//     Generate the N-by-N matrix Q
	golapack.Zlaset(Full, n, n, crogue, crogue, q)
	if m <= n {
		if m > 0 && m < n {
			golapack.Zlacpy(Full, m, n-m, af, q.Off(n-m, 0))
		}
		if m > 1 {
			golapack.Zlacpy(Lower, m-1, m-1, af.Off(1, n-m), q.Off(n-m+2-1, n-m))
		}
	} else {
		if n > 1 {
			golapack.Zlacpy(Lower, n-1, n-1, af.Off(m-n+2-1, 0), q.Off(1, 0))
		}
	}
	if err = golapack.Zungrq(n, n, min(m, n), q, taua, work, lwork); err != nil {
		panic(err)
	}

	//     Generate the P-by-P matrix Z
	golapack.Zlaset(Full, p, p, crogue, crogue, z)
	if p > 1 {
		golapack.Zlacpy(Lower, p-1, n, bf.Off(1, 0), z.Off(1, 0))
	}
	if err = golapack.Zungqr(p, p, min(p, n), z, taub, work, lwork); err != nil {
		panic(err)
	}

	//     Copy R
	golapack.Zlaset(Full, m, n, czero, czero, r)
	if m <= n {
		golapack.Zlacpy(Upper, m, m, af.Off(0, n-m), r.Off(0, n-m))
	} else {
		golapack.Zlacpy(Full, m-n, n, af, r)
		golapack.Zlacpy(Upper, n, n, af.Off(m-n, 0), r.Off(m-n, 0))
	}

	//     Copy T
	golapack.Zlaset(Full, p, n, czero, czero, t)
	golapack.Zlacpy(Upper, p, n, bf, t)

	//     Compute R - A*Q'
	if err = goblas.Zgemm(NoTrans, ConjTrans, m, n, n, -cone, a, q, cone, r); err != nil {
		panic(err)
	}

	//     Compute norm( R - A*Q' ) / ( max(M,N)*norm(A)*ULP ) .
	resid = golapack.Zlange('1', m, n, r, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, m, n)))/anorm)/ulp)
	} else {
		result.Set(0, zero)
	}

	//     Compute T*Q - Z'*B
	if err = goblas.Zgemm(ConjTrans, NoTrans, p, n, p, cone, z, b, czero, bwk); err != nil {
		panic(err)
	}
	if err = goblas.Zgemm(NoTrans, NoTrans, p, n, n, cone, t, q, -cone, bwk); err != nil {
		panic(err)
	}

	//     Compute norm( T*Q - Z'*B ) / ( max(P,N)*norm(A)*ULP ) .
	resid = golapack.Zlange('1', p, n, bwk, rwork)
	if bnorm > zero {
		result.Set(1, ((resid/float64(max(1, p, m)))/bnorm)/ulp)
	} else {
		result.Set(1, zero)
	}

	//     Compute I - Q*Q'
	golapack.Zlaset(Full, n, n, czero, cone, r)
	if err = goblas.Zherk(Upper, NoTrans, n, n, -one, q, one, r); err != nil {
		panic(err)
	}

	//     Compute norm( I - Q'*Q ) / ( N * ULP ) .
	resid = golapack.Zlanhe('1', Upper, n, r, rwork)
	result.Set(2, (resid/float64(max(1, n)))/ulp)

	//     Compute I - Z'*Z
	golapack.Zlaset(Full, p, p, czero, cone, t)
	if err = goblas.Zherk(Upper, ConjTrans, p, p, -one, z, one, t); err != nil {
		panic(err)
	}

	//     Compute norm( I - Z'*Z ) / ( P*ULP ) .
	resid = golapack.Zlanhe('1', Upper, p, t, rwork)
	result.Set(3, (resid/float64(max(1, p)))/ulp)
}
