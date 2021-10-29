package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zgqrts tests ZGGQRF, which computes the GQR factorization of an
// N-by-M matrix A and a N-by-P matrix B: A = Q*R and B = Q*T*Z.
func zgqrts(n, m, p int, a, af, q, r *mat.CMatrix, taua *mat.CVector, b, bf, z, t, bwk *mat.CMatrix, taub, work *mat.CVector, lwork int, rwork, result *mat.Vector) {
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
	golapack.Zlacpy(Full, n, m, a, af)
	golapack.Zlacpy(Full, n, p, b, bf)

	anorm = math.Max(golapack.Zlange('1', n, m, a, rwork), unfl)
	bnorm = math.Max(golapack.Zlange('1', n, p, b, rwork), unfl)

	//     Factorize the matrices A and B in the arrays AF and BF.
	if err = golapack.Zggqrf(n, m, p, af, taua, bf, taub, work, lwork); err != nil {
		panic(err)
	}

	//     Generate the N-by-N matrix Q
	golapack.Zlaset(Full, n, n, crogue, crogue, q)
	golapack.Zlacpy(Lower, n-1, m, af.Off(1, 0), q.Off(1, 0))
	if err = golapack.Zungqr(n, n, min(n, m), q, taua, work, lwork); err != nil {
		panic(err)
	}

	//     Generate the P-by-P matrix Z
	golapack.Zlaset(Full, p, p, crogue, crogue, z)
	if n <= p {
		if n > 0 && n < p {
			golapack.Zlacpy(Full, n, p-n, bf, z.Off(p-n, 0))
		}
		if n > 1 {
			golapack.Zlacpy(Lower, n-1, n-1, bf.Off(1, p-n), z.Off(p-n+2-1, p-n))
		}
	} else {
		if p > 1 {
			golapack.Zlacpy(Lower, p-1, p-1, bf.Off(n-p+2-1, 0), z.Off(1, 0))
		}
	}
	if err = golapack.Zungrq(p, p, min(n, p), z, taub, work, lwork); err != nil {
		panic(err)
	}

	//     Copy R
	golapack.Zlaset(Full, n, m, czero, czero, r)
	golapack.Zlacpy(Upper, n, m, af, r)

	//     Copy T
	golapack.Zlaset(Full, n, p, czero, czero, t)
	if n <= p {
		golapack.Zlacpy(Upper, n, n, bf.Off(0, p-n), t.Off(0, p-n))
	} else {
		golapack.Zlacpy(Full, n-p, p, bf, t)
		golapack.Zlacpy(Upper, p, p, bf.Off(n-p, 0), t.Off(n-p, 0))
	}

	//     Compute R - Q'*A
	if err = goblas.Zgemm(ConjTrans, NoTrans, n, m, n, -cone, q, a, cone, r); err != nil {
		panic(err)
	}

	//     Compute norm( R - Q'*A ) / ( max(M,N)*norm(A)*ULP ) .
	resid = golapack.Zlange('1', n, m, r, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, m, n)))/anorm)/ulp)
	} else {
		result.Set(0, zero)
	}

	//     Compute T*Z - Q'*B
	if err = goblas.Zgemm(NoTrans, NoTrans, n, p, p, cone, t, z, czero, bwk); err != nil {
		panic(err)
	}
	if err = goblas.Zgemm(ConjTrans, NoTrans, n, p, n, -cone, q, b, cone, bwk); err != nil {
		panic(err)
	}

	//     Compute norm( T*Z - Q'*B ) / ( max(P,N)*norm(A)*ULP ) .
	resid = golapack.Zlange('1', n, p, bwk, rwork)
	if bnorm > zero {
		result.Set(1, ((resid/float64(max(1, p, n)))/bnorm)/ulp)
	} else {
		result.Set(1, zero)
	}

	//     Compute I - Q'*Q
	golapack.Zlaset(Full, n, n, czero, cone, r)
	if err = goblas.Zherk(Upper, ConjTrans, n, n, -one, q, one, r); err != nil {
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
