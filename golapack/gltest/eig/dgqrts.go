package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dgqrts tests DGGQRF, which computes the GQR factorization of an
// N-by-M matrix A and a N-by-P matrix B: A = Q*R and B = Q*T*Z.
func dgqrts(n, m, p int, a, af, q, r *mat.Matrix, taua *mat.Vector, b, bf, z, t, bwk *mat.Matrix, taub, work *mat.Vector, lwork int, rwork, result *mat.Vector) {
	var anorm, bnorm, one, resid, rogue, ulp, unfl, zero float64
	var err error

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	ulp = golapack.Dlamch(Precision)
	unfl = golapack.Dlamch(SafeMinimum)

	//     Copy the matrix A to the array AF.
	golapack.Dlacpy(Full, n, m, a, af)
	golapack.Dlacpy(Full, n, p, b, bf)

	anorm = math.Max(golapack.Dlange('1', n, m, a, rwork), unfl)
	bnorm = math.Max(golapack.Dlange('1', n, p, b, rwork), unfl)

	//     Factorize the matrices A and B in the arrays AF and BF.
	if err = golapack.Dggqrf(n, m, p, af, taua, bf, taub, work, lwork); err != nil {
		panic(err)
	}

	//     Generate the N-by-N matrix Q
	golapack.Dlaset(Full, n, n, rogue, rogue, q)
	golapack.Dlacpy(Lower, n-1, m, af.Off(1, 0), q.Off(1, 0))
	if err = golapack.Dorgqr(n, n, min(n, m), q, taua, work, lwork); err != nil {
		panic(err)
	}

	//     Generate the P-by-P matrix Z
	golapack.Dlaset(Full, p, p, rogue, rogue, z)
	if n <= p {
		if n > 0 && n < p {
			golapack.Dlacpy(Full, n, p-n, bf, z.Off(p-n, 0))
		}
		if n > 1 {
			golapack.Dlacpy(Lower, n-1, n-1, bf.Off(1, p-n), z.Off(p-n+2-1, p-n))
		}
	} else {
		if p > 1 {
			golapack.Dlacpy(Lower, p-1, p-1, bf.Off(n-p+2-1, 0), z.Off(1, 0))
		}
	}
	if err = golapack.Dorgrq(p, p, min(n, p), z, taub, work, lwork); err != nil {
		panic(err)
	}

	//     Copy R
	golapack.Dlaset(Full, n, m, zero, zero, r)
	golapack.Dlacpy(Upper, n, m, af, r)

	//     Copy T
	golapack.Dlaset(Full, n, p, zero, zero, t)
	if n <= p {
		golapack.Dlacpy(Upper, n, n, bf.Off(0, p-n), t.Off(0, p-n))
	} else {
		golapack.Dlacpy(Full, n-p, p, bf, t)
		golapack.Dlacpy(Upper, p, p, bf.Off(n-p, 0), t.Off(n-p, 0))
	}

	//     Compute R - Q'*A
	if err = goblas.Dgemm(Trans, NoTrans, n, m, n, -one, q, a, one, r); err != nil {
		panic(err)
	}

	//     Compute norm( R - Q'*A ) / ( MAX(M,N)*norm(A)*ULP ) .
	resid = golapack.Dlange('1', n, m, r, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, m, n)))/anorm)/ulp)
	} else {
		result.Set(0, zero)
	}

	//     Compute T*Z - Q'*B
	if err = goblas.Dgemm(NoTrans, NoTrans, n, p, p, one, t, z, zero, bwk); err != nil {
		panic(err)
	}
	if err = goblas.Dgemm(Trans, NoTrans, n, p, n, -one, q, b, one, bwk); err != nil {
		panic(err)
	}

	//     Compute norm( T*Z - Q'*B ) / ( MAX(P,N)*norm(A)*ULP ) .
	resid = golapack.Dlange('1', n, p, bwk, rwork)
	if bnorm > zero {
		result.Set(1, ((resid/float64(max(1, p, n)))/bnorm)/ulp)
	} else {
		result.Set(1, zero)
	}

	//     Compute I - Q'*Q
	golapack.Dlaset(Full, n, n, zero, one, r)
	if err = goblas.Dsyrk(Upper, Trans, n, n, -one, q, one, r); err != nil {
		panic(err)
	}

	//     Compute norm( I - Q'*Q ) / ( N * ULP ) .
	resid = golapack.Dlansy('1', Upper, n, r, rwork)
	result.Set(2, (resid/float64(max(1, n)))/ulp)

	//     Compute I - Z'*Z
	golapack.Dlaset(Full, p, p, zero, one, t)
	if err = goblas.Dsyrk(Upper, Trans, p, p, -one, z, one, t); err != nil {
		panic(err)
	}

	//     Compute norm( I - Z'*Z ) / ( P*ULP ) .
	resid = golapack.Dlansy('1', Upper, p, t, rwork)
	result.Set(3, (resid/float64(max(1, p)))/ulp)
}
