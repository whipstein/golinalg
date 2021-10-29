package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dgrqts tests DGGRQF, which computes the GRQ factorization of an
// M-by-N matrix A and a P-by-N matrix B: A = R*Q and B = Z*T*Q.
func dgrqts(m, p, n int, a, af, q, r *mat.Matrix, taua *mat.Vector, b, bf, z, t, bwk *mat.Matrix, taub, work *mat.Vector, lwork int, rwork, result *mat.Vector) {
	var anorm, bnorm, one, resid, rogue, ulp, unfl, zero float64
	var err error

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	ulp = golapack.Dlamch(Precision)
	unfl = golapack.Dlamch(SafeMinimum)

	//     Copy the matrix A to the array AF.
	golapack.Dlacpy(Full, m, n, a, af)
	golapack.Dlacpy(Full, p, n, b, bf)

	anorm = math.Max(golapack.Dlange('1', m, n, a, rwork), unfl)
	bnorm = math.Max(golapack.Dlange('1', p, n, b, rwork), unfl)

	//     Factorize the matrices A and B in the arrays AF and BF.
	if err = golapack.Dggrqf(m, p, n, af, taua, bf, taub, work, lwork); err != nil {
		panic(err)
	}

	//     Generate the N-by-N matrix Q
	golapack.Dlaset(Full, n, n, rogue, rogue, q)
	if m <= n {
		if m > 0 && m < n {
			golapack.Dlacpy(Full, m, n-m, af, q.Off(n-m, 0))
		}
		if m > 1 {
			golapack.Dlacpy(Lower, m-1, m-1, af.Off(1, n-m), q.Off(n-m+2-1, n-m))
		}
	} else {
		if n > 1 {
			golapack.Dlacpy(Lower, n-1, n-1, af.Off(m-n+2-1, 0), q.Off(1, 0))
		}
	}
	if err = golapack.Dorgrq(n, n, min(m, n), q, taua, work, lwork); err != nil {
		panic(err)
	}

	//     Generate the P-by-P matrix Z
	golapack.Dlaset(Full, p, p, rogue, rogue, z)
	if p > 1 {
		golapack.Dlacpy(Lower, p-1, n, bf.Off(1, 0), z.Off(1, 0))
	}
	if err = golapack.Dorgqr(p, p, min(p, n), z, taub, work, lwork); err != nil {
		panic(err)
	}

	//     Copy R
	golapack.Dlaset(Full, m, n, zero, zero, r)
	if m <= n {
		golapack.Dlacpy(Upper, m, m, af.Off(0, n-m), r.Off(0, n-m))
	} else {
		golapack.Dlacpy(Full, m-n, n, af, r)
		golapack.Dlacpy(Upper, n, n, af.Off(m-n, 0), r.Off(m-n, 0))
	}

	//     Copy T
	golapack.Dlaset(Full, p, n, zero, zero, t)
	golapack.Dlacpy(Upper, p, n, bf, t)

	//     Compute R - A*Q'
	if err = goblas.Dgemm(NoTrans, Trans, m, n, n, -one, a, q, one, r); err != nil {
		panic(err)
	}

	//     Compute norm( R - A*Q' ) / ( math.Max(M,N)*norm(A)*ULP ) .
	resid = golapack.Dlange('1', m, n, r, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, m, n)))/anorm)/ulp)
	} else {
		result.Set(0, zero)
	}

	//     Compute T*Q - Z'*B
	if err = goblas.Dgemm(Trans, NoTrans, p, n, p, one, z, b, zero, bwk); err != nil {
		panic(err)
	}
	if err = goblas.Dgemm(NoTrans, NoTrans, p, n, n, one, t, q, -one, bwk); err != nil {
		panic(err)
	}

	//     Compute norm( T*Q - Z'*B ) / ( math.Max(P,N)*norm(A)*ULP ) .
	resid = golapack.Dlange('1', p, n, bwk, rwork)
	if bnorm > zero {
		result.Set(1, ((resid/float64(max(1, p, m)))/bnorm)/ulp)
	} else {
		result.Set(1, zero)
	}

	//     Compute I - Q*Q'
	golapack.Dlaset(Full, n, n, zero, one, r)
	if err = goblas.Dsyrk(Upper, NoTrans, n, n, -one, q, one, r); err != nil {
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
