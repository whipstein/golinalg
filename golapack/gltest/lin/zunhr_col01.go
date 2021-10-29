package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zunhrZol01 tests ZunhrCol using Zlatsqr, Zgemqrt and Zungtsqr.
// Therefore, Zlatsqr (part of ZGEQR), Zgemqrt (part ZGEMQR), Zungtsqr
// have to be tested before this test.
func zunhrCol01(m, n, mb1, nb1, nb2 int, result *mat.Vector) {
	var testzeros bool
	var cone, czero complex128
	var anorm, cnorm, dnorm, eps, resid, zero float64
	var i, j, k, l, lwork, nb1Ub, nb2Ub, nrb int
	var err error

	workquery := cvf(1)
	iseed := make([]int, 4)

	zero = 0.0
	cone = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

	srnamt := &gltest.Common.Srnamc.Srnamt

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	//     TEST MATRICES WITH HALF OF MATRIX BEING ZEROS
	testzeros = false

	eps = golapack.Dlamch(Epsilon)
	k = min(m, n)
	l = max(m, n, 1)

	//     Dynamically allocate local arrays
	a := cmf(m, n, opts)
	af := cmf(m, n, opts)
	q := cmf(l, l, opts)
	r := cmf(m, l, opts)
	rwork := vf(l)
	c := cmf(m, n, opts)
	cf := cmf(m, n, opts)
	d := cmf(n, m, opts)
	df := cmf(n, m, opts)

	//     Put random numbers into a and copy to AF
	for j = 1; j <= n; j++ {
		golapack.Zlarnv(2, &iseed, m, a.CVector(0, j-1))
	}
	if testzeros {
		if m >= 4 {
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(2, &iseed, m/2, a.CVector(m/4-1, j-1))
			}
		}
	}
	golapack.Zlacpy(Full, m, n, a, af)

	//     Number of row blocks in Zlatsqr
	nrb = max(1, int(math.Ceil(float64(m-n)/float64(mb1-n))))

	t1 := cmf(nb1, n*nrb, opts)
	t2 := cmf(nb2, n, opts)
	diag := cvf(n)

	//     Begin determine LWORK for the array WORK and allocate memory.
	//
	//     Zlatsqr requires NB1 to be bounded by N.
	nb1Ub = min(nb1, n)

	//     Zgemqrt requires NB2 to be bounded by N.
	nb2Ub = min(nb2, n)

	if err = golapack.Zlatsqr(m, n, mb1, nb1Ub, af, t1, workquery, -1); err != nil {
		panic(err)
	}
	lwork = int(workquery.GetRe(0))
	if err = golapack.Zungtsqr(m, n, mb1, nb1, af, t1, workquery, -1); err != nil {
		panic(err)
	}
	lwork = max(lwork, int(workquery.GetRe(0)))

	//     In Zgemqrt, WORK is N*NB2_UB if SIDE = 'L',
	//                or  M*NB2_UB if SIDE = 'R'.
	lwork = max(lwork, nb2Ub*n, nb2Ub*m)

	work := cvf(lwork)

	//     End allocate memory for WORK.
	//
	//
	//     Begin Householder reconstruction routines
	//
	//     Factor the matrix a in the array AF.
	*srnamt = "Zlatsqr"
	if err = golapack.Zlatsqr(m, n, mb1, nb1Ub, af, t1, work, lwork); err != nil {
		panic(err)
	}

	//     Copy the factor R into the array R.
	*srnamt = "Zlacpy"
	golapack.Zlacpy(Upper, m, n, af, r)

	//     Reconstruct the orthogonal matrix Q.
	*srnamt = "Zungtsqr"
	if err = golapack.Zungtsqr(m, n, mb1, nb1, af, t1, work, lwork); err != nil {
		panic(err)
	}

	//     Perform the Householder reconstruction, the result is stored
	//     the arrays AF and T2.
	*srnamt = "ZunhrCol"
	if err = golapack.ZunhrCol(m, n, nb2, af, t2, diag); err != nil {
		panic(err)
	}

	//     Compute the factor R_hr corresponding to the Householder
	//     reconstructed Q_hr and place it in the upper triangle of AF to
	//     match the Q storage format in ZGEQRT. R_hr = R_tsqr * S,
	//     this means changing the sign of I-th row of the matrix R_tsqr
	//     according to sign of of I-th diagonal element DIAG(I) of the
	//     matrix S.
	*srnamt = "Zlacpy"
	golapack.Zlacpy(Upper, m, n, r, af)

	for i = 1; i <= n; i++ {
		if diag.Get(i-1) == -cone {
			goblas.Zscal(n+1-i, -cone, af.CVector(i-1, i-1, m))
		}
	}

	//     End Householder reconstruction routines.
	//
	//
	//     Generate the m-by-m matrix Q
	golapack.Zlaset(Full, m, m, czero, cone, q)

	*srnamt = "Zgemqrt"
	if err = golapack.Zgemqrt(Left, NoTrans, m, m, k, nb2Ub, af, t2, q, work); err != nil {
		panic(err)
	}

	//     Copy R
	golapack.Zlaset(Full, m, n, czero, czero, r)

	golapack.Zlacpy(Upper, m, n, af, r)

	//     TEST 1
	//     Compute |R - (Q**H)*a| / ( eps * m * |a| ) and store in RESULT(1)
	if err = goblas.Zgemm(ConjTrans, NoTrans, m, n, m, -cone, q, a, cone, r); err != nil {
		panic(err)
	}

	anorm = golapack.Zlange('1', m, n, a, rwork)
	resid = golapack.Zlange('1', m, n, r, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*float64(max(1, m))*anorm))
	} else {
		result.Set(0, zero)
	}

	//     TEST 2
	//     Compute |I - (Q**H)*Q| / ( eps * m ) and store in RESULT(2)
	golapack.Zlaset(Full, m, m, czero, cone, r)
	if err = goblas.Zherk(Upper, ConjTrans, m, m, real(-cone), q, real(cone), r); err != nil {
		panic(err)
	}
	resid = golapack.Zlansy('1', Upper, m, r, rwork)
	result.Set(1, resid/(eps*float64(max(1, m))))

	//     Generate random m-by-n matrix C
	for j = 1; j <= n; j++ {
		golapack.Zlarnv(2, &iseed, m, c.CVector(0, j-1))
	}
	cnorm = golapack.Zlange('1', m, n, c, rwork)
	golapack.Zlacpy(Full, m, n, c, cf)

	//     Apply Q to C as Q*C = CF
	*srnamt = "Zgemqrt"
	if err = golapack.Zgemqrt(Left, NoTrans, m, n, k, nb2Ub, af, t2, cf, work); err != nil {
		panic(err)
	}

	//     TEST 3
	//     Compute |CF - Q*C| / ( eps *  m * |C| )
	if err = goblas.Zgemm(NoTrans, NoTrans, m, n, m, -cone, q, c, cone, cf); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', m, n, cf, rwork)
	if cnorm > zero {
		result.Set(2, resid/(eps*float64(max(1, m))*cnorm))
	} else {
		result.Set(2, zero)
	}

	//     Copy C into CF again
	golapack.Zlacpy(Full, m, n, c, cf)

	//     Apply Q to C as (Q**H)*C = CF
	*srnamt = "Zgemqrt"
	if err = golapack.Zgemqrt(Left, ConjTrans, m, n, k, nb2Ub, af, t2, cf, work); err != nil {
		panic(err)
	}

	//     TEST 4
	//     Compute |CF - (Q**H)*C| / ( eps * m * |C|)
	if err = goblas.Zgemm(ConjTrans, NoTrans, m, n, m, -cone, q, c, cone, cf); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', m, n, cf, rwork)
	if cnorm > zero {
		result.Set(3, resid/(eps*float64(max(1, m))*cnorm))
	} else {
		result.Set(3, zero)
	}

	//     Generate random n-by-m matrix D and a copy DF
	for j = 1; j <= m; j++ {
		golapack.Zlarnv(2, &iseed, n, d.CVector(0, j-1))
	}
	dnorm = golapack.Zlange('1', n, m, d, rwork)
	golapack.Zlacpy(Full, n, m, d, df)

	//     Apply Q to D as D*Q = DF
	*srnamt = "Zgemqrt"
	if err = golapack.Zgemqrt(Right, NoTrans, n, m, k, nb2Ub, af, t2, df, work); err != nil {
		panic(err)
	}

	//     TEST 5
	//     Compute |DF - D*Q| / ( eps * m * |D| )
	if err = goblas.Zgemm(NoTrans, NoTrans, n, m, m, -cone, d, q, cone, df); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', n, m, df, rwork)
	if dnorm > zero {
		result.Set(4, resid/(eps*float64(max(1, m))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy D into DF again
	golapack.Zlacpy(Full, n, m, d, df)

	//     Apply Q to D as D*QT = DF
	*srnamt = "Zgemqrt"
	if err = golapack.Zgemqrt(Right, ConjTrans, n, m, k, nb2Ub, af, t2, df, work); err != nil {
		panic(err)
	}

	//     TEST 6
	//     Compute |DF - D*(Q**H)| / ( eps * m * |D| )
	if err = goblas.Zgemm(NoTrans, ConjTrans, n, m, m, -cone, d, q, cone, df); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', n, m, df, rwork)
	if dnorm > zero {
		result.Set(5, resid/(eps*float64(max(1, m))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
