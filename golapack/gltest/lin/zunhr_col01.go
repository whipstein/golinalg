package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunhrcol01 tests ZUNHR_COL using ZLATSQR, ZGEMQRT and ZUNGTSQR.
// Therefore, ZLATSQR (part of ZGEQR), ZGEMQRT (part ZGEMQR), ZUNGTSQR
// have to be tested before this test.
func Zunhrcol01(m, n, mb1, nb1, nb2 *int, result *mat.Vector) {
	var testzeros bool
	var cone, czero complex128
	var anorm, cnorm, dnorm, eps, resid, zero float64
	var i, info, j, k, l, lwork, nb1Ub, nb2Ub, nrb int

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
	k = minint(*m, *n)
	l = maxint(*m, *n, 1)

	//     Dynamically allocate local arrays
	a := cmf(*m, *n, opts)
	af := cmf(*m, *n, opts)
	q := cmf(l, l, opts)
	r := cmf(*m, l, opts)
	rwork := vf(l)
	c := cmf(*m, *n, opts)
	cf := cmf(*m, *n, opts)
	d := cmf(*n, *m, opts)
	df := cmf(*n, *m, opts)

	//     Put random numbers into a and copy to AF
	for j = 1; j <= (*n); j++ {
		golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, m, a.CVector(0, j-1))
	}
	if testzeros {
		if (*m) >= 4 {
			for j = 1; j <= (*n); j++ {
				golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr((*m)/2), a.CVector((*m)/4-1, j-1))
			}
		}
	}
	golapack.Zlacpy('F', m, n, a, m, af, m)

	//     Number of row blocks in ZLATSQR
	nrb = maxint(1, int(math.Ceil(float64((*m)-(*n))/float64((*mb1)-(*n)))))

	t1 := cmf(*nb1, (*n)*nrb, opts)
	t2 := cmf(*nb2, *n, opts)
	diag := cvf(*n)

	//     Begin determine LWORK for the array WORK and allocate memory.
	//
	//     ZLATSQR requires NB1 to be bounded by N.
	nb1Ub = minint(*nb1, *n)

	//     ZGEMQRT requires NB2 to be bounded by N.
	nb2Ub = minint(*nb2, *n)

	golapack.Zlatsqr(m, n, mb1, &nb1Ub, af, m, t1, nb1, workquery, toPtr(-1), &info)
	lwork = int(workquery.GetRe(0))
	golapack.Zungtsqr(m, n, mb1, nb1, af, m, t1, nb1, workquery, toPtr(-1), &info)
	lwork = maxint(lwork, int(workquery.GetRe(0)))

	//     In ZGEMQRT, WORK is N*NB2_UB if SIDE = 'L',
	//                or  M*NB2_UB if SIDE = 'R'.
	lwork = maxint(lwork, nb2Ub*(*n), nb2Ub*(*m))

	work := cvf(lwork)

	//     End allocate memory for WORK.
	//
	//
	//     Begin Householder reconstruction routines
	//
	//     Factor the matrix a in the array AF.
	*srnamt = "ZLATSQR"
	golapack.Zlatsqr(m, n, mb1, &nb1Ub, af, m, t1, nb1, work, &lwork, &info)

	//     Copy the factor R into the array R.
	*srnamt = "ZLACPY"
	golapack.Zlacpy('U', m, n, af, m, r, m)

	//     Reconstruct the orthogonal matrix Q.
	*srnamt = "ZUNGTSQR"
	golapack.Zungtsqr(m, n, mb1, nb1, af, m, t1, nb1, work, &lwork, &info)

	//     Perform the Householder reconstruction, the result is stored
	//     the arrays AF and T2.
	*srnamt = "ZUNHR_COL"
	golapack.Zunhrcol(m, n, nb2, af, m, t2, nb2, diag, &info)

	//     Compute the factor R_hr corresponding to the Householder
	//     reconstructed Q_hr and place it in the upper triangle of AF to
	//     match the Q storage format in ZGEQRT. R_hr = R_tsqr * S,
	//     this means changing the sign of I-th row of the matrix R_tsqr
	//     according to sign of of I-th diagonal element DIAG(I) of the
	//     matrix S.
	*srnamt = "ZLACPY"
	golapack.Zlacpy('U', m, n, r, m, af, m)

	for i = 1; i <= (*n); i++ {
		if diag.Get(i-1) == -cone {
			goblas.Zscal(toPtr((*n)+1-i), toPtrc128(-cone), af.CVector(i-1, i-1), m)
		}
	}

	//     End Householder reconstruction routines.
	//
	//
	//     Generate the m-by-m matrix Q
	golapack.Zlaset('F', m, m, &czero, &cone, q, m)

	*srnamt = "ZGEMQRT"
	golapack.Zgemqrt('L', 'N', m, m, &k, &nb2Ub, af, m, t2, nb2, q, m, work, &info)

	//     Copy R
	golapack.Zlaset('F', m, n, &czero, &czero, r, m)

	golapack.Zlacpy('U', m, n, af, m, r, m)

	//     TEST 1
	//     Compute |R - (Q**H)*a| / ( eps * m * |a| ) and store in RESULT(1)
	goblas.Zgemm(ConjTrans, NoTrans, m, n, m, toPtrc128(-cone), q, m, a, m, &cone, r, m)

	anorm = golapack.Zlange('1', m, n, a, m, rwork)
	resid = golapack.Zlange('1', m, n, r, m, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*float64(maxint(1, *m))*anorm))
	} else {
		result.Set(0, zero)
	}

	//     TEST 2
	//     Compute |I - (Q**H)*Q| / ( eps * m ) and store in RESULT(2)
	golapack.Zlaset('F', m, m, &czero, &cone, r, m)
	goblas.Zherk(Upper, ConjTrans, m, m, toPtrf64(real(-cone)), q, m, toPtrf64(real(cone)), r, m)
	resid = golapack.Zlansy('1', 'U', m, r, m, rwork)
	result.Set(1, resid/(eps*float64(maxint(1, *m))))

	//     Generate random m-by-n matrix C
	for j = 1; j <= (*n); j++ {
		golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, m, c.CVector(0, j-1))
	}
	cnorm = golapack.Zlange('1', m, n, c, m, rwork)
	golapack.Zlacpy('F', m, n, c, m, cf, m)

	//     Apply Q to C as Q*C = CF
	*srnamt = "ZGEMQRT"
	golapack.Zgemqrt('L', 'N', m, n, &k, &nb2Ub, af, m, t2, nb2, cf, m, work, &info)

	//     TEST 3
	//     Compute |CF - Q*C| / ( eps *  m * |C| )
	goblas.Zgemm(NoTrans, NoTrans, m, n, m, toPtrc128(-cone), q, m, c, m, &cone, cf, m)
	resid = golapack.Zlange('1', m, n, cf, m, rwork)
	if cnorm > zero {
		result.Set(2, resid/(eps*float64(maxint(1, *m))*cnorm))
	} else {
		result.Set(2, zero)
	}

	//     Copy C into CF again
	golapack.Zlacpy('F', m, n, c, m, cf, m)

	//     Apply Q to C as (Q**H)*C = CF
	*srnamt = "ZGEMQRT"
	golapack.Zgemqrt('L', 'C', m, n, &k, &nb2Ub, af, m, t2, nb2, cf, m, work, &info)

	//     TEST 4
	//     Compute |CF - (Q**H)*C| / ( eps * m * |C|)
	goblas.Zgemm(ConjTrans, NoTrans, m, n, m, toPtrc128(-cone), q, m, c, m, &cone, cf, m)
	resid = golapack.Zlange('1', m, n, cf, m, rwork)
	if cnorm > zero {
		result.Set(3, resid/(eps*float64(maxint(1, *m))*cnorm))
	} else {
		result.Set(3, zero)
	}

	//     Generate random n-by-m matrix D and a copy DF
	for j = 1; j <= (*m); j++ {
		golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, n, d.CVector(0, j-1))
	}
	dnorm = golapack.Zlange('1', n, m, d, n, rwork)
	golapack.Zlacpy('F', n, m, d, n, df, n)

	//     Apply Q to D as D*Q = DF
	*srnamt = "ZGEMQRT"
	golapack.Zgemqrt('R', 'N', n, m, &k, &nb2Ub, af, m, t2, nb2, df, n, work, &info)

	//     TEST 5
	//     Compute |DF - D*Q| / ( eps * m * |D| )
	goblas.Zgemm(NoTrans, NoTrans, n, m, m, toPtrc128(-cone), d, n, q, m, &cone, df, n)
	resid = golapack.Zlange('1', n, m, df, n, rwork)
	if dnorm > zero {
		result.Set(4, resid/(eps*float64(maxint(1, *m))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy D into DF again
	golapack.Zlacpy('F', n, m, d, n, df, n)

	//     Apply Q to D as D*QT = DF
	*srnamt = "ZGEMQRT"
	golapack.Zgemqrt('R', 'C', n, m, &k, &nb2Ub, af, m, t2, nb2, df, n, work, &info)

	//     TEST 6
	//     Compute |DF - D*(Q**H)| / ( eps * m * |D| )
	goblas.Zgemm(NoTrans, ConjTrans, n, m, m, toPtrc128(-cone), d, n, q, m, &cone, df, n)
	resid = golapack.Zlange('1', n, m, df, n, rwork)
	if dnorm > zero {
		result.Set(5, resid/(eps*float64(maxint(1, *m))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
