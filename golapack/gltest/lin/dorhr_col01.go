package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorhrcol01 tests DORHR_COL using DLATSQR, DGEMQRT and DORGTSQR.
// Therefore, DLATSQR (part of DGEQR), DGEMQRT (part DGEMQR), DORGTSQR
// have to be tested before this test.
func DorhrCol01(m, n, mb1, nb1, nb2 *int, result *mat.Vector) {
	var testzeros bool
	var anorm, cnorm, dnorm, eps, one, resid, zero float64
	var i, info, j, k, l, lwork, nb1Ub, nb2Ub, nrb int
	srnamt := &gltest.Common.Srnamc.Srnamt
	iseed := make([]int, 4)

	workquery := vf(1)

	zero = 0.0
	one = 1.0

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	//     TEST MATRICES WITH HALF OF MATRIX BEING ZEROS
	testzeros = false

	eps = golapack.Dlamch(Epsilon)
	k = minint(*m, *n)
	l = maxint(*m, *n, 1)

	//     Dynamically allocate local arrays
	// Allocate(A(m, n), Af(m, n), Q(&l, &l), R(m, &l), Rwork(&l), C(m, n), Cf(m, n), D(n, m), Df(n, m))
	a := mf(*m, *n, opts)
	af := mf(*m, *n, opts)
	q := mf(l, l, opts)
	r := mf(*m, l, opts)
	rwork := vf(l)
	c := mf(*m, *n, opts)
	cf := mf(*m, *n, opts)
	d := mf(*n, *m, opts)
	df := mf(*n, *m, opts)

	//     Put random numbers into A and copy to AF
	for j = 1; j <= (*n); j++ {
		golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, m, a.Vector(1-1, j-1))
	}
	if testzeros {
		if (*m) >= 4 {
			for j = 1; j <= (*n); j++ {
				golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr((*m)/2), a.Vector((*m)/4-1, j-1))
			}
		}
	}
	golapack.Dlacpy('F', m, n, a, m, af, m)

	//     Number of row blocks in DLATSQR
	nrb = maxint(1, int(math.Ceil(float64((*m)-(*n))/float64((*mb1)-(*n)))))

	// Allocate(T1(nb1, (*n)*nrb))
	// Allocate(T2(nb2, n))
	// Allocate(Diag(n))
	t1 := mf(*nb1, (*n)*nrb, opts)
	t2 := mf(*nb2, *n, opts)
	diag := vf(*n)

	//     Begin determine LWORK for the array WORK and allocate memory.
	//
	//     DLATSQR requires NB1 to be bounded by N.
	nb1Ub = minint(*nb1, *n)

	//     DGEMQRT requires NB2 to be bounded by N.
	nb2Ub = minint(*nb2, *n)

	golapack.Dlatsqr(m, n, mb1, &nb1Ub, af, m, t1, nb1, workquery, toPtr(-1), &info)
	lwork = int(workquery.Get(0))
	golapack.Dorgtsqr(m, n, mb1, nb1, af, m, t1, nb1, workquery, toPtr(-1), &info)
	lwork = maxint(lwork, int(workquery.Get(0)))

	//     In DGEMQRT, WORK is N*NB2_UB if SIDE = 'L',
	//                or  M*NB2_UB if SIDE = 'R'.
	lwork = maxint(lwork, nb2Ub*(*n), nb2Ub*(*m))

	// Allocate(Work(&lwork))
	work := vf(lwork)

	//     End allocate memory for WORK.
	//
	//
	//     Begin Householder reconstruction routines
	//
	//     Factor the matrix A in the array AF.
	*srnamt = "DLATSQR"
	golapack.Dlatsqr(m, n, mb1, &nb1Ub, af, m, t1, nb1, work, &lwork, &info)

	//     Copy the factor R into the array R.
	*srnamt = "DLACPY"
	golapack.Dlacpy('U', n, n, af, m, r, m)

	//     Reconstruct the orthogonal matrix Q.
	*srnamt = "DORGTSQR"
	golapack.Dorgtsqr(m, n, mb1, nb1, af, m, t1, nb1, work, &lwork, &info)

	//     Perform the Householder reconstruction, the result is stored
	//     the arrays AF and T2.
	*srnamt = "DORHR_COL"
	golapack.DorhrCol(m, n, nb2, af, m, t2, nb2, diag, &info)

	//     Compute the factor R_hr corresponding to the Householder
	//     reconstructed Q_hr and place it in the upper triangle of AF to
	//     match the Q storage format in DGEQRT. R_hr = R_tsqr * S,
	//     this means changing the sign of I-th row of the matrix R_tsqr
	//     according to sign of of I-th diagonal element DIAG(I) of the
	//     matrix S.
	*srnamt = "DLACPY"
	golapack.Dlacpy('U', n, n, r, m, af, m)

	for i = 1; i <= (*n); i++ {
		if diag.Get(i-1) == -one {
			goblas.Dscal(toPtr((*n)+1-i), toPtrf64(-one), af.Vector(i-1, i-1), m)
		}
	}

	//     End Householder reconstruction routines.
	//
	//
	//     Generate the m-by-m matrix Q
	golapack.Dlaset('F', m, m, &zero, &one, q, m)

	*srnamt = "DGEMQRT"
	golapack.Dgemqrt('L', 'N', m, m, &k, &nb2Ub, af, m, t2, nb2, q, m, work, &info)

	//     Copy R
	golapack.Dlaset('F', m, n, &zero, &zero, r, m)

	golapack.Dlacpy('U', m, n, af, m, r, m)

	//     TEST 1
	//     Compute |R - (Q**T)*A| / ( eps * m * |A| ) and store in RESULT(1)
	goblas.Dgemm(Trans, NoTrans, m, n, m, toPtrf64(-one), q, m, a, m, &one, r, m)

	anorm = golapack.Dlange('1', m, n, a, m, rwork)
	resid = golapack.Dlange('1', m, n, r, m, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*float64(maxint(1, *m))*anorm))
	} else {
		result.Set(0, zero)
	}

	//     TEST 2
	//     Compute |I - (Q**T)*Q| / ( eps * m ) and store in RESULT(2)
	golapack.Dlaset('F', m, m, &zero, &one, r, m)
	goblas.Dsyrk(Upper, Trans, m, m, toPtrf64(-one), q, m, &one, r, m)
	resid = golapack.Dlansy('1', 'U', m, r, m, rwork)
	result.Set(1, resid/(eps*float64(maxint(1, *m))))

	//     Generate random m-by-n matrix C
	for j = 1; j <= (*n); j++ {
		golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, m, c.Vector(1-1, j-1))
	}
	cnorm = golapack.Dlange('1', m, n, c, m, rwork)
	golapack.Dlacpy('F', m, n, c, m, cf, m)

	//     Apply Q to C as Q*C = CF
	*srnamt = "DGEMQRT"
	golapack.Dgemqrt('L', 'N', m, n, &k, &nb2Ub, af, m, t2, nb2, cf, m, work, &info)

	//     TEST 3
	//     Compute |CF - Q*C| / ( eps *  m * |C| )
	goblas.Dgemm(NoTrans, NoTrans, m, n, m, toPtrf64(-one), q, m, c, m, &one, cf, m)
	resid = golapack.Dlange('1', m, n, cf, m, rwork)
	if cnorm > zero {
		result.Set(2, resid/(eps*float64(maxint(1, *m))*cnorm))
	} else {
		result.Set(2, zero)
	}

	//     Copy C into CF again
	golapack.Dlacpy('F', m, n, c, m, cf, m)

	//     Apply Q to C as (Q**T)*C = CF
	*srnamt = "DGEMQRT"
	golapack.Dgemqrt('L', 'T', m, n, &k, &nb2Ub, af, m, t2, nb2, cf, m, work, &info)

	//     TEST 4
	//     Compute |CF - (Q**T)*C| / ( eps * m * |C|)
	goblas.Dgemm(Trans, NoTrans, m, n, m, toPtrf64(-one), q, m, c, m, &one, cf, m)
	resid = golapack.Dlange('1', m, n, cf, m, rwork)
	if cnorm > zero {
		result.Set(3, resid/(eps*float64(maxint(1, *m))*cnorm))
	} else {
		result.Set(3, zero)
	}

	//     Generate random n-by-m matrix D and a copy DF
	for j = 1; j <= (*m); j++ {
		golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, n, d.Vector(1-1, j-1))
	}
	dnorm = golapack.Dlange('1', n, m, d, n, rwork)
	golapack.Dlacpy('F', n, m, d, n, df, n)

	//     Apply Q to D as D*Q = DF
	*srnamt = "DGEMQRT"
	golapack.Dgemqrt('R', 'N', n, m, &k, &nb2Ub, af, m, t2, nb2, df, n, work, &info)

	//     TEST 5
	//     Compute |DF - D*Q| / ( eps * m * |D| )
	goblas.Dgemm(NoTrans, NoTrans, n, m, m, toPtrf64(-one), d, n, q, m, &one, df, n)
	resid = golapack.Dlange('1', n, m, df, n, rwork)
	if dnorm > zero {
		result.Set(4, resid/(eps*float64(maxint(1, *m))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy D into DF again
	golapack.Dlacpy('F', n, m, d, n, df, n)

	//     Apply Q to D as D*QT = DF
	*srnamt = "DGEMQRT"
	golapack.Dgemqrt('R', 'T', n, m, &k, &nb2Ub, af, m, t2, nb2, df, n, work, &info)

	//     TEST 6
	//     Compute |DF - D*(Q**T)| / ( eps * m * |D| )
	goblas.Dgemm(NoTrans, Trans, n, m, m, toPtrf64(-one), d, n, q, m, &one, df, n)
	resid = golapack.Dlange('1', n, m, df, n, rwork)
	if dnorm > zero {
		result.Set(5, resid/(eps*float64(maxint(1, *m))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
