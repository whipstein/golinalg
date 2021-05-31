package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtsqr01 tests DGEQR , DGELQ, DGEMLQ and DGEMQR.
func Dtsqr01(tssw byte, m, n, mb, nb *int, result *mat.Vector) {
	var testzeros, ts bool
	var anorm, cnorm, dnorm, eps, one, resid, zero float64
	var info, j, k, l, lwork, mnb, tsize int
	srnamt := &gltest.Common.Srnamc.Srnamt
	iseed := make([]int, 4)

	tquery := vf(5)
	workquery := vf(1)

	zero = 0.0
	one = 1.0

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	//     TEST TALL SKINNY OR SHORT WIDE
	ts = tssw == 'T'

	//     TEST MATRICES WITH HALF OF MATRIX BEING ZEROS
	testzeros = false

	eps = golapack.Dlamch(Epsilon)
	k = minint(*m, *n)
	l = maxint(*m, *n, 1)
	mnb = maxint(*mb, *nb)
	lwork = maxint(3, l) * mnb

	//     Dynamically allocate local arrays
	// Allocate(A(m, n), Af(m, n), Q(&l, &l), R(m, &l), Rwork(&l), C(m, n), Cf(m, n), D(n, m), Df(n, m), Lq(&l, n))
	a := mf(*m, *n, opts)
	af := mf(*m, *n, opts)
	q := mf(l, l, opts)
	r := mf(*m, l, opts)
	rwork := vf(l)
	c := mf(*m, *n, opts)
	cf := mf(*m, *n, opts)
	d := mf(*n, *m, opts)
	df := mf(*n, *m, opts)
	lq := mf(l, *n, opts)

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

	if ts {
		//     Factor the matrix A in the array AF.
		golapack.Dgeqr(m, n, af, m, tquery, toPtr(-1), workquery, toPtr(-1), &info)
		tsize = int(tquery.Get(0))
		lwork = int(workquery.Get(0))
		golapack.Dgemqr('L', 'N', m, m, &k, af, m, tquery, &tsize, cf, m, workquery, toPtr(-1), &info)
		lwork = maxint(lwork, int(workquery.Get(0)))
		golapack.Dgemqr('L', 'N', m, n, &k, af, m, tquery, &tsize, cf, m, workquery, toPtr(-1), &info)
		lwork = maxint(lwork, int(workquery.Get(0)))
		golapack.Dgemqr('L', 'T', m, n, &k, af, m, tquery, &tsize, cf, m, workquery, toPtr(-1), &info)
		lwork = maxint(lwork, int(workquery.Get(0)))
		golapack.Dgemqr('R', 'N', n, m, &k, af, m, tquery, &tsize, df, n, workquery, toPtr(-1), &info)
		lwork = maxint(lwork, int(workquery.Get(0)))
		golapack.Dgemqr('R', 'T', n, m, &k, af, m, tquery, &tsize, df, n, workquery, toPtr(-1), &info)
		lwork = maxint(lwork, int(workquery.Get(0)))

		// Allocate(T(&tsize))
		// Allocate(Work(&lwork))
		t := vf(tsize)
		work := vf(lwork)

		*srnamt = "DGEQR"
		golapack.Dgeqr(m, n, af, m, t, &tsize, work, &lwork, &info)

		//     Generate the m-by-m matrix Q
		q.UpdateRows(*m)
		golapack.Dlaset('F', m, m, &zero, &one, q, m)
		*srnamt = "DGEMQR"
		golapack.Dgemqr('L', 'N', m, m, &k, af, m, t, &tsize, q, m, work, &lwork, &info)

		//     Copy R
		golapack.Dlaset('F', m, n, &zero, &zero, r, m)
		golapack.Dlacpy('U', m, n, af, m, r, m)

		//     Compute |R - Q'*A| / |A| and store in RESULT(1)
		goblas.Dgemm(Trans, NoTrans, m, n, m, toPtrf64(-one), q, m, a, m, &one, r, m)
		anorm = golapack.Dlange('1', m, n, a, m, rwork)
		resid = golapack.Dlange('1', m, n, r, m, rwork)
		if anorm > zero {
			result.Set(0, resid/(eps*float64(maxint(1, *m))*anorm))
		} else {
			result.Set(0, zero)
		}

		//     Compute |I - Q'*Q| and store in RESULT(2)
		golapack.Dlaset('F', m, m, &zero, &one, r, m)
		goblas.Dsyrk(Upper, ConjTrans, m, m, toPtrf64(-one), q, m, &one, r, m)
		resid = golapack.Dlansy('1', 'U', m, r, m, rwork)
		result.Set(1, resid/(eps*float64(maxint(1, *m))))

		//     Generate random m-by-n matrix C and a copy CF
		for j = 1; j <= (*n); j++ {
			golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, m, c.Vector(1-1, j-1))
		}
		cnorm = golapack.Dlange('1', m, n, c, m, rwork)
		golapack.Dlacpy('F', m, n, c, m, cf, m)

		//     Apply Q to C as Q*C
		*srnamt = "DGEMQR"
		golapack.Dgemqr('L', 'N', m, n, &k, af, m, t, &tsize, cf, m, work, &lwork, &info)

		//     Compute |Q*C - Q*C| / |C|
		goblas.Dgemm(NoTrans, NoTrans, m, n, m, toPtrf64(-one), q, m, c, m, &one, cf, m)
		resid = golapack.Dlange('1', m, n, cf, m, rwork)
		if cnorm > zero {
			result.Set(2, resid/(eps*float64(maxint(1, *m))*cnorm))
		} else {
			result.Set(2, zero)
		}

		//     Copy C into CF again
		golapack.Dlacpy('F', m, n, c, m, cf, m)

		//     Apply Q to C as QT*C
		*srnamt = "DGEMQR"
		golapack.Dgemqr('L', 'T', m, n, &k, af, m, t, &tsize, cf, m, work, &lwork, &info)

		//     Compute |QT*C - QT*C| / |C|
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

		//     Apply Q to D as D*Q
		*srnamt = "DGEMQR"
		golapack.Dgemqr('R', 'N', n, m, &k, af, m, t, &tsize, df, n, work, &lwork, &info)

		//     Compute |D*Q - D*Q| / |D|
		goblas.Dgemm(NoTrans, NoTrans, n, m, m, toPtrf64(-one), d, n, q, m, &one, df, n)
		resid = golapack.Dlange('1', n, m, df, n, rwork)
		if dnorm > zero {
			result.Set(4, resid/(eps*float64(maxint(1, *m))*dnorm))
		} else {
			result.Set(4, zero)
		}

		//     Copy D into DF again
		golapack.Dlacpy('F', n, m, d, n, df, n)

		//     Apply Q to D as D*QT
		golapack.Dgemqr('R', 'T', n, m, &k, af, m, t, &tsize, df, n, work, &lwork, &info)

		//     Compute |D*QT - D*QT| / |D|
		goblas.Dgemm(NoTrans, Trans, n, m, m, toPtrf64(-one), d, n, q, m, &one, df, n)
		resid = golapack.Dlange('1', n, m, df, n, rwork)
		if cnorm > zero {
			result.Set(5, resid/(eps*float64(maxint(1, *m))*dnorm))
		} else {
			result.Set(5, zero)
		}

		//     Short and wide
	} else {
		golapack.Dgelq(m, n, af, m, tquery, toPtr(-1), workquery, toPtr(-1), &info)
		tsize = int(tquery.Get(0))
		lwork = int(workquery.Get(0))
		golapack.Dgemlq('R', 'N', n, n, &k, af, m, tquery, &tsize, q, n, workquery, toPtr(-1), &info)
		lwork = maxint(lwork, int(workquery.Get(0)))
		golapack.Dgemlq('L', 'N', n, m, &k, af, m, tquery, &tsize, df, n, workquery, toPtr(-1), &info)
		lwork = maxint(lwork, int(workquery.Get(0)))
		golapack.Dgemlq('L', 'T', n, m, &k, af, m, tquery, &tsize, df, n, workquery, toPtr(-1), &info)
		lwork = maxint(lwork, int(workquery.Get(0)))
		golapack.Dgemlq('R', 'N', m, n, &k, af, m, tquery, &tsize, cf, m, workquery, toPtr(-1), &info)
		lwork = maxint(lwork, int(workquery.Get(0)))
		golapack.Dgemlq('R', 'T', m, n, &k, af, m, tquery, &tsize, cf, m, workquery, toPtr(-1), &info)
		lwork = maxint(lwork, int(workquery.Get(0)))

		// Allocate(T(&tsize))
		// Allocate(Work(&lwork))
		t := vf(tsize)
		work := vf(lwork)

		*srnamt = "DGELQ"
		golapack.Dgelq(m, n, af, m, t, &tsize, work, &lwork, &info)

		//     Generate the n-by-n matrix Q
		q.UpdateRows(*n)
		golapack.Dlaset('F', n, n, &zero, &one, q, n)
		*srnamt = "DGEMLQ"
		golapack.Dgemlq('R', 'N', n, n, &k, af, m, t, &tsize, q, n, work, &lwork, &info)

		//     Copy R
		golapack.Dlaset('F', m, n, &zero, &zero, lq, &l)
		golapack.Dlacpy('L', m, n, af, m, lq, &l)

		//     Compute |L - A*Q'| / |A| and store in RESULT(1)
		goblas.Dgemm(NoTrans, Trans, m, n, n, toPtrf64(-one), a, m, q, n, &one, lq, &l)
		anorm = golapack.Dlange('1', m, n, a, m, rwork)
		resid = golapack.Dlange('1', m, n, lq, &l, rwork)
		if anorm > zero {
			result.Set(0, resid/(eps*float64(maxint(1, *n))*anorm))
		} else {
			result.Set(0, zero)
		}

		//     Compute |I - Q'*Q| and store in RESULT(2)
		golapack.Dlaset('F', n, n, &zero, &one, lq, &l)
		goblas.Dsyrk(Upper, ConjTrans, n, n, toPtrf64(-one), q, n, &one, lq, &l)
		resid = golapack.Dlansy('1', 'U', n, lq, &l, rwork)
		result.Set(1, resid/(eps*float64(maxint(1, *n))))

		//     Generate random m-by-n matrix C and a copy CF
		for j = 1; j <= (*m); j++ {
			golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, n, d.Vector(1-1, j-1))
		}
		dnorm = golapack.Dlange('1', n, m, d, n, rwork)
		golapack.Dlacpy('F', n, m, d, n, df, n)

		//     Apply Q to C as Q*C
		golapack.Dgemlq('L', 'N', n, m, &k, af, m, t, &tsize, df, n, work, &lwork, &info)

		//     Compute |Q*D - Q*D| / |D|
		goblas.Dgemm(NoTrans, NoTrans, n, m, n, toPtrf64(-one), q, n, d, n, &one, df, n)
		resid = golapack.Dlange('1', n, m, df, n, rwork)
		if dnorm > zero {
			result.Set(2, resid/(eps*float64(maxint(1, *n))*dnorm))
		} else {
			result.Set(2, zero)
		}

		//     Copy D into DF again
		golapack.Dlacpy('F', n, m, d, n, df, n)

		//     Apply Q to D as QT*D
		golapack.Dgemlq('L', 'T', n, m, &k, af, m, t, &tsize, df, n, work, &lwork, &info)

		//     Compute |QT*D - QT*D| / |D|
		goblas.Dgemm(Trans, NoTrans, n, m, n, toPtrf64(-one), q, n, d, n, &one, df, n)
		resid = golapack.Dlange('1', n, m, df, n, rwork)
		if dnorm > zero {
			result.Set(3, resid/(eps*float64(maxint(1, *n))*dnorm))
		} else {
			result.Set(3, zero)
		}

		//     Generate random n-by-m matrix D and a copy DF
		for j = 1; j <= (*n); j++ {
			golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, m, c.Vector(1-1, j-1))
		}
		cnorm = golapack.Dlange('1', m, n, c, m, rwork)
		golapack.Dlacpy('F', m, n, c, m, cf, m)

		//     Apply Q to C as C*Q
		golapack.Dgemlq('R', 'N', m, n, &k, af, m, t, &tsize, cf, m, work, &lwork, &info)

		//     Compute |C*Q - C*Q| / |C|
		goblas.Dgemm(NoTrans, NoTrans, m, n, n, toPtrf64(-one), c, m, q, n, &one, cf, m)
		resid = golapack.Dlange('1', n, m, df, n, rwork)
		if cnorm > zero {
			result.Set(4, resid/(eps*float64(maxint(1, *n))*cnorm))
		} else {
			result.Set(4, zero)
		}

		//     Copy C into CF again
		golapack.Dlacpy('F', m, n, c, m, cf, m)

		//     Apply Q to D as D*QT
		golapack.Dgemlq('R', 'T', m, n, &k, af, m, t, &tsize, cf, m, work, &lwork, &info)

		//     Compute |C*QT - C*QT| / |C|
		goblas.Dgemm(NoTrans, Trans, m, n, n, toPtrf64(-one), c, m, q, n, &one, cf, m)
		resid = golapack.Dlange('1', m, n, cf, m, rwork)
		if cnorm > zero {
			result.Set(5, resid/(eps*float64(maxint(1, *n))*cnorm))
		} else {
			result.Set(5, zero)
		}

	}
}
