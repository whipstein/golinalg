package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztsqr01 tests ZGEQR , ZGELQ, ZGEMLQ and ZGEMQR.
func Ztsqr01(tssw byte, m, n, mb, nb *int, result *mat.Vector) {
	var testzeros, ts bool
	var czero, one complex128
	var anorm, cnorm, dnorm, eps, resid, zero float64
	var info, j, k, l, lwork, mnb, tsize int
	var err error
	_ = err

	tquery := cvf(5)
	workquery := cvf(1)
	iseed := make([]int, 4)

	zero = 0.0
	one = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	//     TEST TALL SKINNY OR SHORT WIDE
	ts = string(tssw) == "TS"

	//     TEST MATRICES WITH HALF OF MATRIX BEING ZEROS
	testzeros = false

	eps = golapack.Dlamch(Epsilon)
	k = min(*m, *n)
	l = max(*m, *n, 1)
	mnb = max(*mb, *nb)
	lwork = max(3, l) * mnb

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
	lq := cmf(l, *n, opts)

	//     Put random numbers into A and copy to af
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

	if ts {
		//     Factor the matrix A in the array af.
		golapack.Zgeqr(m, n, af, m, tquery, toPtr(-1), workquery, toPtr(-1), &info)
		tsize = int(tquery.GetRe(0))
		lwork = int(workquery.GetRe(0))
		golapack.Zgemqr('L', 'N', m, m, &k, af, m, tquery, &tsize, cf, m, workquery, toPtr(-1), &info)
		lwork = max(lwork, int(workquery.GetRe(0)))
		golapack.Zgemqr('L', 'N', m, n, &k, af, m, tquery, &tsize, cf, m, workquery, toPtr(-1), &info)
		lwork = max(lwork, int(workquery.GetRe(0)))
		golapack.Zgemqr('L', 'C', m, n, &k, af, m, tquery, &tsize, cf, m, workquery, toPtr(-1), &info)
		lwork = max(lwork, int(workquery.GetRe(0)))
		golapack.Zgemqr('R', 'N', n, m, &k, af, m, tquery, &tsize, df, n, workquery, toPtr(-1), &info)
		lwork = max(lwork, int(workquery.GetRe(0)))
		golapack.Zgemqr('R', 'C', n, m, &k, af, m, tquery, &tsize, df, n, workquery, toPtr(-1), &info)
		lwork = max(lwork, int(workquery.GetRe(0)))
		t := cvf(tsize)
		work := cvf(lwork)
		*srnamt = "ZGEQR"
		golapack.Zgeqr(m, n, af, m, t, &tsize, work, &lwork, &info)

		//     Generate the m-by-m matrix Q
		golapack.Zlaset('F', m, m, &czero, &one, q, m)
		*srnamt = "ZGEMQR"
		golapack.Zgemqr('L', 'N', m, m, &k, af, m, t, &tsize, q, m, work, &lwork, &info)

		//     Copy R
		golapack.Zlaset('F', m, n, &czero, &czero, r, m)
		golapack.Zlacpy('U', m, n, af, m, r, m)

		//     Compute |R - Q'*A| / |A| and store in RESULT(1)
		err = goblas.Zgemm(ConjTrans, NoTrans, *m, *n, *m, -one, q, a, one, r)
		anorm = golapack.Zlange('1', m, n, a, m, rwork)
		resid = golapack.Zlange('1', m, n, r, m, rwork)
		if anorm > zero {
			result.Set(0, resid/(eps*float64(max(1, *m))*anorm))
		} else {
			result.Set(0, zero)
		}

		//     Compute |I - Q'*Q| and store in RESULT(2)
		golapack.Zlaset('F', m, m, &czero, &one, r, m)
		err = goblas.Zherk(Upper, ConjTrans, *m, *m, real(-one), q, real(one), r)
		resid = golapack.Zlansy('1', 'U', m, r, m, rwork)
		result.Set(1, resid/(eps*float64(max(1, *m))))

		//     Generate random m-by-n matrix C and a copy CF
		for j = 1; j <= (*n); j++ {
			golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, m, c.CVector(0, j-1))
		}
		cnorm = golapack.Zlange('1', m, n, c, m, rwork)
		golapack.Zlacpy('F', m, n, c, m, cf, m)

		//     Apply Q to C as Q*C
		*srnamt = "ZGEMQR"
		golapack.Zgemqr('L', 'N', m, n, &k, af, m, t, &tsize, cf, m, work, &lwork, &info)

		//     Compute |Q*C - Q*C| / |C|
		err = goblas.Zgemm(NoTrans, NoTrans, *m, *n, *m, -one, q, c, one, cf)
		resid = golapack.Zlange('1', m, n, cf, m, rwork)
		if cnorm > zero {
			result.Set(2, resid/(eps*float64(max(1, *m))*cnorm))
		} else {
			result.Set(2, zero)
		}

		//     Copy C into CF again
		golapack.Zlacpy('F', m, n, c, m, cf, m)

		//     Apply Q to C as QT*C
		*srnamt = "ZGEMQR"
		golapack.Zgemqr('L', 'C', m, n, &k, af, m, t, &tsize, cf, m, work, &lwork, &info)

		//     Compute |QT*C - QT*C| / |C|
		goblas.Zgemm(ConjTrans, NoTrans, *m, *n, *m, -one, q, c, one, cf)
		resid = golapack.Zlange('1', m, n, cf, m, rwork)
		if cnorm > zero {
			result.Set(3, resid/(eps*float64(max(1, *m))*cnorm))
		} else {
			result.Set(3, zero)
		}

		//     Generate random n-by-m matrix D and a copy DF
		for j = 1; j <= (*m); j++ {
			golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, n, d.CVector(0, j-1))
		}
		dnorm = golapack.Zlange('1', n, m, d, n, rwork)
		golapack.Zlacpy('F', n, m, d, n, df, n)

		//     Apply Q to D as D*Q
		*srnamt = "ZGEMQR"
		golapack.Zgemqr('R', 'N', n, m, &k, af, m, t, &tsize, df, n, work, &lwork, &info)

		//     Compute |D*Q - D*Q| / |D|
		err = goblas.Zgemm(NoTrans, NoTrans, *n, *m, *m, -one, d, q, one, df)
		resid = golapack.Zlange('1', n, m, df, n, rwork)
		if dnorm > zero {
			result.Set(4, resid/(eps*float64(max(1, *m))*dnorm))
		} else {
			result.Set(4, zero)
		}

		//     Copy D into DF again
		golapack.Zlacpy('F', n, m, d, n, df, n)

		//     Apply Q to D as D*QT
		golapack.Zgemqr('R', 'C', n, m, &k, af, m, t, &tsize, df, n, work, &lwork, &info)

		//     Compute |D*QT - D*QT| / |D|
		err = goblas.Zgemm(NoTrans, ConjTrans, *n, *m, *m, -one, d, q, one, df)
		resid = golapack.Zlange('1', n, m, df, n, rwork)
		if cnorm > zero {
			result.Set(5, resid/(eps*float64(max(1, *m))*dnorm))
		} else {
			result.Set(5, zero)
		}

		//     Short and wide
	} else {
		golapack.Zgelq(m, n, af, m, tquery, toPtr(-1), workquery, toPtr(-1), &info)
		tsize = int(tquery.GetRe(0))
		lwork = int(workquery.GetRe(0))
		golapack.Zgemlq('R', 'N', n, n, &k, af, m, tquery, &tsize, q, n, workquery, toPtr(-1), &info)
		lwork = max(lwork, int(workquery.GetRe(0)))
		golapack.Zgemlq('L', 'N', n, m, &k, af, m, tquery, &tsize, df, n, workquery, toPtr(-1), &info)
		lwork = max(lwork, int(workquery.GetRe(0)))
		golapack.Zgemlq('L', 'C', n, m, &k, af, m, tquery, &tsize, df, n, workquery, toPtr(-1), &info)
		lwork = max(lwork, int(workquery.GetRe(0)))
		golapack.Zgemlq('R', 'N', m, n, &k, af, m, tquery, &tsize, cf, m, workquery, toPtr(-1), &info)
		lwork = max(lwork, int(workquery.GetRe(0)))
		golapack.Zgemlq('R', 'C', m, n, &k, af, m, tquery, &tsize, cf, m, workquery, toPtr(-1), &info)
		lwork = max(lwork, int(workquery.GetRe(0)))
		t := cvf(tsize)
		work := cvf(lwork)
		*srnamt = "ZGELQ"
		golapack.Zgelq(m, n, af, m, t, &tsize, work, &lwork, &info)

		//     Generate the n-by-n matrix Q
		golapack.Zlaset('F', n, n, &czero, &one, q, n)
		*srnamt = "ZGEMLQ"
		golapack.Zgemlq('R', 'N', n, n, &k, af, m, t, &tsize, q, n, work, &lwork, &info)

		//     Copy R
		golapack.Zlaset('F', m, n, &czero, &czero, lq, &l)
		golapack.Zlacpy('L', m, n, af, m, lq, &l)

		//     Compute |L - A*Q'| / |A| and store in RESULT(1)
		err = goblas.Zgemm(NoTrans, ConjTrans, *m, *n, *n, -one, a, q, one, lq)
		anorm = golapack.Zlange('1', m, n, a, m, rwork)
		resid = golapack.Zlange('1', m, n, lq, &l, rwork)
		if anorm > zero {
			result.Set(0, resid/(eps*float64(max(1, *n))*anorm))
		} else {
			result.Set(0, zero)
		}

		//     Compute |I - Q'*Q| and store in RESULT(2)
		golapack.Zlaset('F', n, n, &czero, &one, lq, &l)
		err = goblas.Zherk(Upper, ConjTrans, *n, *n, real(-one), q, real(one), lq)
		resid = golapack.Zlansy('1', 'U', n, lq, &l, rwork)
		result.Set(1, resid/(eps*float64(max(1, *n))))

		//     Generate random m-by-n matrix C and a copy CF
		for j = 1; j <= (*m); j++ {
			golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, n, d.CVector(0, j-1))
		}
		dnorm = golapack.Zlange('1', n, m, d, n, rwork)
		golapack.Zlacpy('F', n, m, d, n, df, n)

		//     Apply Q to C as Q*C
		golapack.Zgemlq('L', 'N', n, m, &k, af, m, t, &tsize, df, n, work, &lwork, &info)

		//     Compute |Q*D - Q*D| / |D|
		err = goblas.Zgemm(NoTrans, NoTrans, *n, *m, *n, -one, q, d, one, df)
		resid = golapack.Zlange('1', n, m, df, n, rwork)
		if dnorm > zero {
			result.Set(2, resid/(eps*float64(max(1, *n))*dnorm))
		} else {
			result.Set(2, zero)
		}

		//     Copy D into DF again
		golapack.Zlacpy('F', n, m, d, n, df, n)

		//     Apply Q to D as QT*D
		golapack.Zgemlq('L', 'C', n, m, &k, af, m, t, &tsize, df, n, work, &lwork, &info)

		//     Compute |QT*D - QT*D| / |D|
		err = goblas.Zgemm(ConjTrans, NoTrans, *n, *m, *n, -one, q, d, one, df)
		resid = golapack.Zlange('1', n, m, df, n, rwork)
		if dnorm > zero {
			result.Set(3, resid/(eps*float64(max(1, *n))*dnorm))
		} else {
			result.Set(3, zero)
		}

		//     Generate random n-by-m matrix D and a copy DF
		for j = 1; j <= (*n); j++ {
			golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, m, c.CVector(0, j-1))
		}
		cnorm = golapack.Zlange('1', m, n, c, m, rwork)
		golapack.Zlacpy('F', m, n, c, m, cf, m)

		//     Apply Q to C as C*Q
		golapack.Zgemlq('R', 'N', m, n, &k, af, m, t, &tsize, cf, m, work, &lwork, &info)

		//     Compute |C*Q - C*Q| / |C|
		err = goblas.Zgemm(NoTrans, NoTrans, *m, *n, *n, -one, c, q, one, cf)
		resid = golapack.Zlange('1', n, m, df, n, rwork)
		if cnorm > zero {
			result.Set(4, resid/(eps*float64(max(1, *n))*cnorm))
		} else {
			result.Set(4, zero)
		}

		//     Copy C into CF again
		golapack.Zlacpy('F', m, n, c, m, cf, m)

		//     Apply Q to D as D*QT
		golapack.Zgemlq('R', 'C', m, n, &k, af, m, t, &tsize, cf, m, work, &lwork, &info)

		//     Compute |C*QT - C*QT| / |C|
		err = goblas.Zgemm(NoTrans, ConjTrans, *m, *n, *n, -one, c, q, one, cf)
		resid = golapack.Zlange('1', m, n, cf, m, rwork)
		if cnorm > zero {
			result.Set(5, resid/(eps*float64(max(1, *n))*cnorm))
		} else {
			result.Set(5, zero)
		}

	}
}
