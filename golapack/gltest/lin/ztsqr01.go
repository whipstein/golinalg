package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// ztsqr01 tests Zgeqr , Zgelq, Zgemlq and Zgemqr.
func ztsqr01(tssw byte, m, n, mb, nb int, result *mat.Vector) {
	var testzeros, ts bool
	var czero, one complex128
	var anorm, cnorm, dnorm, eps, resid, zero float64
	var j, k, l, lwork, mnb, tsize int
	var err error

	tquery := cvf(5)
	workquery := cvf(1)
	iseed := make([]int, 4)

	zero = 0.0
	one = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	//     TEST TALL SKINNY OR SHORT WIDE
	ts = tssw == 'T'

	//     TEST MATRICES WITH HALF OF MATRIX BEING ZEROS
	testzeros = false

	eps = golapack.Dlamch(Epsilon)
	k = min(m, n)
	l = max(m, n, 1)
	mnb = max(mb, nb)
	lwork = max(3, l) * mnb

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
	lq := cmf(l, n, opts)

	//     Put random numbers into A and copy to af
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

	if ts {
		//     Factor the matrix A in the array af.
		if err = golapack.Zgeqr(m, n, af, tquery, -1, workquery, -1); err != nil {
			panic(err)
		}
		tsize = int(tquery.GetRe(0))
		lwork = int(workquery.GetRe(0))
		if err = golapack.Zgemqr(Left, NoTrans, m, m, k, af, tquery, tsize, cf, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.GetRe(0)))
		if err = golapack.Zgemqr(Left, NoTrans, m, n, k, af, tquery, tsize, cf, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.GetRe(0)))
		if err = golapack.Zgemqr(Left, ConjTrans, m, n, k, af, tquery, tsize, cf, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.GetRe(0)))
		if err = golapack.Zgemqr(Right, NoTrans, n, m, k, af, tquery, tsize, df, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.GetRe(0)))
		if err = golapack.Zgemqr(Right, ConjTrans, n, m, k, af, tquery, tsize, df, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.GetRe(0)))
		t := cvf(tsize)
		work := cvf(lwork)
		*srnamt = "Zgeqr"
		if err = golapack.Zgeqr(m, n, af, t, tsize, work, lwork); err != nil {
			panic(err)
		}

		//     Generate the m-by-m matrix Q
		golapack.Zlaset(Full, m, m, czero, one, q)
		*srnamt = "Zgemqr"
		if err = golapack.Zgemqr(Left, NoTrans, m, m, k, af, t, tsize, q, work, lwork); err != nil {
			panic(err)
		}

		//     Copy R
		golapack.Zlaset(Full, m, n, czero, czero, r)
		golapack.Zlacpy(Upper, m, n, af, r)

		//     Compute |R - Q'*A| / |A| and store in RESULT(1)
		if err = goblas.Zgemm(ConjTrans, NoTrans, m, n, m, -one, q, a, one, r); err != nil {
			panic(err)
		}
		anorm = golapack.Zlange('1', m, n, a, rwork)
		resid = golapack.Zlange('1', m, n, r, rwork)
		if anorm > zero {
			result.Set(0, resid/(eps*float64(max(1, m))*anorm))
		} else {
			result.Set(0, zero)
		}

		//     Compute |I - Q'*Q| and store in RESULT(2)
		golapack.Zlaset(Full, m, m, czero, one, r)
		if err = goblas.Zherk(Upper, ConjTrans, m, m, real(-one), q, real(one), r); err != nil {
			panic(err)
		}
		resid = golapack.Zlansy('1', Upper, m, r, rwork)
		result.Set(1, resid/(eps*float64(max(1, m))))

		//     Generate random m-by-n matrix C and a copy CF
		for j = 1; j <= n; j++ {
			golapack.Zlarnv(2, &iseed, m, c.CVector(0, j-1))
		}
		cnorm = golapack.Zlange('1', m, n, c, rwork)
		golapack.Zlacpy(Full, m, n, c, cf)

		//     Apply Q to C as Q*C
		*srnamt = "Zgemqr"
		if err = golapack.Zgemqr(Left, NoTrans, m, n, k, af, t, tsize, cf, work, lwork); err != nil {
			panic(err)
		}

		//     Compute |Q*C - Q*C| / |C|
		if err = goblas.Zgemm(NoTrans, NoTrans, m, n, m, -one, q, c, one, cf); err != nil {
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

		//     Apply Q to C as QT*C
		*srnamt = "Zgemqr"
		if err = golapack.Zgemqr(Left, ConjTrans, m, n, k, af, t, tsize, cf, work, lwork); err != nil {
			panic(err)
		}

		//     Compute |QT*C - QT*C| / |C|
		goblas.Zgemm(ConjTrans, NoTrans, m, n, m, -one, q, c, one, cf)
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

		//     Apply Q to D as D*Q
		*srnamt = "Zgemqr"
		if err = golapack.Zgemqr(Right, NoTrans, n, m, k, af, t, tsize, df, work, lwork); err != nil {
			panic(err)
		}

		//     Compute |D*Q - D*Q| / |D|
		if err = goblas.Zgemm(NoTrans, NoTrans, n, m, m, -one, d, q, one, df); err != nil {
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

		//     Apply Q to D as D*QT
		if err = golapack.Zgemqr(Right, ConjTrans, n, m, k, af, t, tsize, df, work, lwork); err != nil {
			panic(err)
		}

		//     Compute |D*QT - D*QT| / |D|
		if err = goblas.Zgemm(NoTrans, ConjTrans, n, m, m, -one, d, q, one, df); err != nil {
			panic(err)
		}
		resid = golapack.Zlange('1', n, m, df, rwork)
		if cnorm > zero {
			result.Set(5, resid/(eps*float64(max(1, m))*dnorm))
		} else {
			result.Set(5, zero)
		}

		//     Short and wide
	} else {
		if err = golapack.Zgelq(m, n, af, tquery, -1, workquery, -1); err != nil {
			panic(err)
		}
		tsize = int(tquery.GetRe(0))
		lwork = int(workquery.GetRe(0))
		if err = golapack.Zgemlq(Right, NoTrans, n, n, k, af, tquery, tsize, q, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.GetRe(0)))
		if err = golapack.Zgemlq(Left, NoTrans, n, m, k, af, tquery, tsize, df, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.GetRe(0)))
		if err = golapack.Zgemlq(Left, ConjTrans, n, m, k, af, tquery, tsize, df, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.GetRe(0)))
		if err = golapack.Zgemlq(Right, NoTrans, m, n, k, af, tquery, tsize, cf, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.GetRe(0)))
		if err = golapack.Zgemlq(Right, ConjTrans, m, n, k, af, tquery, tsize, cf, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.GetRe(0)))
		t := cvf(tsize)
		work := cvf(lwork)
		*srnamt = "Zgelq"
		if err = golapack.Zgelq(m, n, af, t, tsize, work, lwork); err != nil {
			panic(err)
		}

		//     Generate the n-by-n matrix Q
		golapack.Zlaset(Full, n, n, czero, one, q)
		*srnamt = "Zgemlq"
		if err = golapack.Zgemlq(Right, NoTrans, n, n, k, af, t, tsize, q, work, lwork); err != nil {
			panic(err)
		}

		//     Copy R
		golapack.Zlaset(Full, m, n, czero, czero, lq)
		golapack.Zlacpy(Lower, m, n, af, lq)

		//     Compute |L - A*Q'| / |A| and store in RESULT(1)
		if err = goblas.Zgemm(NoTrans, ConjTrans, m, n, n, -one, a, q, one, lq); err != nil {
			panic(err)
		}
		anorm = golapack.Zlange('1', m, n, a, rwork)
		resid = golapack.Zlange('1', m, n, lq, rwork)
		if anorm > zero {
			result.Set(0, resid/(eps*float64(max(1, n))*anorm))
		} else {
			result.Set(0, zero)
		}

		//     Compute |I - Q'*Q| and store in RESULT(2)
		golapack.Zlaset(Full, n, n, czero, one, lq)
		if err = goblas.Zherk(Upper, ConjTrans, n, n, real(-one), q, real(one), lq); err != nil {
			panic(err)
		}
		resid = golapack.Zlansy('1', Upper, n, lq, rwork)
		result.Set(1, resid/(eps*float64(max(1, n))))

		//     Generate random m-by-n matrix C and a copy CF
		for j = 1; j <= m; j++ {
			golapack.Zlarnv(2, &iseed, n, d.CVector(0, j-1))
		}
		dnorm = golapack.Zlange('1', n, m, d, rwork)
		golapack.Zlacpy(Full, n, m, d, df)

		//     Apply Q to C as Q*C
		if err = golapack.Zgemlq(Left, NoTrans, n, m, k, af, t, tsize, df, work, lwork); err != nil {
			panic(err)
		}

		//     Compute |Q*D - Q*D| / |D|
		if err = goblas.Zgemm(NoTrans, NoTrans, n, m, n, -one, q, d, one, df); err != nil {
			panic(err)
		}
		resid = golapack.Zlange('1', n, m, df, rwork)
		if dnorm > zero {
			result.Set(2, resid/(eps*float64(max(1, n))*dnorm))
		} else {
			result.Set(2, zero)
		}

		//     Copy D into DF again
		golapack.Zlacpy(Full, n, m, d, df)

		//     Apply Q to D as QT*D
		if err = golapack.Zgemlq(Left, ConjTrans, n, m, k, af, t, tsize, df, work, lwork); err != nil {
			panic(err)
		}

		//     Compute |QT*D - QT*D| / |D|
		if err = goblas.Zgemm(ConjTrans, NoTrans, n, m, n, -one, q, d, one, df); err != nil {
			panic(err)
		}
		resid = golapack.Zlange('1', n, m, df, rwork)
		if dnorm > zero {
			result.Set(3, resid/(eps*float64(max(1, n))*dnorm))
		} else {
			result.Set(3, zero)
		}

		//     Generate random n-by-m matrix D and a copy DF
		for j = 1; j <= n; j++ {
			golapack.Zlarnv(2, &iseed, m, c.CVector(0, j-1))
		}
		cnorm = golapack.Zlange('1', m, n, c, rwork)
		golapack.Zlacpy(Full, m, n, c, cf)

		//     Apply Q to C as C*Q
		if err = golapack.Zgemlq(Right, NoTrans, m, n, k, af, t, tsize, cf, work, lwork); err != nil {
			panic(err)
		}

		//     Compute |C*Q - C*Q| / |C|
		if err = goblas.Zgemm(NoTrans, NoTrans, m, n, n, -one, c, q, one, cf); err != nil {
			panic(err)
		}
		resid = golapack.Zlange('1', n, m, df, rwork)
		if cnorm > zero {
			result.Set(4, resid/(eps*float64(max(1, n))*cnorm))
		} else {
			result.Set(4, zero)
		}

		//     Copy C into CF again
		golapack.Zlacpy(Full, m, n, c, cf)

		//     Apply Q to D as D*QT
		if err = golapack.Zgemlq(Right, ConjTrans, m, n, k, af, t, tsize, cf, work, lwork); err != nil {
			panic(err)
		}

		//     Compute |C*QT - C*QT| / |C|
		if err = goblas.Zgemm(NoTrans, ConjTrans, m, n, n, -one, c, q, one, cf); err != nil {
			panic(err)
		}
		resid = golapack.Zlange('1', m, n, cf, rwork)
		if cnorm > zero {
			result.Set(5, resid/(eps*float64(max(1, n))*cnorm))
		} else {
			result.Set(5, zero)
		}

	}
}
