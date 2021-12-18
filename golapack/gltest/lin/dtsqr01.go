package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dtsqr01 tests DGEQR , Dgelq, Dgemlq and DGEMQR.
func dtsqr01(tssw byte, m, n, mb, nb int, result *mat.Vector) {
	var testzeros, ts bool
	var anorm, cnorm, dnorm, eps, one, resid, zero float64
	var j, k, l, lwork, mnb, tsize int
	var err error

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
	k = min(m, n)
	l = max(m, n, 1)
	mnb = max(mb, nb)
	lwork = max(3, l) * mnb

	//     Dynamically allocate local arrays
	// Allocate(A(m, n), Af(m, n), Q(&l, &l), R(m, &l), Rwork(&l), C(m, n), Cf(m, n), D(n, m), Df(n, m), Lq(&l, n))
	a := mf(m, n, opts)
	af := mf(m, n, opts)
	q := mf(l, l, opts)
	r := mf(m, l, opts)
	rwork := vf(l)
	c := mf(m, n, opts)
	cf := mf(m, n, opts)
	d := mf(n, m, opts)
	df := mf(n, m, opts)
	lq := mf(l, n, opts)

	//     Put random numbers into A and copy to AF
	for j = 1; j <= n; j++ {
		golapack.Dlarnv(2, &iseed, m, a.Off(1-1, j-1).Vector())
	}
	if testzeros {
		if m >= 4 {
			for j = 1; j <= n; j++ {
				golapack.Dlarnv(2, &iseed, m/2, a.Off(m/4-1, j-1).Vector())
			}
		}
	}
	golapack.Dlacpy(Full, m, n, a, af)

	if ts {
		//     Factor the matrix A in the array AF.
		if err = golapack.Dgeqr(m, n, af, tquery, -1, workquery, -1); err != nil {
			panic(err)
		}
		tsize = int(tquery.Get(0))
		lwork = int(workquery.Get(0))
		if err = golapack.Dgemqr(Left, NoTrans, m, m, k, af, tquery, tsize, cf, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.Get(0)))
		if err = golapack.Dgemqr(Left, NoTrans, m, n, k, af, tquery, tsize, cf, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.Get(0)))
		if err = golapack.Dgemqr(Left, Trans, m, n, k, af, tquery, tsize, cf, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.Get(0)))
		if err = golapack.Dgemqr(Right, NoTrans, n, m, k, af, tquery, tsize, df, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.Get(0)))
		if err = golapack.Dgemqr(Right, Trans, n, m, k, af, tquery, tsize, df, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.Get(0)))

		// Allocate(T(&tsize))
		// Allocate(Work(&lwork))
		t := vf(tsize)
		work := vf(lwork)

		*srnamt = "Dgeqr"
		if err = golapack.Dgeqr(m, n, af, t, tsize, work, lwork); err != nil {
			panic(err)
		}

		//     Generate the m-by-m matrix Q
		golapack.Dlaset('F', m, m, zero, one, q.UpdateRows(m))
		*srnamt = "Dgemqr"
		if err = golapack.Dgemqr(Left, NoTrans, m, m, k, af, t, tsize, q, work, lwork); err != nil {
			panic(err)
		}

		//     Copy R
		golapack.Dlaset('F', m, n, zero, zero, r)
		golapack.Dlacpy(Upper, m, n, af, r)

		//     Compute |R - Q'*A| / |A| and store in RESULT(1)
		if err = r.Gemm(Trans, NoTrans, m, n, m, -one, q, a, one); err != nil {
			panic(err)
		}
		anorm = golapack.Dlange('1', m, n, a, rwork)
		resid = golapack.Dlange('1', m, n, r, rwork)
		if anorm > zero {
			result.Set(0, resid/(eps*float64(max(1, m))*anorm))
		} else {
			result.Set(0, zero)
		}

		//     Compute |I - Q'*Q| and store in RESULT(2)
		golapack.Dlaset('F', m, m, zero, one, r)
		if err = r.Syrk(Upper, ConjTrans, m, m, -one, q, one); err != nil {
			panic(err)
		}
		resid = golapack.Dlansy('1', Upper, m, r, rwork)
		result.Set(1, resid/(eps*float64(max(1, m))))

		//     Generate random m-by-n matrix C and a copy CF
		for j = 1; j <= n; j++ {
			golapack.Dlarnv(2, &iseed, m, c.Off(1-1, j-1).Vector())
		}
		cnorm = golapack.Dlange('1', m, n, c, rwork)
		golapack.Dlacpy(Full, m, n, c, cf)

		//     Apply Q to C as Q*C
		*srnamt = "Dgemqr"
		if err = golapack.Dgemqr(Left, NoTrans, m, n, k, af, t, tsize, cf, work, lwork); err != nil {
			panic(err)
		}

		//     Compute |Q*C - Q*C| / |C|
		if err = cf.Gemm(NoTrans, NoTrans, m, n, m, -one, q, c, one); err != nil {
			panic(err)
		}
		resid = golapack.Dlange('1', m, n, cf, rwork)
		if cnorm > zero {
			result.Set(2, resid/(eps*float64(max(1, m))*cnorm))
		} else {
			result.Set(2, zero)
		}

		//     Copy C into CF again
		golapack.Dlacpy(Full, m, n, c, cf)

		//     Apply Q to C as QT*C
		*srnamt = "Dgemqr"
		if err = golapack.Dgemqr(Left, Trans, m, n, k, af, t, tsize, cf, work, lwork); err != nil {
			panic(err)
		}

		//     Compute |QT*C - QT*C| / |C|
		if err = cf.Gemm(Trans, NoTrans, m, n, m, -one, q, c, one); err != nil {
			panic(err)
		}
		resid = golapack.Dlange('1', m, n, cf, rwork)
		if cnorm > zero {
			result.Set(3, resid/(eps*float64(max(1, m))*cnorm))
		} else {
			result.Set(3, zero)
		}

		//     Generate random n-by-m matrix D and a copy DF
		for j = 1; j <= m; j++ {
			golapack.Dlarnv(2, &iseed, n, d.Off(1-1, j-1).Vector())
		}
		dnorm = golapack.Dlange('1', n, m, d, rwork)
		golapack.Dlacpy(Full, n, m, d, df)

		//     Apply Q to D as D*Q
		*srnamt = "Dgemqr"
		if err = golapack.Dgemqr(Right, NoTrans, n, m, k, af, t, tsize, df, work, lwork); err != nil {
			panic(err)
		}

		//     Compute |D*Q - D*Q| / |D|
		if err = df.Gemm(NoTrans, NoTrans, n, m, m, -one, d, q, one); err != nil {
			panic(err)
		}
		resid = golapack.Dlange('1', n, m, df, rwork)
		if dnorm > zero {
			result.Set(4, resid/(eps*float64(max(1, m))*dnorm))
		} else {
			result.Set(4, zero)
		}

		//     Copy D into DF again
		golapack.Dlacpy(Full, n, m, d, df)

		//     Apply Q to D as D*QT
		if err = golapack.Dgemqr(Right, Trans, n, m, k, af, t, tsize, df, work, lwork); err != nil {
			panic(err)
		}

		//     Compute |D*QT - D*QT| / |D|
		if err = df.Gemm(NoTrans, Trans, n, m, m, -one, d, q, one); err != nil {
			panic(err)
		}
		resid = golapack.Dlange('1', n, m, df, rwork)
		if cnorm > zero {
			result.Set(5, resid/(eps*float64(max(1, m))*dnorm))
		} else {
			result.Set(5, zero)
		}

		//     Short and wide
	} else {
		if err = golapack.Dgelq(m, n, af, tquery, -1, workquery, -1); err != nil {
			panic(err)
		}
		tsize = int(tquery.Get(0))
		lwork = int(workquery.Get(0))
		if err = golapack.Dgemlq(Right, NoTrans, n, n, k, af, tquery, tsize, q, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.Get(0)))
		if err = golapack.Dgemlq(Left, NoTrans, n, m, k, af, tquery, tsize, df, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.Get(0)))
		if err = golapack.Dgemlq(Left, Trans, n, m, k, af, tquery, tsize, df, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.Get(0)))
		if err = golapack.Dgemlq(Right, NoTrans, m, n, k, af, tquery, tsize, cf, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.Get(0)))
		if err = golapack.Dgemlq(Right, Trans, m, n, k, af, tquery, tsize, cf, workquery, -1); err != nil {
			panic(err)
		}
		lwork = max(lwork, int(workquery.Get(0)))

		// Allocate(T(&tsize))
		// Allocate(Work(&lwork))
		t := vf(tsize)
		work := vf(lwork)

		*srnamt = "Dgelq"
		if err = golapack.Dgelq(m, n, af, t, tsize, work, lwork); err != nil {
			panic(err)
		}

		//     Generate the n-by-n matrix Q
		golapack.Dlaset('F', n, n, zero, one, q.UpdateRows(n))
		*srnamt = "Dgemlq"
		if err = golapack.Dgemlq(Right, NoTrans, n, n, k, af, t, tsize, q, work, lwork); err != nil {
			panic(err)
		}

		//     Copy R
		golapack.Dlaset('F', m, n, zero, zero, lq)
		golapack.Dlacpy(Lower, m, n, af, lq)

		//     Compute |L - A*Q'| / |A| and store in RESULT(1)
		if err = lq.Gemm(NoTrans, Trans, m, n, n, -one, a, q, one); err != nil {
			panic(err)
		}
		anorm = golapack.Dlange('1', m, n, a, rwork)
		resid = golapack.Dlange('1', m, n, lq, rwork)
		if anorm > zero {
			result.Set(0, resid/(eps*float64(max(1, n))*anorm))
		} else {
			result.Set(0, zero)
		}

		//     Compute |I - Q'*Q| and store in RESULT(2)
		golapack.Dlaset('F', n, n, zero, one, lq)
		if err = lq.Syrk(Upper, ConjTrans, n, n, -one, q, one); err != nil {
			panic(err)
		}
		resid = golapack.Dlansy('1', Upper, n, lq, rwork)
		result.Set(1, resid/(eps*float64(max(1, n))))

		//     Generate random m-by-n matrix C and a copy CF
		for j = 1; j <= m; j++ {
			golapack.Dlarnv(2, &iseed, n, d.Off(1-1, j-1).Vector())
		}
		dnorm = golapack.Dlange('1', n, m, d, rwork)
		golapack.Dlacpy(Full, n, m, d, df)

		//     Apply Q to C as Q*C
		if err = golapack.Dgemlq(Left, NoTrans, n, m, k, af, t, tsize, df, work, lwork); err != nil {
			panic(err)
		}

		//     Compute |Q*D - Q*D| / |D|
		if err = df.Gemm(NoTrans, NoTrans, n, m, n, -one, q, d, one); err != nil {
			panic(err)
		}
		resid = golapack.Dlange('1', n, m, df, rwork)
		if dnorm > zero {
			result.Set(2, resid/(eps*float64(max(1, n))*dnorm))
		} else {
			result.Set(2, zero)
		}

		//     Copy D into DF again
		golapack.Dlacpy(Full, n, m, d, df)

		//     Apply Q to D as QT*D
		if err = golapack.Dgemlq(Left, Trans, n, m, k, af, t, tsize, df, work, lwork); err != nil {
			panic(err)
		}

		//     Compute |QT*D - QT*D| / |D|
		if err = df.Gemm(Trans, NoTrans, n, m, n, -one, q, d, one); err != nil {
			panic(err)
		}
		resid = golapack.Dlange('1', n, m, df, rwork)
		if dnorm > zero {
			result.Set(3, resid/(eps*float64(max(1, n))*dnorm))
		} else {
			result.Set(3, zero)
		}

		//     Generate random n-by-m matrix D and a copy DF
		for j = 1; j <= n; j++ {
			golapack.Dlarnv(2, &iseed, m, c.Off(1-1, j-1).Vector())
		}
		cnorm = golapack.Dlange('1', m, n, c, rwork)
		golapack.Dlacpy(Full, m, n, c, cf)

		//     Apply Q to C as C*Q
		if err = golapack.Dgemlq(Right, NoTrans, m, n, k, af, t, tsize, cf, work, lwork); err != nil {
			panic(err)
		}

		//     Compute |C*Q - C*Q| / |C|
		if err = cf.Gemm(NoTrans, NoTrans, m, n, n, -one, c, q, one); err != nil {
			panic(err)
		}
		resid = golapack.Dlange('1', n, m, df, rwork)
		if cnorm > zero {
			result.Set(4, resid/(eps*float64(max(1, n))*cnorm))
		} else {
			result.Set(4, zero)
		}

		//     Copy C into CF again
		golapack.Dlacpy(Full, m, n, c, cf)

		//     Apply Q to D as D*QT
		if err = golapack.Dgemlq(Right, Trans, m, n, k, af, t, tsize, cf, work, lwork); err != nil {
			panic(err)
		}

		//     Compute |C*QT - C*QT| / |C|
		if err = cf.Gemm(NoTrans, Trans, m, n, n, -one, c, q, one); err != nil {
			panic(err)
		}
		resid = golapack.Dlange('1', m, n, cf, rwork)
		if cnorm > zero {
			result.Set(5, resid/(eps*float64(max(1, n))*cnorm))
		} else {
			result.Set(5, zero)
		}

	}
}
