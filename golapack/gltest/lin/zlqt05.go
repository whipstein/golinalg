package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zlqt05 tests ZTPLQT and ZTPMLQT.
func zlqt05(m, n, l, nb int, result *mat.Vector) {
	var czero, one complex128
	var anorm, cnorm, dnorm, eps, resid, zero float64
	var j, k, lwork, n2, np1 int
	var err error

	iseed := make([]int, 4)

	zero = 0.0
	one = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)
	k = m
	n2 = m + n
	if n > 0 {
		np1 = m + 1
	} else {
		np1 = 1
	}
	lwork = n2 * n2 * nb

	//     Dynamically allocate all arrays
	a := cmf(m, n2, opts)
	af := cmf(m, n2, opts)
	q := cmf(n2, n2, opts)
	r := cmf(n2, n2, opts)
	rwork := vf(n2)
	work := cvf(lwork)
	t := cmf(nb, m, opts)
	c := cmf(n2, m, opts)
	cf := cmf(n2, m, opts)
	d := cmf(m, n2, opts)
	df := cmf(m, n2, opts)

	//     Put random stuff into A
	golapack.Zlaset(Full, m, n2, czero, czero, a)
	golapack.Zlaset(Full, nb, m, czero, czero, t)
	for j = 1; j <= m; j++ {
		golapack.Zlarnv(2, &iseed, m-j+1, a.Off(j-1, j-1).CVector())
	}
	if n > 0 {
		for j = 1; j <= n-l; j++ {
			golapack.Zlarnv(2, &iseed, m, a.Off(0, min(n+m, m+1)+j-1-1).CVector())
		}
	}
	if l > 0 {
		for j = 1; j <= l; j++ {
			golapack.Zlarnv(2, &iseed, m-j+1, a.Off(j-1, min(n+m, n+m-l+1)+j-1-1).CVector())
		}
	}

	//     Copy the matrix A to the array af.
	golapack.Zlacpy(Full, m, n2, a, af)

	//     Factor the matrix A in the array af.
	if err = golapack.Ztplqt(m, n, l, nb, af, af.Off(0, np1-1), t, work); err != nil {
		panic(err)
	}

	//     Generate the (M+N)-by-(M+N) matrix Q by applying H to I
	golapack.Zlaset(Full, n2, n2, czero, one, q)
	if err = golapack.Zgemlqt(Left, NoTrans, n2, n2, k, nb, af, t, q, work); err != nil {
		panic(err)
	}

	//     Copy L
	golapack.Zlaset(Full, n2, n2, czero, czero, r)
	golapack.Zlacpy(Lower, m, n2, af, r)

	//     Compute |L - A*Q*C| / |A| and store in RESULT(1)
	if err = r.Gemm(NoTrans, ConjTrans, m, n2, n2, -one, a, q, one); err != nil {
		panic(err)
	}
	anorm = golapack.Zlange('1', m, n2, a, rwork)
	resid = golapack.Zlange('1', m, n2, r, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*anorm*float64(max(1, n2))))
	} else {
		result.Set(0, zero)
	}

	//     Compute |I - Q*Q'| and store in RESULT(2)
	golapack.Zlaset(Full, n2, n2, czero, one, r)
	if err = r.Herk(Upper, NoTrans, n2, n2, real(-one), q, real(one)); err != nil {
		panic(err)
	}
	resid = golapack.Zlansy('1', Upper, n2, r, rwork)
	result.Set(1, resid/(eps*float64(max(1, n2))))

	//     Generate random m-by-n matrix C and a copy CF
	golapack.Zlaset(Full, n2, m, czero, one, c)
	for j = 1; j <= m; j++ {
		golapack.Zlarnv(2, &iseed, n2, c.Off(0, j-1).CVector())
	}
	cnorm = golapack.Zlange('1', n2, m, c, rwork)
	golapack.Zlacpy(Full, n2, m, c, cf)

	//     Apply Q to C as Q*C
	if err = golapack.Ztpmlqt(Left, NoTrans, n, m, k, l, nb, af.Off(0, np1-1), t, cf, cf.Off(np1-1, 0), work); err != nil {
		panic(err)
	}

	//     Compute |Q*C - Q*C| / |C|
	if err = cf.Gemm(NoTrans, NoTrans, n2, m, n2, -one, q, c, one); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', n2, m, cf, rwork)
	if cnorm > zero {
		result.Set(2, resid/(eps*float64(max(1, n2))*cnorm))
	} else {
		result.Set(2, zero)
	}

	//     Copy C into CF again
	golapack.Zlacpy(Full, n2, m, c, cf)

	//     Apply Q to C as QT*C
	if err = golapack.Ztpmlqt(Left, ConjTrans, n, m, k, l, nb, af.Off(0, np1-1), t, cf, cf.Off(np1-1, 0), work); err != nil {
		panic(err)
	}

	//     Compute |QT*C - QT*C| / |C|
	if err = cf.Gemm(ConjTrans, NoTrans, n2, m, n2, -one, q, c, one); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', n2, m, cf, rwork)
	if cnorm > zero {
		result.Set(3, resid/(eps*float64(max(1, n2))*cnorm))
	} else {
		result.Set(3, zero)
	}

	//     Generate random m-by-n matrix D and a copy DF
	for j = 1; j <= n2; j++ {
		golapack.Zlarnv(2, &iseed, m, d.Off(0, j-1).CVector())
	}
	dnorm = golapack.Zlange('1', m, n2, d, rwork)
	golapack.Zlacpy(Full, m, n2, d, df)

	//     Apply Q to D as D*Q
	if err = golapack.Ztpmlqt(Right, NoTrans, m, n, k, l, nb, af.Off(0, np1-1), t, df, df.Off(0, np1-1), work); err != nil {
		panic(err)
	}

	//     Compute |D*Q - D*Q| / |D|
	if err = df.Gemm(NoTrans, NoTrans, m, n2, n2, -one, d, q, one); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', m, n2, df, rwork)
	if cnorm > zero {
		result.Set(4, resid/(eps*float64(max(1, n2))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy D into DF again
	golapack.Zlacpy(Full, m, n2, d, df)

	//     Apply Q to D as D*QT
	if err = golapack.Ztpmlqt(Right, ConjTrans, m, n, k, l, nb, af.Off(0, np1-1), t, df, df.Off(0, np1-1), work); err != nil {
		panic(err)
	}

	//     Compute |D*QT - D*QT| / |D|
	if err = df.Gemm(NoTrans, ConjTrans, m, n2, n2, -one, d, q, one); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', m, n2, df, rwork)
	if cnorm > zero {
		result.Set(5, resid/(eps*float64(max(1, n2))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
