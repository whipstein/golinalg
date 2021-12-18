package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zqrt04 tests ZGEQRT and ZGEMQRT.
func zqrt04(m, n, nb int, result *mat.Vector) {
	var czero, one complex128
	var anorm, cnorm, dnorm, eps, resid, zero float64
	var j, k, l, lwork int
	var err error

	iseed := make([]int, 4)

	zero = 0.0
	one = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)
	k = min(m, n)
	l = max(m, n)
	lwork = max(2, l) * max(2, l) * nb

	//     Dynamically allocate local arrays
	a := cmf(m, n, opts)
	af := cmf(m, n, opts)
	q := cmf(m, m, opts)
	r := cmf(m, l, opts)
	rwork := vf(l)
	work := cvf(lwork)
	t := cmf(nb, n, opts)
	c := cmf(m, n, opts)
	cf := cmf(m, n, opts)
	d := cmf(n, m, opts)
	df := cmf(n, m, opts)

	//     Put random numbers into A and copy to af
	// ldt = nb
	for j = 1; j <= n; j++ {
		golapack.Zlarnv(2, &iseed, m, a.Off(0, j-1).CVector())
	}
	golapack.Zlacpy(Full, m, n, a, af)

	//     Factor the matrix A in the array af.
	if err = golapack.Zgeqrt(m, n, nb, af, t, work); err != nil {
		panic(err)
	}

	//     Generate the m-by-m matrix Q
	golapack.Zlaset(Full, m, m, czero, one, q)
	if err = golapack.Zgemqrt(Right, NoTrans, m, m, k, nb, af, t, q, work); err != nil {
		panic(err)
	}

	//     Copy R
	golapack.Zlaset(Full, m, n, czero, czero, r)
	golapack.Zlacpy(Upper, m, n, af, r)

	//     Compute |R - Q'*A| / |A| and store in RESULT(1)
	err = r.Gemm(ConjTrans, NoTrans, m, n, m, -one, q, a, one)
	anorm = golapack.Zlange('1', m, n, a, rwork)
	resid = golapack.Zlange('1', m, n, r, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*float64(max(1, m))*anorm))
	} else {
		result.Set(0, zero)
	}

	//     Compute |I - Q'*Q| and store in RESULT(2)
	golapack.Zlaset(Full, m, m, czero, one, r)
	err = r.Herk(Upper, ConjTrans, m, m, real(-one), q, real(one))
	resid = golapack.Zlansy('1', Upper, m, r, rwork)
	result.Set(1, resid/(eps*float64(max(1, m))))
	//
	//     Generate random m-by-n matrix C and a copy CF
	//
	for j = 1; j <= n; j++ {
		golapack.Zlarnv(2, &iseed, m, c.Off(0, j-1).CVector())
	}
	cnorm = golapack.Zlange('1', m, n, c, rwork)
	golapack.Zlacpy(Full, m, n, c, cf)
	//
	//     Apply Q to C as Q*C
	//
	if err = golapack.Zgemqrt(Left, NoTrans, m, n, k, nb, af, t, cf, work); err != nil {
		panic(err)
	}
	//
	//     Compute |Q*C - Q*C| / |C|
	//
	err = cf.Gemm(NoTrans, NoTrans, m, n, m, -one, q, c, one)
	resid = golapack.Zlange('1', m, n, cf, rwork)
	if cnorm > zero {
		result.Set(2, resid/(eps*float64(max(1, m))*cnorm))
	} else {
		result.Set(2, zero)
	}
	//
	//     Copy C into CF again
	//
	golapack.Zlacpy(Full, m, n, c, cf)
	//
	//     Apply Q to C as QT*C
	//
	if err = golapack.Zgemqrt(Left, ConjTrans, m, n, k, nb, af, t, cf, work); err != nil {
		panic(err)
	}
	//
	//     Compute |QT*C - QT*C| / |C|
	//
	cf.Gemm(ConjTrans, NoTrans, m, n, m, -one, q, c, one)
	resid = golapack.Zlange('1', m, n, cf, rwork)
	if cnorm > zero {
		result.Set(3, resid/(eps*float64(max(1, m))*cnorm))
	} else {
		result.Set(3, zero)
	}
	//
	//     Generate random n-by-m matrix D and a copy DF
	//
	for j = 1; j <= (m); j++ {
		golapack.Zlarnv(2, &iseed, n, d.Off(0, j-1).CVector())
	}
	dnorm = golapack.Zlange('1', n, m, d, rwork)
	golapack.Zlacpy(Full, n, m, d, df)
	//
	//     Apply Q to D as D*Q
	//
	if err = golapack.Zgemqrt(Right, NoTrans, n, m, k, nb, af, t, df, work); err != nil {
		panic(err)
	}
	//
	//     Compute |D*Q - D*Q| / |D|
	//
	if err = df.Gemm(NoTrans, NoTrans, n, m, m, -one, d, q, one); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', n, m, df, rwork)
	if cnorm > zero {
		result.Set(4, resid/(eps*float64(max(1, m))*dnorm))
	} else {
		result.Set(4, zero)
	}
	//
	//     Copy D into DF again
	//
	golapack.Zlacpy(Full, n, m, d, df)
	//
	//     Apply Q to D as D*QT
	//
	if err = golapack.Zgemqrt(Right, ConjTrans, n, m, k, nb, af, t, df, work); err != nil {
		panic(err)
	}
	//
	//     Compute |D*QT - D*QT| / |D|
	//
	if err = df.Gemm(NoTrans, ConjTrans, n, m, m, -one, d, q, one); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', n, m, df, rwork)
	if cnorm > zero {
		result.Set(5, resid/(eps*float64(max(1, m))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
