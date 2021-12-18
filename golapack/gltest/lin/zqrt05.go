package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zqrt05 tests ZTPQRT and ZTPMQRT.
func zqrt05(m, n, l, nb int, result *mat.Vector) {
	var czero, one complex128
	var anorm, cnorm, dnorm, eps, resid, zero float64
	var j, k, lwork, m2, np1 int
	var err error

	iseed := make([]int, 4)

	zero = 0.0
	one = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)
	k = n
	m2 = m + n
	if m > 0 {
		np1 = n + 1
	} else {
		np1 = 1
	}
	lwork = m2 * m2 * nb

	//     Dynamically allocate all arrays
	a := cmf(m2, n, opts)
	af := cmf(m2, n, opts)
	q := cmf(m2, m2, opts)
	r := cmf(m2, m2, opts)
	rwork := vf(m2)
	work := cvf(lwork)
	t := cmf(nb, n, opts)
	c := cmf(m2, n, opts)
	cf := cmf(m2, n, opts)
	d := cmf(n, m2, opts)
	df := cmf(n, m2, opts)

	//     Put random stuff into A
	golapack.Zlaset(Full, m2, n, czero, czero, a)
	golapack.Zlaset(Full, nb, n, czero, czero, t)
	for j = 1; j <= n; j++ {
		golapack.Zlarnv(2, &iseed, j, a.Off(0, j-1).CVector())
	}
	if m > 0 {
		for j = 1; j <= n; j++ {
			golapack.Zlarnv(2, &iseed, m-l, a.Off(min(n+m, n+1)-1, j-1).CVector())
		}
	}
	if l > 0 {
		for j = 1; j <= n; j++ {
			golapack.Zlarnv(2, &iseed, min(j, l), a.Off(min(n+m, n+m-l+1)-1, j-1).CVector())
		}
	}

	//     Copy the matrix A to the array af.
	golapack.Zlacpy(Full, m2, n, a, af)

	//     Factor the matrix A in the array af.
	if err = golapack.Ztpqrt(m, n, l, nb, af, af.Off(np1-1, 0), t, work); err != nil {
		panic(err)
	}

	//     Generate the (M+N)-by-(M+N) matrix Q by applying H to I
	golapack.Zlaset(Full, m2, m2, czero, one, q)
	if err = golapack.Zgemqrt(Right, NoTrans, m2, m2, k, nb, af, t, q, work); err != nil {
		panic(err)
	}

	//     Copy R
	golapack.Zlaset(Full, m2, n, czero, czero, r)
	golapack.Zlacpy(Upper, m2, n, af, r)

	//     Compute |R - Q'*A| / |A| and store in RESULT(1)
	if err = r.Gemm(ConjTrans, NoTrans, m2, n, m2, -one, q, a, one); err != nil {
		panic(err)
	}
	anorm = golapack.Zlange('1', m2, n, a, rwork)
	resid = golapack.Zlange('1', m2, n, r, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*anorm*float64(max(1, m2))))
	} else {
		result.Set(0, zero)
	}

	//     Compute |I - Q'*Q| and store in RESULT(2)
	golapack.Zlaset(Full, m2, m2, czero, one, r)
	if err = r.Herk(Upper, ConjTrans, m2, m2, real(-one), q, real(one)); err != nil {
		panic(err)
	}
	resid = golapack.Zlansy('1', Upper, m2, r, rwork)
	result.Set(1, resid/(eps*float64(max(1, m2))))

	//     Generate random m-by-n matrix C and a copy CF
	for j = 1; j <= n; j++ {
		golapack.Zlarnv(2, &iseed, m2, c.Off(0, j-1).CVector())
	}
	cnorm = golapack.Zlange('1', m2, n, c, rwork)
	golapack.Zlacpy(Full, m2, n, c, cf)

	//     Apply Q to C as Q*C
	if err = golapack.Ztpmqrt(Left, NoTrans, m, n, k, l, nb, af.Off(np1-1, 0), t, cf, cf.Off(np1-1, 0), work); err != nil {
		panic(err)
	}

	//     Compute |Q*C - Q*C| / |C|
	if err = cf.Gemm(NoTrans, NoTrans, m2, n, m2, -one, q, c, one); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', m2, n, cf, rwork)
	if cnorm > zero {
		result.Set(2, resid/(eps*float64(max(1, m2))*cnorm))
	} else {
		result.Set(2, zero)
	}

	//     Copy C into CF again
	golapack.Zlacpy(Full, m2, n, c, cf)

	//     Apply Q to C as QT*C
	if err = golapack.Ztpmqrt(Left, ConjTrans, m, n, k, l, nb, af.Off(np1-1, 0), t, cf, cf.Off(np1-1, 0), work); err != nil {
		panic(err)
	}

	//     Compute |QT*C - QT*C| / |C|
	if err = cf.Gemm(ConjTrans, NoTrans, m2, n, m2, -one, q, c, one); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', m2, n, cf, rwork)
	if cnorm > zero {
		result.Set(3, resid/(eps*float64(max(1, m2))*cnorm))
	} else {
		result.Set(3, zero)
	}

	//     Generate random n-by-m matrix D and a copy DF
	for j = 1; j <= m2; j++ {
		golapack.Zlarnv(2, &iseed, n, d.Off(0, j-1).CVector())
	}
	dnorm = golapack.Zlange('1', n, m2, d, rwork)
	golapack.Zlacpy(Full, n, m2, d, df)

	//     Apply Q to D as D*Q
	if err = golapack.Ztpmqrt(Right, NoTrans, n, m, n, l, nb, af.Off(np1-1, 0), t, df, df.Off(0, np1-1), work); err != nil {
		panic(err)
	}

	//     Compute |D*Q - D*Q| / |D|
	if err = df.Gemm(NoTrans, NoTrans, n, m2, m2, -one, d, q, one); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', n, m2, df, rwork)
	if cnorm > zero {
		result.Set(4, resid/(eps*float64(max(1, m2))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy D into DF again
	golapack.Zlacpy(Full, n, m2, d, df)

	//     Apply Q to D as D*QT
	if err = golapack.Ztpmqrt(Right, ConjTrans, n, m, n, l, nb, af.Off(np1-1, 0), t, df, df.Off(0, np1-1), work); err != nil {
		panic(err)
	}

	//     Compute |D*QT - D*QT| / |D|
	if err = df.Gemm(NoTrans, ConjTrans, n, m2, m2, -one, d, q, one); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', n, m2, df, rwork)
	if cnorm > zero {
		result.Set(5, resid/(eps*float64(max(1, m2))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
