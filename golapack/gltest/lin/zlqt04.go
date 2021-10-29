package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zlqt04 tests ZGELQT and ZUNMLQT.
func zlqt04(m, n, nb int, result *mat.Vector) {
	var czero, one complex128
	var anorm, cnorm, dnorm, eps, resid, zero float64
	var j, k, ll, lwork int
	var err error

	iseed := make([]int, 4)

	zero = 0.0
	one = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)
	k = min(m, n)
	ll = max(m, n)
	lwork = max(2, ll) * max(2, ll) * nb

	//     Dynamically allocate local arrays
	a := cmf(m, n, opts)
	af := cmf(m, n, opts)
	q := cmf(n, n, opts)
	l := cmf(ll, n, opts)
	rwork := vf(ll)
	work := cvf(lwork)
	t := cmf(nb, n, opts)
	c := cmf(m, n, opts)
	cf := cmf(m, n, opts)
	d := cmf(n, m, opts)
	df := cmf(n, m, opts)

	//     Put random numbers into A and copy to af
	// ldt = nb
	for j = 1; j <= n; j++ {
		golapack.Zlarnv(2, &iseed, m, a.CVector(0, j-1))
	}
	golapack.Zlacpy(Full, m, n, a, af)

	//     Factor the matrix A in the array af.
	if err = golapack.Zgelqt(m, n, nb, af, t, work); err != nil {
		panic(err)
	}

	//     Generate the n-by-n matrix Q
	golapack.Zlaset(Full, n, n, czero, one, q)
	if err = golapack.Zgemlqt(Right, NoTrans, n, n, k, nb, af, t, q, work); err != nil {
		panic(err)
	}

	//     Copy L
	golapack.Zlaset(Full, ll, n, czero, czero, l)
	golapack.Zlacpy(Lower, m, n, af, l)

	//     Compute |L - A*Q'| / |A| and store in RESULT(1)
	if err = goblas.Zgemm(NoTrans, ConjTrans, m, n, n, -one, a, q, one, l); err != nil {
		panic(err)
	}
	anorm = golapack.Zlange('1', m, n, a, rwork)
	resid = golapack.Zlange('1', m, n, l, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*float64(max(1, m))*anorm))
	} else {
		result.Set(0, zero)
	}

	//     Compute |I - Q'*Q| and store in RESULT(2)
	golapack.Zlaset(Full, n, n, czero, one, l)
	if err = goblas.Zherk(Upper, ConjTrans, n, n, real(-one), q, real(one), l); err != nil {
		panic(err)
	}
	resid = golapack.Zlansy('1', Upper, n, l, rwork)
	result.Set(1, resid/(eps*float64(max(1, n))))

	//     Generate random m-by-n matrix C and a copy CF
	for j = 1; j <= (m); j++ {
		golapack.Zlarnv(2, &iseed, n, d.CVector(0, j-1))
	}
	dnorm = golapack.Zlange('1', n, m, d, rwork)
	golapack.Zlacpy(Full, n, m, d, df)

	//     Apply Q to C as Q*C
	if err = golapack.Zgemlqt(Left, NoTrans, n, m, k, nb, af, t, df, work); err != nil {
		panic(err)
	}

	//     Compute |Q*D - Q*D| / |D|
	if err = goblas.Zgemm(NoTrans, NoTrans, n, m, n, -one, q, d, one, df); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', n, m, df, rwork)
	if dnorm > zero {
		result.Set(2, resid/(eps*float64(max(1, m))*dnorm))
	} else {
		result.Set(2, zero)
	}

	//     Copy D into DF again
	golapack.Zlacpy(Full, n, m, d, df)

	//     Apply Q to D as QT*D
	if err = golapack.Zgemlqt(Left, ConjTrans, n, m, k, nb, af, t, df, work); err != nil {
		panic(err)
	}

	//     Compute |QT*D - QT*D| / |D|
	if err = goblas.Zgemm(ConjTrans, NoTrans, n, m, n, -one, q, d, one, df); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', n, m, df, rwork)
	if dnorm > zero {
		result.Set(3, resid/(eps*float64(max(1, m))*dnorm))
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
	if err = golapack.Zgemlqt(Right, NoTrans, m, n, k, nb, af, t, cf, work); err != nil {
		panic(err)
	}

	//     Compute |C*Q - C*Q| / |C|
	if err = goblas.Zgemm(NoTrans, NoTrans, m, n, n, -one, c, q, one, cf); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', n, m, df, rwork)
	if cnorm > zero {
		result.Set(4, resid/(eps*float64(max(1, m))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy C into CF again
	golapack.Zlacpy(Full, m, n, c, cf)

	//     Apply Q to D as D*QT
	if err = golapack.Zgemlqt(Right, ConjTrans, m, n, k, nb, af, t, cf, work); err != nil {
		panic(err)
	}

	//     Compute |C*QT - C*QT| / |C|
	if err = goblas.Zgemm(NoTrans, ConjTrans, m, n, n, -one, c, q, one, cf); err != nil {
		panic(err)
	}
	resid = golapack.Zlange('1', m, n, cf, rwork)
	if cnorm > zero {
		result.Set(5, resid/(eps*float64(max(1, m))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
