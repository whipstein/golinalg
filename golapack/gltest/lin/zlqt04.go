package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
)

// Dlqt04 tests ZGELQT and ZUNMLQT.
func Zlqt04(m, n, nb *int, result *mat.Vector) {
	var czero, one complex128
	var anorm, cnorm, dnorm, eps, resid, zero float64
	var info, j, k, ldt, ll, lwork int
	iseed := make([]int, 4)

	zero = 0.0
	one = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)
	k = minint(*m, *n)
	ll = maxint(*m, *n)
	lwork = maxint(2, ll) * maxint(2, ll) * (*nb)

	//     Dynamically allocate local arrays
	a := cmf(*m, *n, opts)
	af := cmf(*m, *n, opts)
	q := cmf(*n, *n, opts)
	l := cmf(ll, *n, opts)
	rwork := vf(ll)
	work := cvf(lwork)
	t := cmf(*nb, *n, opts)
	c := cmf(*m, *n, opts)
	cf := cmf(*m, *n, opts)
	d := cmf(*n, *m, opts)
	df := cmf(*n, *m, opts)

	//     Put random numbers into A and copy to af
	ldt = (*nb)
	for j = 1; j <= (*n); j++ {
		golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, m, a.CVector(0, j-1))
	}
	golapack.Zlacpy('F', m, n, a, m, af, m)

	//     Factor the matrix A in the array af.
	golapack.Zgelqt(m, n, nb, af, m, t, &ldt, work, &info)

	//     Generate the n-by-n matrix Q
	golapack.Zlaset('F', n, n, &czero, &one, q, n)
	golapack.Zgemlqt('R', 'N', n, n, &k, nb, af, m, t, &ldt, q, n, work, &info)

	//     Copy L
	golapack.Zlaset('F', &ll, n, &czero, &czero, l, &ll)
	golapack.Zlacpy('L', m, n, af, m, l, &ll)

	//     Compute |L - A*Q'| / |A| and store in RESULT(1)
	goblas.Zgemm(NoTrans, ConjTrans, m, n, n, toPtrc128(-one), a, m, q, n, &one, l, &ll)
	anorm = golapack.Zlange('1', m, n, a, m, rwork)
	resid = golapack.Zlange('1', m, n, l, &ll, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*float64(maxint(1, *m))*anorm))
	} else {
		result.Set(0, zero)
	}

	//     Compute |I - Q'*Q| and store in RESULT(2)
	golapack.Zlaset('F', n, n, &czero, &one, l, &ll)
	goblas.Zherk(Upper, ConjTrans, n, n, toPtrf64(real(-one)), q, n, toPtrf64(real(one)), l, &ll)
	resid = golapack.Zlansy('1', 'U', n, l, &ll, rwork)
	result.Set(1, resid/(eps*float64(maxint(1, *n))))

	//     Generate random m-by-n matrix C and a copy CF
	for j = 1; j <= (*m); j++ {
		golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, n, d.CVector(0, j-1))
	}
	dnorm = golapack.Zlange('1', n, m, d, n, rwork)
	golapack.Zlacpy('F', n, m, d, n, df, n)

	//     Apply Q to C as Q*C
	golapack.Zgemlqt('L', 'N', n, m, &k, nb, af, m, t, nb, df, n, work, &info)

	//     Compute |Q*D - Q*D| / |D|
	goblas.Zgemm(NoTrans, NoTrans, n, m, n, toPtrc128(-one), q, n, d, n, &one, df, n)
	resid = golapack.Zlange('1', n, m, df, n, rwork)
	if dnorm > zero {
		result.Set(2, resid/(eps*float64(maxint(1, *m))*dnorm))
	} else {
		result.Set(2, zero)
	}

	//     Copy D into DF again
	golapack.Zlacpy('F', n, m, d, n, df, n)

	//     Apply Q to D as QT*D
	golapack.Zgemlqt('L', 'C', n, m, &k, nb, af, m, t, nb, df, n, work, &info)

	//     Compute |QT*D - QT*D| / |D|
	goblas.Zgemm(ConjTrans, NoTrans, n, m, n, toPtrc128(-one), q, n, d, n, &one, df, n)
	resid = golapack.Zlange('1', n, m, df, n, rwork)
	if dnorm > zero {
		result.Set(3, resid/(eps*float64(maxint(1, *m))*dnorm))
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
	golapack.Zgemlqt('R', 'N', m, n, &k, nb, af, m, t, nb, cf, m, work, &info)

	//     Compute |C*Q - C*Q| / |C|
	goblas.Zgemm(NoTrans, NoTrans, m, n, n, toPtrc128(-one), c, m, q, n, &one, cf, m)
	resid = golapack.Zlange('1', n, m, df, n, rwork)
	if cnorm > zero {
		result.Set(4, resid/(eps*float64(maxint(1, *m))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy C into CF again
	golapack.Zlacpy('F', m, n, c, m, cf, m)

	//     Apply Q to D as D*QT
	golapack.Zgemlqt('R', 'C', m, n, &k, nb, af, m, t, nb, cf, m, work, &info)

	//     Compute |C*QT - C*QT| / |C|
	goblas.Zgemm(NoTrans, ConjTrans, m, n, n, toPtrc128(-one), c, m, q, n, &one, cf, m)
	resid = golapack.Zlange('1', m, n, cf, m, rwork)
	if cnorm > zero {
		result.Set(5, resid/(eps*float64(maxint(1, *m))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
