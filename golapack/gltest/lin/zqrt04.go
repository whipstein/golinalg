package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zqrt04 tests ZGEQRT and ZGEMQRT.
func Zqrt04(m, n, nb *int, result *mat.Vector) {
	var czero, one complex128
	var anorm, cnorm, dnorm, eps, resid, zero float64
	var info, j, k, l, ldt, lwork int
	iseed := make([]int, 4)

	zero = 0.0
	one = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)
	k = minint(*m, *n)
	l = maxint(*m, *n)
	lwork = maxint(2, l) * maxint(2, l) * (*nb)

	//     Dynamically allocate local arrays
	a := cmf(*m, *n, opts)
	af := cmf(*m, *n, opts)
	q := cmf(*m, *m, opts)
	r := cmf(*m, l, opts)
	rwork := vf(l)
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
	golapack.Zgeqrt(m, n, nb, af, m, t, &ldt, work, &info)

	//     Generate the m-by-m matrix Q
	golapack.Zlaset('F', m, m, &czero, &one, q, m)
	golapack.Zgemqrt('R', 'N', m, m, &k, nb, af, m, t, &ldt, q, m, work, &info)

	//     Copy R
	golapack.Zlaset('F', m, n, &czero, &czero, r, m)
	golapack.Zlacpy('U', m, n, af, m, r, m)

	//     Compute |R - Q'*A| / |A| and store in RESULT(1)
	goblas.Zgemm(ConjTrans, NoTrans, m, n, m, toPtrc128(-one), q, m, a, m, &one, r, m)
	anorm = golapack.Zlange('1', m, n, a, m, rwork)
	resid = golapack.Zlange('1', m, n, r, m, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*float64(maxint(1, *m))*anorm))
	} else {
		result.Set(0, zero)
	}

	//     Compute |I - Q'*Q| and store in RESULT(2)
	golapack.Zlaset('F', m, m, &czero, &one, r, m)
	goblas.Zherk(Upper, ConjTrans, m, m, toPtrf64(real(-one)), q, m, toPtrf64(real(one)), r, m)
	resid = golapack.Zlansy('1', 'U', m, r, m, rwork)
	result.Set(1, resid/(eps*float64(maxint(1, *m))))
	//
	//     Generate random m-by-n matrix C and a copy CF
	//
	for j = 1; j <= (*n); j++ {
		golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, m, c.CVector(0, j-1))
	}
	cnorm = golapack.Zlange('1', m, n, c, m, rwork)
	golapack.Zlacpy('F', m, n, c, m, cf, m)
	//
	//     Apply Q to C as Q*C
	//
	golapack.Zgemqrt('L', 'N', m, n, &k, nb, af, m, t, nb, cf, m, work, &info)
	//
	//     Compute |Q*C - Q*C| / |C|
	//
	goblas.Zgemm(NoTrans, NoTrans, m, n, m, toPtrc128(-one), q, m, c, m, &one, cf, m)
	resid = golapack.Zlange('1', m, n, cf, m, rwork)
	if cnorm > zero {
		result.Set(2, resid/(eps*float64(maxint(1, *m))*cnorm))
	} else {
		result.Set(2, zero)
	}
	//
	//     Copy C into CF again
	//
	golapack.Zlacpy('F', m, n, c, m, cf, m)
	//
	//     Apply Q to C as QT*C
	//
	golapack.Zgemqrt('L', 'C', m, n, &k, nb, af, m, t, nb, cf, m, work, &info)
	//
	//     Compute |QT*C - QT*C| / |C|
	//
	goblas.Zgemm(ConjTrans, NoTrans, m, n, m, toPtrc128(-one), q, m, c, m, &one, cf, m)
	resid = golapack.Zlange('1', m, n, cf, m, rwork)
	if cnorm > zero {
		result.Set(3, resid/(eps*float64(maxint(1, *m))*cnorm))
	} else {
		result.Set(3, zero)
	}
	//
	//     Generate random n-by-m matrix D and a copy DF
	//
	for j = 1; j <= (*m); j++ {
		golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, n, d.CVector(0, j-1))
	}
	dnorm = golapack.Zlange('1', n, m, d, n, rwork)
	golapack.Zlacpy('F', n, m, d, n, df, n)
	//
	//     Apply Q to D as D*Q
	//
	golapack.Zgemqrt('R', 'N', n, m, &k, nb, af, m, t, nb, df, n, work, &info)
	//
	//     Compute |D*Q - D*Q| / |D|
	//
	goblas.Zgemm(NoTrans, NoTrans, n, m, m, toPtrc128(-one), d, n, q, m, &one, df, n)
	resid = golapack.Zlange('1', n, m, df, n, rwork)
	if cnorm > zero {
		result.Set(4, resid/(eps*float64(maxint(1, *m))*dnorm))
	} else {
		result.Set(4, zero)
	}
	//
	//     Copy D into DF again
	//
	golapack.Zlacpy('F', n, m, d, n, df, n)
	//
	//     Apply Q to D as D*QT
	//
	golapack.Zgemqrt('R', 'C', n, m, &k, nb, af, m, t, nb, df, n, work, &info)
	//
	//     Compute |D*QT - D*QT| / |D|
	//
	goblas.Zgemm(NoTrans, ConjTrans, n, m, m, toPtrc128(-one), d, n, q, m, &one, df, n)
	resid = golapack.Zlange('1', n, m, df, n, rwork)
	if cnorm > zero {
		result.Set(5, resid/(eps*float64(maxint(1, *m))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
