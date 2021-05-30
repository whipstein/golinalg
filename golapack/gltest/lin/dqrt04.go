package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
)

// Dqrt04 tests DGEQRT and DGEMQRT.
func Dqrt04(m, n, nb *int, result *mat.Vector) {
	var anorm, cnorm, dnorm, eps, one, resid, zero float64
	var info, j, k, l, ldt, lwork int
	iseed := make([]int, 4)

	zero = 0.0
	one = 1.0

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)
	k = minint(*m, *n)
	l = maxint(*m, *n)
	lwork = maxint(2, l) * maxint(2, l) * (*nb)

	//     Dynamically allocate local arrays
	// Allocate(A(m, n), Af(m, n), Q(m, m), R(m, &l), Rwork(&l), Work(&lwork), T(nb, n), C(m, n), Cf(m, n), D(n, m), Df(n, m))
	a := mf(*m, *n, opts)
	af := mf(*m, *n, opts)
	q := mf(*m, *m, opts)
	r := mf(*m, l, opts)
	rwork := vf(l)
	work := vf(lwork)
	c := mf(*m, *n, opts)
	cf := mf(*m, *n, opts)
	d := mf(*n, *m, opts)
	df := mf(*n, *m, opts)

	//     Put random numbers into A and copy to AF
	ldt = (*nb)
	t := mf(ldt, (*nb)*(*n)/ldt, opts)
	for j = 1; j <= (*n); j++ {
		golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, m, a.Vector(1-1, j-1))
	}
	golapack.Dlacpy('F', m, n, a, m, af, m)

	//     Factor the matrix A in the array AF.
	golapack.Dgeqrt(m, n, nb, af, m, t, &ldt, work, &info)

	//     Generate the m-by-m matrix Q
	golapack.Dlaset('F', m, m, &zero, &one, q, m)
	golapack.Dgemqrt('R', 'N', m, m, &k, nb, af, m, t, &ldt, q, m, work, &info)

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
	golapack.Dgemqrt('L', 'N', m, n, &k, nb, af, m, t, nb, cf, m, work, &info)

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
	golapack.Dgemqrt('L', 'T', m, n, &k, nb, af, m, t, nb, cf, m, work, &info)

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
	golapack.Dgemqrt('R', 'N', n, m, &k, nb, af, m, t, nb, df, n, work, &info)

	//     Compute |D*Q - D*Q| / |D|
	goblas.Dgemm(NoTrans, NoTrans, n, m, m, toPtrf64(-one), d, n, q, m, &one, df, n)
	resid = golapack.Dlange('1', n, m, df, n, rwork)
	if cnorm > zero {
		result.Set(4, resid/(eps*float64(maxint(1, *m))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy D into DF again
	golapack.Dlacpy('F', n, m, d, n, df, n)

	//     Apply Q to D as D*QT
	golapack.Dgemqrt('R', 'T', n, m, &k, nb, af, m, t, nb, df, n, work, &info)

	//     Compute |D*QT - D*QT| / |D|
	goblas.Dgemm(NoTrans, Trans, n, m, m, toPtrf64(-one), d, n, q, m, &one, df, n)
	resid = golapack.Dlange('1', n, m, df, n, rwork)
	if cnorm > zero {
		result.Set(5, resid/(eps*float64(maxint(1, *m))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
