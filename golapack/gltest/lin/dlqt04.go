package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dlqt04 tests DGELQT and DGEMLQT.
func Dlqt04(m, n, nb *int, result *mat.Vector) {
	var anorm, cnorm, dnorm, eps, one, resid, zero float64
	var info, j, k, ldt, ll, lwork int
	var err error
	_ = err

	iseed := make([]int, 4)

	zero = 0.0
	one = 1.0

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)
	k = min(*m, *n)
	ll = max(*m, *n)
	lwork = max(2, ll) * max(2, ll) * (*nb)

	//     Dynamically allocate local arrays
	// Allocate(A(m, n), Af(m, n), Q(n, n), L(&ll, n), Rwork(&ll), Work(&lwork), T(nb, n), C(m, n), Cf(m, n), D(n, m), Df(n, m))
	a := mf(*m, *n, opts)
	af := mf(*m, *n, opts)
	q := mf(*n, *n, opts)
	l := mf(ll, *n, opts)
	rwork := vf(ll)
	work := vf(lwork)
	t := mf(*nb, *n, opts)
	c := mf(*m, *n, opts)
	cf := mf(*m, *n, opts)
	d := mf(*n, *m, opts)
	df := mf(*n, *m, opts)

	//     Put random numbers into A and copy to AF
	ldt = (*nb)
	for j = 1; j <= (*n); j++ {
		golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, m, a.Vector(1-1, j-1))
	}
	golapack.Dlacpy('F', m, n, a, m, af, m)

	//     Factor the matrix A in the array AF.
	golapack.Dgelqt(m, n, nb, af, m, t, &ldt, work, &info)

	//     Generate the n-by-n matrix Q
	golapack.Dlaset('F', n, n, &zero, &one, q, n)
	golapack.Dgemlqt('R', 'N', n, n, &k, nb, af, m, t, &ldt, q, n, work, &info)

	//     Copy R
	golapack.Dlaset('F', m, n, &zero, &zero, l, &ll)
	golapack.Dlacpy('L', m, n, af, m, l, &ll)

	//     Compute |L - A*Q'| / |A| and store in RESULT(1)
	err = goblas.Dgemm(NoTrans, Trans, *m, *n, *n, -one, a, q, one, l)
	anorm = golapack.Dlange('1', m, n, a, m, rwork)
	resid = golapack.Dlange('1', m, n, l, &ll, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*float64(max(1, *m))*anorm))
	} else {
		result.Set(0, zero)
	}

	//     Compute |I - Q'*Q| and store in RESULT(2)
	golapack.Dlaset('F', n, n, &zero, &one, l, &ll)
	err = goblas.Dsyrk(Upper, ConjTrans, *n, *n, -one, q, one, l)
	resid = golapack.Dlansy('1', 'U', n, l, &ll, rwork)
	result.Set(1, resid/(eps*float64(max(1, *n))))

	//     Generate random m-by-n matrix C and a copy CF
	for j = 1; j <= (*m); j++ {
		golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, n, d.Vector(1-1, j-1))
	}
	dnorm = golapack.Dlange('1', n, m, d, n, rwork)
	golapack.Dlacpy('F', n, m, d, n, df, n)

	//     Apply Q to C as Q*C
	golapack.Dgemlqt('L', 'N', n, m, &k, nb, af, m, t, nb, df, n, work, &info)

	//     Compute |Q*D - Q*D| / |D|
	err = goblas.Dgemm(NoTrans, NoTrans, *n, *m, *n, -one, q, d, one, df)
	resid = golapack.Dlange('1', n, m, df, n, rwork)
	if dnorm > zero {
		result.Set(2, resid/(eps*float64(max(1, *m))*dnorm))
	} else {
		result.Set(2, zero)
	}

	//     Copy D into DF again
	golapack.Dlacpy('F', n, m, d, n, df, n)

	//     Apply Q to D as QT*D
	golapack.Dgemlqt('L', 'T', n, m, &k, nb, af, m, t, nb, df, n, work, &info)

	//     Compute |QT*D - QT*D| / |D|
	err = goblas.Dgemm(Trans, NoTrans, *n, *m, *n, -one, q, d, one, df)
	resid = golapack.Dlange('1', n, m, df, n, rwork)
	if dnorm > zero {
		result.Set(3, resid/(eps*float64(max(1, *m))*dnorm))
	} else {
		result.Set(3, zero)
	}

	//     Generate random n-by-m matrix D and a copy DF
	for j = 1; j <= (*n); j++ {
		golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, m, c.Vector(1-1, j-1))
	}
	cnorm = golapack.Dlange('1', m, n, c, m, rwork)
	golapack.Dlacpy('F', m, n, c, m, cf, m)

	//     Apply Q to C as C*Q
	golapack.Dgemlqt('R', 'N', m, n, &k, nb, af, m, t, nb, cf, m, work, &info)

	//     Compute |C*Q - C*Q| / |C|
	err = goblas.Dgemm(NoTrans, NoTrans, *m, *n, *n, -one, c, q, one, cf)
	resid = golapack.Dlange('1', n, m, df, n, rwork)
	if cnorm > zero {
		result.Set(4, resid/(eps*float64(max(1, *m))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy C into CF again
	golapack.Dlacpy('F', m, n, c, m, cf, m)

	//     Apply Q to D as D*QT
	golapack.Dgemlqt('R', 'T', m, n, &k, nb, af, m, t, nb, cf, m, work, &info)

	//     Compute |C*QT - C*QT| / |C|
	err = goblas.Dgemm(NoTrans, Trans, *m, *n, *n, -one, c, q, one, cf)
	resid = golapack.Dlange('1', m, n, cf, m, rwork)
	if cnorm > zero {
		result.Set(5, resid/(eps*float64(max(1, *m))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
