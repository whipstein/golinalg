package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dlqt04 tests DGELQT and DGEMLQT.
func dlqt04(m, n, nb int, result *mat.Vector) {
	var anorm, cnorm, dnorm, eps, one, resid, zero float64
	var j, k, ll, lwork int
	var err error

	iseed := make([]int, 4)

	zero = 0.0
	one = 1.0

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)
	k = min(m, n)
	ll = max(m, n)
	lwork = max(2, ll) * max(2, ll) * nb

	//     Dynamically allocate local arrays
	// Allocate(A(m, n), Af(m, n), Q(n, n), L(&ll, n), Rwork(&ll), Work(&lwork), T(nb, n), C(m, n), Cf(m, n), D(n, m), Df(n, m))
	a := mf(m, n, opts)
	af := mf(m, n, opts)
	q := mf(n, n, opts)
	l := mf(ll, n, opts)
	rwork := vf(ll)
	work := vf(lwork)
	t := mf(nb, n, opts)
	c := mf(m, n, opts)
	cf := mf(m, n, opts)
	d := mf(n, m, opts)
	df := mf(n, m, opts)

	//     Put random numbers into A and copy to AF
	for j = 1; j <= n; j++ {
		golapack.Dlarnv(2, &iseed, m, a.Vector(1-1, j-1))
	}
	golapack.Dlacpy(Full, m, n, a, af)

	//     Factor the matrix A in the array AF.
	if err = golapack.Dgelqt(m, n, nb, af, t, work); err != nil {
		panic(err)
	}

	//     Generate the n-by-n matrix Q
	golapack.Dlaset(Full, n, n, zero, one, q)
	if err = golapack.Dgemlqt(Right, NoTrans, n, n, k, nb, af, t, q, work); err != nil {
		panic(err)
	}

	//     Copy R
	golapack.Dlaset(Full, m, n, zero, zero, l)
	golapack.Dlacpy(Lower, m, n, af, l)

	//     Compute |L - A*Q'| / |A| and store in RESULT(1)
	if err = goblas.Dgemm(NoTrans, Trans, m, n, n, -one, a, q, one, l); err != nil {
		panic(err)
	}
	anorm = golapack.Dlange('1', m, n, a, rwork)
	resid = golapack.Dlange('1', m, n, l, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*float64(max(1, m))*anorm))
	} else {
		result.Set(0, zero)
	}

	//     Compute |I - Q'*Q| and store in RESULT(2)
	golapack.Dlaset(Full, n, n, zero, one, l)
	if err = goblas.Dsyrk(Upper, ConjTrans, n, n, -one, q, one, l); err != nil {
		panic(err)
	}
	resid = golapack.Dlansy('1', Upper, n, l, rwork)
	result.Set(1, resid/(eps*float64(max(1, n))))

	//     Generate random m-by-n matrix C and a copy CF
	for j = 1; j <= m; j++ {
		golapack.Dlarnv(2, &iseed, n, d.Vector(1-1, j-1))
	}
	dnorm = golapack.Dlange('1', n, m, d, rwork)
	golapack.Dlacpy(Full, n, m, d, df)

	//     Apply Q to C as Q*C
	if err = golapack.Dgemlqt(Left, NoTrans, n, m, k, nb, af, t, df, work); err != nil {
		panic(err)
	}

	//     Compute |Q*D - Q*D| / |D|
	if err = goblas.Dgemm(NoTrans, NoTrans, n, m, n, -one, q, d, one, df); err != nil {
		panic(err)
	}
	resid = golapack.Dlange('1', n, m, df, rwork)
	if dnorm > zero {
		result.Set(2, resid/(eps*float64(max(1, m))*dnorm))
	} else {
		result.Set(2, zero)
	}

	//     Copy D into DF again
	golapack.Dlacpy(Full, n, m, d, df)

	//     Apply Q to D as QT*D
	if err = golapack.Dgemlqt(Left, Trans, n, m, k, nb, af, t, df, work); err != nil {
		panic(err)
	}

	//     Compute |QT*D - QT*D| / |D|
	if err = goblas.Dgemm(Trans, NoTrans, n, m, n, -one, q, d, one, df); err != nil {
		panic(err)
	}
	resid = golapack.Dlange('1', n, m, df, rwork)
	if dnorm > zero {
		result.Set(3, resid/(eps*float64(max(1, m))*dnorm))
	} else {
		result.Set(3, zero)
	}

	//     Generate random n-by-m matrix D and a copy DF
	for j = 1; j <= n; j++ {
		golapack.Dlarnv(2, &iseed, m, c.Vector(1-1, j-1))
	}
	cnorm = golapack.Dlange('1', m, n, c, rwork)
	golapack.Dlacpy(Full, m, n, c, cf)

	//     Apply Q to C as C*Q
	if err = golapack.Dgemlqt(Right, NoTrans, m, n, k, nb, af, t, cf, work); err != nil {
		panic(err)
	}

	//     Compute |C*Q - C*Q| / |C|
	if err = goblas.Dgemm(NoTrans, NoTrans, m, n, n, -one, c, q, one, cf); err != nil {
		panic(err)
	}
	resid = golapack.Dlange('1', n, m, df, rwork)
	if cnorm > zero {
		result.Set(4, resid/(eps*float64(max(1, m))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy C into CF again
	golapack.Dlacpy(Full, m, n, c, cf)

	//     Apply Q to D as D*QT
	if err = golapack.Dgemlqt(Right, Trans, m, n, k, nb, af, t, cf, work); err != nil {
		panic(err)
	}

	//     Compute |C*QT - C*QT| / |C|
	if err = goblas.Dgemm(NoTrans, Trans, m, n, n, -one, c, q, one, cf); err != nil {
		panic(err)
	}
	resid = golapack.Dlange('1', m, n, cf, rwork)
	if cnorm > zero {
		result.Set(5, resid/(eps*float64(max(1, m))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
