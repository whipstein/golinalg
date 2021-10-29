package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dqrt04 tests DGEQRT and DGEMQRT.
func dqrt04(m, n, nb int, result *mat.Vector) {
	var anorm, cnorm, dnorm, eps, one, resid, zero float64
	var j, k, l, ldt, lwork int
	var err error

	iseed := make([]int, 4)

	zero = 0.0
	one = 1.0

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)
	k = min(m, n)
	l = max(m, n)
	lwork = max(2, l) * max(2, l) * nb

	//     Dynamically allocate local arrays
	// Allocate(A(m, n), Af(m, n), Q(m, m), R(m, &l), Rwork(&l), Work(&lwork), T(nb, n), C(m, n), Cf(m, n), D(n, m), Df(n, m))
	a := mf(m, n, opts)
	af := mf(m, n, opts)
	q := mf(m, m, opts)
	r := mf(m, l, opts)
	rwork := vf(l)
	work := vf(lwork)
	c := mf(m, n, opts)
	cf := mf(m, n, opts)
	d := mf(n, m, opts)
	df := mf(n, m, opts)

	//     Put random numbers into A and copy to AF
	ldt = nb
	t := mf(ldt, nb*n/ldt, opts)
	for j = 1; j <= n; j++ {
		golapack.Dlarnv(2, &iseed, m, a.Vector(1-1, j-1))
	}
	golapack.Dlacpy(Full, m, n, a, af)

	//     Factor the matrix A in the array AF.
	if err = golapack.Dgeqrt(m, n, nb, af, t, work); err != nil {
		panic(err)
	}

	//     Generate the m-by-m matrix Q
	golapack.Dlaset(Full, m, m, zero, one, q)
	if err = golapack.Dgemqrt(Right, NoTrans, m, m, k, nb, af, t, q, work); err != nil {
		panic(err)
	}

	//     Copy R
	golapack.Dlaset(Full, m, n, zero, zero, r)
	golapack.Dlacpy(Upper, m, n, af, r)

	//     Compute |R - Q'*A| / |A| and store in RESULT(1)
	if err = goblas.Dgemm(Trans, NoTrans, m, n, m, -one, q, a, one, r); err != nil {
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
	golapack.Dlaset(Full, m, m, zero, one, r)
	if err = goblas.Dsyrk(Upper, ConjTrans, m, m, -one, q, one, r); err != nil {
		panic(err)
	}
	resid = golapack.Dlansy('1', Upper, m, r, rwork)
	result.Set(1, resid/(eps*float64(max(1, m))))

	//     Generate random m-by-n matrix C and a copy CF
	for j = 1; j <= n; j++ {
		golapack.Dlarnv(2, &iseed, m, c.Vector(1-1, j-1))
	}
	cnorm = golapack.Dlange('1', m, n, c, rwork)
	golapack.Dlacpy(Full, m, n, c, cf)

	//     Apply Q to C as Q*C
	if err = golapack.Dgemqrt(Left, NoTrans, m, n, k, nb, af, t, cf, work); err != nil {
		panic(err)
	}

	//     Compute |Q*C - Q*C| / |C|
	if err = goblas.Dgemm(NoTrans, NoTrans, m, n, m, -one, q, c, one, cf); err != nil {
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
	if err = golapack.Dgemqrt(Left, Trans, m, n, k, nb, af, t, cf, work); err != nil {
		panic(err)
	}

	//     Compute |QT*C - QT*C| / |C|
	if err = goblas.Dgemm(Trans, NoTrans, m, n, m, -one, q, c, one, cf); err != nil {
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
		golapack.Dlarnv(2, &iseed, n, d.Vector(1-1, j-1))
	}
	dnorm = golapack.Dlange('1', n, m, d, rwork)
	golapack.Dlacpy(Full, n, m, d, df)

	//     Apply Q to D as D*Q
	if err = golapack.Dgemqrt(Right, NoTrans, n, m, k, nb, af, t, df, work); err != nil {
		panic(err)
	}

	//     Compute |D*Q - D*Q| / |D|
	if err = goblas.Dgemm(NoTrans, NoTrans, n, m, m, -one, d, q, one, df); err != nil {
		panic(err)
	}
	resid = golapack.Dlange('1', n, m, df, rwork)
	if cnorm > zero {
		result.Set(4, resid/(eps*float64(max(1, m))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy D into DF again
	golapack.Dlacpy(Full, n, m, d, df)

	//     Apply Q to D as D*QT
	if err = golapack.Dgemqrt(Right, Trans, n, m, k, nb, af, t, df, work); err != nil {
		panic(err)
	}

	//     Compute |D*QT - D*QT| / |D|
	if err = goblas.Dgemm(NoTrans, Trans, n, m, m, -one, d, q, one, df); err != nil {
		panic(err)
	}
	resid = golapack.Dlange('1', n, m, df, rwork)
	if cnorm > zero {
		result.Set(5, resid/(eps*float64(max(1, m))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
