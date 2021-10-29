package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dqrt05 tests DTPQRT and DTPMQRT.
func dqrt05(m, n, l, nb int, result *mat.Vector) {
	var anorm, cnorm, dnorm, eps, one, resid, zero float64
	var j, k, lwork, m2, np1 int
	var err error

	iseed := make([]int, 4)

	zero = 0.0
	one = 1.0

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
	a := mf(m2, n, opts)
	af := mf(m2, n, opts)
	q := mf(m2, m2, opts)
	r := mf(m2, m2, opts)
	rwork := vf(m2)
	work := vf(lwork)
	t := mf(nb, n, opts)
	c := mf(m2, n, opts)
	cf := mf(m2, n, opts)
	d := mf(n, m2, opts)
	df := mf(n, m2, opts)

	//     Put random stuff into A
	// ldt = nb
	golapack.Dlaset('F', m2, n, zero, zero, a)
	golapack.Dlaset('F', nb, n, zero, zero, t)
	for j = 1; j <= n; j++ {
		golapack.Dlarnv(2, &iseed, j, a.Vector(0, j-1))
	}
	if m > 0 {
		for j = 1; j <= n; j++ {
			golapack.Dlarnv(2, &iseed, m-l, a.Vector(min(n+m, n+1)-1, j-1))
		}
	}
	if l > 0 {
		for j = 1; j <= n; j++ {
			golapack.Dlarnv(2, &iseed, min(j, l), a.Vector(min(n+m, n+m-l+1)-1, j-1))
		}
	}

	//     Copy the matrix A to the array af.
	golapack.Dlacpy(Full, m2, n, a, af)

	//     Factor the matrix A in the array af.
	if err = golapack.Dtpqrt(m, n, l, nb, af, af.Off(np1-1, 0), t, work); err != nil {
		panic(err)
	}

	//     Generate the (M+N)-by-(M+N) matrix Q by applying H to I
	golapack.Dlaset(Full, m2, m2, zero, one, q)
	if err = golapack.Dgemqrt(Right, NoTrans, m2, m2, k, nb, af, t, q, work); err != nil {
		panic(err)
	}

	//     Copy R
	golapack.Dlaset(Full, m2, n, zero, zero, r)
	golapack.Dlacpy(Upper, m2, n, af, r)

	//     Compute |R - Q'*A| / |A| and store in RESULT(1)
	if err = goblas.Dgemm(Trans, NoTrans, m2, n, m2, -one, q, a, one, r); err != nil {
		panic(err)
	}
	anorm = golapack.Dlange('1', m2, n, a, rwork)
	resid = golapack.Dlange('1', m2, n, r, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*anorm*float64(max(1, m2))))
	} else {
		result.Set(0, zero)
	}

	//     Compute |I - Q'*Q| and store in RESULT(2)
	golapack.Dlaset(Full, m2, m2, zero, one, r)
	if err = goblas.Dsyrk(Upper, ConjTrans, m2, m2, -one, q, one, r); err != nil {
		panic(err)
	}
	resid = golapack.Dlansy('1', Upper, m2, r, rwork)
	result.Set(1, resid/(eps*float64(max(1, m2))))

	//     Generate random m-by-n matrix C and a copy CF
	for j = 1; j <= n; j++ {
		golapack.Dlarnv(2, &iseed, m2, c.Vector(0, j-1))
	}
	cnorm = golapack.Dlange('1', m2, n, c, rwork)
	golapack.Dlacpy(Full, m2, n, c, cf)

	//     Apply Q to C as Q*C
	if err = golapack.Dtpmqrt(Left, NoTrans, m, n, k, l, nb, af.Off(np1-1, 0), t, cf, cf.Off(np1-1, 0), work); err != nil {
		panic(err)
	}

	//     Compute |Q*C - Q*C| / |C|
	if err = goblas.Dgemm(NoTrans, NoTrans, m2, n, m2, -one, q, c, one, cf); err != nil {
		panic(err)
	}
	resid = golapack.Dlange('1', m2, n, cf, rwork)
	if cnorm > zero {
		result.Set(2, resid/(eps*float64(max(1, m2))*cnorm))
	} else {
		result.Set(2, zero)
	}

	//     Copy C into CF again
	golapack.Dlacpy(Full, m2, n, c, cf)

	//     Apply Q to C as QT*C
	if err = golapack.Dtpmqrt(Left, Trans, m, n, k, l, nb, af.Off(np1-1, 0), t, cf, cf.Off(np1-1, 0), work); err != nil {
		panic(err)
	}

	//     Compute |QT*C - QT*C| / |C|
	if err = goblas.Dgemm(Trans, NoTrans, m2, n, m2, -one, q, c, one, cf); err != nil {
		panic(err)
	}
	resid = golapack.Dlange('1', m2, n, cf, rwork)
	if cnorm > zero {
		result.Set(3, resid/(eps*float64(max(1, m2))*cnorm))
	} else {
		result.Set(3, zero)
	}

	//     Generate random n-by-m matrix D and a copy DF
	for j = 1; j <= m2; j++ {
		golapack.Dlarnv(2, &iseed, n, d.Vector(0, j-1))
	}
	dnorm = golapack.Dlange('1', n, m2, d, rwork)
	golapack.Dlacpy(Full, n, m2, d, df)

	//     Apply Q to D as D*Q
	if err = golapack.Dtpmqrt(Right, NoTrans, n, m, n, l, nb, af.Off(np1-1, 0), t, df, df.Off(0, np1-1), work); err != nil {
		panic(err)
	}

	//     Compute |D*Q - D*Q| / |D|
	if err = goblas.Dgemm(NoTrans, NoTrans, n, m2, m2, -one, d, q, one, df); err != nil {
		panic(err)
	}
	resid = golapack.Dlange('1', n, m2, df, rwork)
	if cnorm > zero {
		result.Set(4, resid/(eps*float64(max(1, m2))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy D into DF again
	golapack.Dlacpy(Full, n, m2, d, df)

	//     Apply Q to D as D*QT
	if err = golapack.Dtpmqrt(Right, Trans, n, m, n, l, nb, af.Off(np1-1, 0), t, df, df.Off(0, np1-1), work); err != nil {
		panic(err)
	}

	//     Compute |D*QT - D*QT| / |D|
	if err = goblas.Dgemm(NoTrans, Trans, n, m2, m2, -one, d, q, one, df); err != nil {
		panic(err)
	}
	resid = golapack.Dlange('1', n, m2, df, rwork)
	if cnorm > zero {
		result.Set(5, resid/(eps*float64(max(1, m2))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
