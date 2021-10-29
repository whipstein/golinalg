package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dlqt05 tests DTPLQT and DTPMLQT.
func dlqt05(m, n, l, nb int, result *mat.Vector) {
	var anorm, cnorm, dnorm, eps, one, resid, zero float64
	var j, k, lwork, n2, np1 int
	var err error

	iseed := make([]int, 4)

	zero = 0.0
	one = 1.0

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)
	k = m
	n2 = m + n
	if n > 0 {
		np1 = m + 1
	} else {
		np1 = 1
	}
	lwork = n2 * n2 * nb

	//     Dynamically allocate all arrays
	// Allocate(A(m, &n2), Af(m, &n2), Q(&n2, &n2), R(&n2, &n2), Rwork(&n2), Work(&lwork), T(nb, m), C(&n2, m), Cf(&n2, m), D(m, &n2), Df(m, &n2))
	a := mf(m, n2, opts)
	af := mf(m, n2, opts)
	q := mf(n2, n2, opts)
	r := mf(n2, n2, opts)
	rwork := vf(n2)
	work := vf(lwork)
	t := mf(nb, m, opts)
	c := mf(n2, m, opts)
	cf := mf(n2, m, opts)
	d := mf(m, n2, opts)
	df := mf(m, n2, opts)

	//     Put random stuff into A
	// ldt = nb
	golapack.Dlaset('F', m, n2, zero, zero, a)
	golapack.Dlaset('F', nb, m, zero, zero, t)
	for j = 1; j <= m; j++ {
		golapack.Dlarnv(2, &iseed, m-j+1, a.Vector(j-1, j-1))
	}
	if n > 0 {
		for j = 1; j <= n-l; j++ {
			golapack.Dlarnv(2, &iseed, m, a.Vector(1-1, min(n+m, m+1)+j-1-1))
		}
	}
	if l > 0 {
		for j = 1; j <= l; j++ {
			golapack.Dlarnv(2, &iseed, m-j+1, a.Vector(j-1, min(n+m, n+m-l+1)+j-1-1))
		}
	}

	//     Copy the matrix A to the array AF.
	golapack.Dlacpy(Full, m, n2, a, af)

	//     Factor the matrix A in the array AF.
	if err = dtplqt(m, n, l, nb, af, af.Off(1-1, np1-1), t, work); err != nil {
		panic(err)
	}

	//     Generate the (M+N)-by-(M+N) matrix Q by applying H to I
	golapack.Dlaset(Full, n2, n2, zero, one, q)
	if err = golapack.Dgemlqt(Left, NoTrans, n2, n2, k, nb, af, t, q, work); err != nil {
		panic(err)
	}

	//     Copy L
	golapack.Dlaset(Full, n2, n2, zero, zero, r)
	golapack.Dlacpy(Lower, m, n2, af, r)

	//     Compute |L - A*Q*T| / |A| and store in RESULT(1)
	if err = goblas.Dgemm(NoTrans, Trans, m, n2, n2, -one, a, q, one, r); err != nil {
		panic(err)
	}
	anorm = golapack.Dlange('1', m, n2, a, rwork)
	resid = golapack.Dlange('1', m, n2, r, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*anorm*float64(max(1, n2))))
	} else {
		result.Set(0, zero)
	}

	//     Compute |I - Q*Q'| and store in RESULT(2)
	golapack.Dlaset(Full, n2, n2, zero, one, r)
	if err = goblas.Dsyrk(Upper, NoTrans, n2, n2, -one, q, one, r); err != nil {
		panic(err)
	}
	resid = golapack.Dlansy('1', Upper, n2, r, rwork)
	result.Set(1, resid/(eps*float64(max(1, n2))))

	//     Generate random m-by-n matrix C and a copy CF
	golapack.Dlaset('F', n2, m, zero, one, c)
	for j = 1; j <= m; j++ {
		golapack.Dlarnv(2, &iseed, n2, c.Vector(1-1, j-1))
	}
	cnorm = golapack.Dlange('1', n2, m, c, rwork)
	golapack.Dlacpy(Full, n2, m, c, cf)

	//     Apply Q to C as Q*C
	if err = golapack.Dtpmlqt(Left, NoTrans, n, m, k, l, nb, af.Off(1-1, np1-1), t, cf, cf.Off(np1-1, 1-1), work); err != nil {
		panic(err)
	}

	//     Compute |Q*C - Q*C| / |C|
	if err = goblas.Dgemm(NoTrans, NoTrans, n2, m, n2, -one, q, c, one, cf); err != nil {
		panic(err)
	}
	resid = golapack.Dlange('1', n2, m, cf, rwork)
	if cnorm > zero {
		result.Set(2, resid/(eps*float64(max(1, n2))*cnorm))
	} else {
		result.Set(2, zero)
	}

	//     Copy C into CF again
	golapack.Dlacpy(Full, n2, m, c, cf)

	//     Apply Q to C as QT*C
	if err = golapack.Dtpmlqt(Left, Trans, n, m, k, l, nb, af.Off(1-1, np1-1), t, cf, cf.Off(np1-1, 1-1), work); err != nil {
		panic(err)
	}

	//     Compute |QT*C - QT*C| / |C|
	if err = goblas.Dgemm(Trans, NoTrans, n2, m, n2, -one, q, c, one, cf); err != nil {
		panic(err)
	}
	resid = golapack.Dlange('1', n2, m, cf, rwork)
	if cnorm > zero {
		result.Set(3, resid/(eps*float64(max(1, n2))*cnorm))
	} else {
		result.Set(3, zero)
	}

	//     Generate random m-by-n matrix D and a copy DF
	for j = 1; j <= n2; j++ {
		golapack.Dlarnv(2, &iseed, m, d.Vector(1-1, j-1))
	}
	dnorm = golapack.Dlange('1', m, n2, d, rwork)
	golapack.Dlacpy(Full, m, n2, d, df)

	//     Apply Q to D as D*Q
	if err = golapack.Dtpmlqt(Right, NoTrans, m, n, k, l, nb, af.Off(1-1, np1-1), t, df, df.Off(1-1, np1-1), work); err != nil {
		panic(err)
	}

	//     Compute |D*Q - D*Q| / |D|
	if err = goblas.Dgemm(NoTrans, NoTrans, m, n2, n2, -one, d, q, one, df); err != nil {
		panic(err)
	}
	resid = golapack.Dlange('1', m, n2, df, rwork)
	if cnorm > zero {
		result.Set(4, resid/(eps*float64(max(1, n2))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy D into DF again
	golapack.Dlacpy(Full, m, n2, d, df)

	//     Apply Q to D as D*QT
	if err = golapack.Dtpmlqt(Right, Trans, m, n, k, l, nb, af.Off(1-1, np1-1), t, df, df.Off(1-1, np1-1), work); err != nil {
		panic(err)
	}

	//     Compute |D*QT - D*QT| / |D|
	if err = goblas.Dgemm(NoTrans, Trans, m, n2, n2, -one, d, q, one, df); err != nil {
		panic(err)
	}
	resid = golapack.Dlange('1', m, n2, df, rwork)
	if cnorm > zero {
		result.Set(5, resid/(eps*float64(max(1, n2))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
