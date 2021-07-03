package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dqrt05 tests DTPQRT and DTPMQRT.
func Dqrt05(m, n, l, nb *int, result *mat.Vector) {
	var anorm, cnorm, dnorm, eps, one, resid, zero float64
	var info, j, k, ldt, lwork, m2, np1 int
	var err error
	_ = err

	iseed := make([]int, 4)

	zero = 0.0
	one = 1.0

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)
	k = (*n)
	m2 = (*m) + (*n)
	if (*m) > 0 {
		np1 = (*n) + 1
	} else {
		np1 = 1
	}
	lwork = m2 * m2 * (*nb)

	//     Dynamically allocate all arrays
	a := mf(m2, *n, opts)
	af := mf(m2, *n, opts)
	q := mf(m2, m2, opts)
	r := mf(m2, m2, opts)
	rwork := vf(m2)
	work := vf(lwork)
	t := mf(*nb, *n, opts)
	c := mf(m2, *n, opts)
	cf := mf(m2, *n, opts)
	d := mf(*n, m2, opts)
	df := mf(*n, m2, opts)

	//     Put random stuff into A
	ldt = (*nb)
	golapack.Dlaset('F', &m2, n, &zero, &zero, a, &m2)
	golapack.Dlaset('F', nb, n, &zero, &zero, t, nb)
	for j = 1; j <= (*n); j++ {
		golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, &j, a.Vector(0, j-1))
	}
	if (*m) > 0 {
		for j = 1; j <= (*n); j++ {
			golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr((*m)-(*l)), a.Vector(minint((*n)+(*m), (*n)+1)-1, j-1))
		}
	}
	if (*l) > 0 {
		for j = 1; j <= (*n); j++ {
			golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr(minint(j, *l)), a.Vector(minint((*n)+(*m), (*n)+(*m)-(*l)+1)-1, j-1))
		}
	}

	//     Copy the matrix A to the array af.
	golapack.Dlacpy('F', &m2, n, a, &m2, af, &m2)

	//     Factor the matrix A in the array af.
	golapack.Dtpqrt(m, n, l, nb, af, &m2, af.Off(np1-1, 0), &m2, t, &ldt, work, &info)

	//     Generate the (M+N)-by-(M+N) matrix Q by applying H to I
	golapack.Dlaset('F', &m2, &m2, &zero, &one, q, &m2)
	golapack.Dgemqrt('R', 'N', &m2, &m2, &k, nb, af, &m2, t, &ldt, q, &m2, work, &info)

	//     Copy R
	golapack.Dlaset('F', &m2, n, &zero, &zero, r, &m2)
	golapack.Dlacpy('U', &m2, n, af, &m2, r, &m2)

	//     Compute |R - Q'*A| / |A| and store in RESULT(1)
	err = goblas.Dgemm(Trans, NoTrans, m2, *n, m2, -one, q, m2, a, m2, one, r, m2)
	anorm = golapack.Dlange('1', &m2, n, a, &m2, rwork)
	resid = golapack.Dlange('1', &m2, n, r, &m2, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*anorm*float64(maxint(1, m2))))
	} else {
		result.Set(0, zero)
	}

	//     Compute |I - Q'*Q| and store in RESULT(2)
	golapack.Dlaset('F', &m2, &m2, &zero, &one, r, &m2)
	err = goblas.Dsyrk(Upper, ConjTrans, m2, m2, -one, q, m2, one, r, m2)
	resid = golapack.Dlansy('1', 'U', &m2, r, &m2, rwork)
	result.Set(1, resid/(eps*float64(maxint(1, m2))))

	//     Generate random m-by-n matrix C and a copy CF
	for j = 1; j <= (*n); j++ {
		golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, &m2, c.Vector(0, j-1))
	}
	cnorm = golapack.Dlange('1', &m2, n, c, &m2, rwork)
	golapack.Dlacpy('F', &m2, n, c, &m2, cf, &m2)

	//     Apply Q to C as Q*C
	golapack.Dtpmqrt('L', 'N', m, n, &k, l, nb, af.Off(np1-1, 0), &m2, t, &ldt, cf, &m2, cf.Off(np1-1, 0), &m2, work, &info)

	//     Compute |Q*C - Q*C| / |C|
	err = goblas.Dgemm(NoTrans, NoTrans, m2, *n, m2, -one, q, m2, c, m2, one, cf, m2)
	resid = golapack.Dlange('1', &m2, n, cf, &m2, rwork)
	if cnorm > zero {
		result.Set(2, resid/(eps*float64(maxint(1, m2))*cnorm))
	} else {
		result.Set(2, zero)
	}

	//     Copy C into CF again
	golapack.Dlacpy('F', &m2, n, c, &m2, cf, &m2)

	//     Apply Q to C as QT*C
	golapack.Dtpmqrt('L', 'T', m, n, &k, l, nb, af.Off(np1-1, 0), &m2, t, &ldt, cf, &m2, cf.Off(np1-1, 0), &m2, work, &info)

	//     Compute |QT*C - QT*C| / |C|
	err = goblas.Dgemm(Trans, NoTrans, m2, *n, m2, -one, q, m2, c, m2, one, cf, m2)
	resid = golapack.Dlange('1', &m2, n, cf, &m2, rwork)
	if cnorm > zero {
		result.Set(3, resid/(eps*float64(maxint(1, m2))*cnorm))
	} else {
		result.Set(3, zero)
	}

	//     Generate random n-by-m matrix D and a copy DF
	for j = 1; j <= m2; j++ {
		golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, n, d.Vector(0, j-1))
	}
	dnorm = golapack.Dlange('1', n, &m2, d, n, rwork)
	golapack.Dlacpy('F', n, &m2, d, n, df, n)

	//     Apply Q to D as D*Q
	golapack.Dtpmqrt('R', 'N', n, m, n, l, nb, af.Off(np1-1, 0), &m2, t, &ldt, df, n, df.Off(0, np1-1), n, work, &info)

	//     Compute |D*Q - D*Q| / |D|
	err = goblas.Dgemm(NoTrans, NoTrans, *n, m2, m2, -one, d, *n, q, m2, one, df, *n)
	resid = golapack.Dlange('1', n, &m2, df, n, rwork)
	if cnorm > zero {
		result.Set(4, resid/(eps*float64(maxint(1, m2))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy D into DF again
	golapack.Dlacpy('F', n, &m2, d, n, df, n)

	//     Apply Q to D as D*QT
	golapack.Dtpmqrt('R', 'T', n, m, n, l, nb, af.Off(np1-1, 0), &m2, t, &ldt, df, n, df.Off(0, np1-1), n, work, &info)

	//     Compute |D*QT - D*QT| / |D|
	err = goblas.Dgemm(NoTrans, Trans, *n, m2, m2, -one, d, *n, q, m2, one, df, *n)
	resid = golapack.Dlange('1', n, &m2, df, n, rwork)
	if cnorm > zero {
		result.Set(5, resid/(eps*float64(maxint(1, m2))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
