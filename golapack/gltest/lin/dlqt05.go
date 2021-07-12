package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dlqt05 tests DTPLQT and DTPMLQT.
func Dlqt05(m *int, n *int, l *int, nb *int, result *mat.Vector) {
	var anorm, cnorm, dnorm, eps, one, resid, zero float64
	var info, j, k, ldt, lwork, n2, np1 int
	var err error
	_ = err

	iseed := make([]int, 4)

	zero = 0.0
	one = 1.0

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)
	k = (*m)
	n2 = (*m) + (*n)
	if (*n) > 0 {
		np1 = (*m) + 1
	} else {
		np1 = 1
	}
	lwork = n2 * n2 * (*nb)

	//     Dynamically allocate all arrays
	// Allocate(A(m, &n2), Af(m, &n2), Q(&n2, &n2), R(&n2, &n2), Rwork(&n2), Work(&lwork), T(nb, m), C(&n2, m), Cf(&n2, m), D(m, &n2), Df(m, &n2))
	a := mf(*m, n2, opts)
	af := mf(*m, n2, opts)
	q := mf(n2, n2, opts)
	r := mf(n2, n2, opts)
	rwork := vf(n2)
	work := vf(lwork)
	t := mf(*nb, *m, opts)
	c := mf(n2, *m, opts)
	cf := mf(n2, *m, opts)
	d := mf(*m, n2, opts)
	df := mf(*m, n2, opts)

	//     Put random stuff into A
	ldt = (*nb)
	golapack.Dlaset('F', m, &n2, &zero, &zero, a, m)
	golapack.Dlaset('F', nb, m, &zero, &zero, t, nb)
	for j = 1; j <= (*m); j++ {
		golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr((*m)-j+1), a.Vector(j-1, j-1))
	}
	if (*n) > 0 {
		for j = 1; j <= (*n)-(*l); j++ {
			golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, m, a.Vector(1-1, min((*n)+(*m), (*m)+1)+j-1-1))
		}
	}
	if (*l) > 0 {
		for j = 1; j <= (*l); j++ {
			golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr((*m)-j+1), a.Vector(j-1, min((*n)+(*m), (*n)+(*m)-(*l)+1)+j-1-1))
		}
	}

	//     Copy the matrix A to the array AF.
	golapack.Dlacpy('F', m, &n2, a, m, af, m)

	//     Factor the matrix A in the array AF.
	Dtplqt(m, n, l, nb, af, m, af.Off(1-1, np1-1), m, t, &ldt, work, &info)

	//     Generate the (M+N)-by-(M+N) matrix Q by applying H to I
	golapack.Dlaset('F', &n2, &n2, &zero, &one, q, &n2)
	golapack.Dgemlqt('L', 'N', &n2, &n2, &k, nb, af, m, t, &ldt, q, &n2, work, &info)

	//     Copy L
	golapack.Dlaset('F', &n2, &n2, &zero, &zero, r, &n2)
	golapack.Dlacpy('L', m, &n2, af, m, r, &n2)

	//     Compute |L - A*Q*T| / |A| and store in RESULT(1)
	err = goblas.Dgemm(NoTrans, Trans, *m, n2, n2, -one, a, q, one, r)
	anorm = golapack.Dlange('1', m, &n2, a, m, rwork)
	resid = golapack.Dlange('1', m, &n2, r, &n2, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*anorm*float64(max(1, n2))))
	} else {
		result.Set(0, zero)
	}

	//     Compute |I - Q*Q'| and store in RESULT(2)
	golapack.Dlaset('F', &n2, &n2, &zero, &one, r, &n2)
	err = goblas.Dsyrk(Upper, NoTrans, n2, n2, -one, q, one, r)
	resid = golapack.Dlansy('1', 'U', &n2, r, &n2, rwork)
	result.Set(1, resid/(eps*float64(max(1, n2))))

	//     Generate random m-by-n matrix C and a copy CF
	golapack.Dlaset('F', &n2, m, &zero, &one, c, &n2)
	for j = 1; j <= (*m); j++ {
		golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, &n2, c.Vector(1-1, j-1))
	}
	cnorm = golapack.Dlange('1', &n2, m, c, &n2, rwork)
	golapack.Dlacpy('F', &n2, m, c, &n2, cf, &n2)

	//     Apply Q to C as Q*C
	golapack.Dtpmlqt('L', 'N', n, m, &k, l, nb, af.Off(1-1, np1-1), m, t, &ldt, cf, &n2, cf.Off(np1-1, 1-1), &n2, work, &info)

	//     Compute |Q*C - Q*C| / |C|
	err = goblas.Dgemm(NoTrans, NoTrans, n2, *m, n2, -one, q, c, one, cf)
	resid = golapack.Dlange('1', &n2, m, cf, &n2, rwork)
	if cnorm > zero {
		result.Set(2, resid/(eps*float64(max(1, n2))*cnorm))
	} else {
		result.Set(2, zero)
	}

	//     Copy C into CF again
	golapack.Dlacpy('F', &n2, m, c, &n2, cf, &n2)

	//     Apply Q to C as QT*C
	golapack.Dtpmlqt('L', 'T', n, m, &k, l, nb, af.Off(1-1, np1-1), m, t, &ldt, cf, &n2, cf.Off(np1-1, 1-1), &n2, work, &info)

	//     Compute |QT*C - QT*C| / |C|
	err = goblas.Dgemm(Trans, NoTrans, n2, *m, n2, -one, q, c, one, cf)
	resid = golapack.Dlange('1', &n2, m, cf, &n2, rwork)
	if cnorm > zero {
		result.Set(3, resid/(eps*float64(max(1, n2))*cnorm))
	} else {
		result.Set(3, zero)
	}

	//     Generate random m-by-n matrix D and a copy DF
	for j = 1; j <= n2; j++ {
		golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, m, d.Vector(1-1, j-1))
	}
	dnorm = golapack.Dlange('1', m, &n2, d, m, rwork)
	golapack.Dlacpy('F', m, &n2, d, m, df, m)

	//     Apply Q to D as D*Q
	golapack.Dtpmlqt('R', 'N', m, n, &k, l, nb, af.Off(1-1, np1-1), m, t, &ldt, df, m, df.Off(1-1, np1-1), m, work, &info)

	//     Compute |D*Q - D*Q| / |D|
	err = goblas.Dgemm(NoTrans, NoTrans, *m, n2, n2, -one, d, q, one, df)
	resid = golapack.Dlange('1', m, &n2, df, m, rwork)
	if cnorm > zero {
		result.Set(4, resid/(eps*float64(max(1, n2))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy D into DF again
	golapack.Dlacpy('F', m, &n2, d, m, df, m)

	//     Apply Q to D as D*QT
	golapack.Dtpmlqt('R', 'T', m, n, &k, l, nb, af.Off(1-1, np1-1), m, t, &ldt, df, m, df.Off(1-1, np1-1), m, work, &info)

	//     Compute |D*QT - D*QT| / |D|
	err = goblas.Dgemm(NoTrans, Trans, *m, n2, n2, -one, d, q, one, df)
	resid = golapack.Dlange('1', m, &n2, df, m, rwork)
	if cnorm > zero {
		result.Set(5, resid/(eps*float64(max(1, n2))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
