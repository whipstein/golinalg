package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zlqt05 tests ZTPLQT and ZTPMLQT.
func Zlqt05(m, n, l, nb *int, result *mat.Vector) {
	var czero, one complex128
	var anorm, cnorm, dnorm, eps, resid, zero float64
	var info, j, k, ldt, lwork, n2, np1 int
	var err error
	_ = err

	iseed := make([]int, 4)

	zero = 0.0
	one = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

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
	a := cmf(*m, n2, opts)
	af := cmf(*m, n2, opts)
	q := cmf(n2, n2, opts)
	r := cmf(n2, n2, opts)
	rwork := vf(n2)
	work := cvf(lwork)
	t := cmf(*nb, *m, opts)
	c := cmf(n2, *m, opts)
	cf := cmf(n2, *m, opts)
	d := cmf(*m, n2, opts)
	df := cmf(*m, n2, opts)

	//     Put random stuff into A
	ldt = (*nb)
	golapack.Zlaset('F', m, &n2, &czero, &czero, a, m)
	golapack.Zlaset('F', nb, m, &czero, &czero, t, nb)
	for j = 1; j <= (*m); j++ {
		golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr((*m)-j+1), a.CVector(j-1, j-1))
	}
	if (*n) > 0 {
		for j = 1; j <= (*n)-(*l); j++ {
			golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, m, a.CVector(0, min((*n)+(*m), (*m)+1)+j-1-1))
		}
	}
	if (*l) > 0 {
		for j = 1; j <= (*l); j++ {
			golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr((*m)-j+1), a.CVector(j-1, min((*n)+(*m), (*n)+(*m)-(*l)+1)+j-1-1))
		}
	}

	//     Copy the matrix A to the array af.
	golapack.Zlacpy('F', m, &n2, a, m, af, m)

	//     Factor the matrix A in the array af.
	golapack.Ztplqt(m, n, l, nb, af, m, af.Off(0, np1-1), m, t, &ldt, work, &info)

	//     Generate the (M+N)-by-(M+N) matrix Q by applying H to I
	golapack.Zlaset('F', &n2, &n2, &czero, &one, q, &n2)
	golapack.Zgemlqt('L', 'N', &n2, &n2, &k, nb, af, m, t, &ldt, q, &n2, work, &info)

	//     Copy L
	golapack.Zlaset('F', &n2, &n2, &czero, &czero, r, &n2)
	golapack.Zlacpy('L', m, &n2, af, m, r, &n2)

	//     Compute |L - A*Q*C| / |A| and store in RESULT(1)
	err = goblas.Zgemm(NoTrans, ConjTrans, *m, n2, n2, -one, a, q, one, r)
	anorm = golapack.Zlange('1', m, &n2, a, m, rwork)
	resid = golapack.Zlange('1', m, &n2, r, &n2, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*anorm*float64(max(1, n2))))
	} else {
		result.Set(0, zero)
	}

	//     Compute |I - Q*Q'| and store in RESULT(2)
	golapack.Zlaset('F', &n2, &n2, &czero, &one, r, &n2)
	err = goblas.Zherk(Upper, NoTrans, n2, n2, real(-one), q, real(one), r)
	resid = golapack.Zlansy('1', 'U', &n2, r, &n2, rwork)
	result.Set(1, resid/(eps*float64(max(1, n2))))

	//     Generate random m-by-n matrix C and a copy CF
	golapack.Zlaset('F', &n2, m, &czero, &one, c, &n2)
	for j = 1; j <= (*m); j++ {
		golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, &n2, c.CVector(0, j-1))
	}
	cnorm = golapack.Zlange('1', &n2, m, c, &n2, rwork)
	golapack.Zlacpy('F', &n2, m, c, &n2, cf, &n2)

	//     Apply Q to C as Q*C
	golapack.Ztpmlqt('L', 'N', n, m, &k, l, nb, af.Off(0, np1-1), m, t, &ldt, cf, &n2, cf.Off(np1-1, 0), &n2, work, &info)

	//     Compute |Q*C - Q*C| / |C|
	err = goblas.Zgemm(NoTrans, NoTrans, n2, *m, n2, -one, q, c, one, cf)
	resid = golapack.Zlange('1', &n2, m, cf, &n2, rwork)
	if cnorm > zero {
		result.Set(2, resid/(eps*float64(max(1, n2))*cnorm))
	} else {
		result.Set(2, zero)
	}

	//     Copy C into CF again
	golapack.Zlacpy('F', &n2, m, c, &n2, cf, &n2)

	//     Apply Q to C as QT*C
	golapack.Ztpmlqt('L', 'C', n, m, &k, l, nb, af.Off(0, np1-1), m, t, &ldt, cf, &n2, cf.Off(np1-1, 0), &n2, work, &info)

	//     Compute |QT*C - QT*C| / |C|
	err = goblas.Zgemm(ConjTrans, NoTrans, n2, *m, n2, -one, q, c, one, cf)
	resid = golapack.Zlange('1', &n2, m, cf, &n2, rwork)
	if cnorm > zero {
		result.Set(3, resid/(eps*float64(max(1, n2))*cnorm))
	} else {
		result.Set(3, zero)
	}

	//     Generate random m-by-n matrix D and a copy DF
	for j = 1; j <= n2; j++ {
		golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, m, d.CVector(0, j-1))
	}
	dnorm = golapack.Zlange('1', m, &n2, d, m, rwork)
	golapack.Zlacpy('F', m, &n2, d, m, df, m)

	//     Apply Q to D as D*Q
	golapack.Ztpmlqt('R', 'N', m, n, &k, l, nb, af.Off(0, np1-1), m, t, &ldt, df, m, df.Off(0, np1-1), m, work, &info)

	//     Compute |D*Q - D*Q| / |D|
	err = goblas.Zgemm(NoTrans, NoTrans, *m, n2, n2, -one, d, q, one, df)
	resid = golapack.Zlange('1', m, &n2, df, m, rwork)
	if cnorm > zero {
		result.Set(4, resid/(eps*float64(max(1, n2))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy D into DF again
	golapack.Zlacpy('F', m, &n2, d, m, df, m)

	//     Apply Q to D as D*QT
	golapack.Ztpmlqt('R', 'C', m, n, &k, l, nb, af.Off(0, np1-1), m, t, &ldt, df, m, df.Off(0, np1-1), m, work, &info)

	//     Compute |D*QT - D*QT| / |D|
	err = goblas.Zgemm(NoTrans, ConjTrans, *m, n2, n2, -one, d, q, one, df)
	resid = golapack.Zlange('1', m, &n2, df, m, rwork)
	if cnorm > zero {
		result.Set(5, resid/(eps*float64(max(1, n2))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
