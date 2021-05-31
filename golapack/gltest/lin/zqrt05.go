package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zqrt05 tests ZTPQRT and ZTPMQRT.
func Zqrt05(m, n, l, nb *int, result *mat.Vector) {
	var czero, one complex128
	var anorm, cnorm, dnorm, eps, resid, zero float64
	var info, j, k, ldt, lwork, m2, np1 int
	iseed := make([]int, 4)

	zero = 0.0
	one = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

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
	a := cmf(m2, *n, opts)
	af := cmf(m2, *n, opts)
	q := cmf(m2, m2, opts)
	r := cmf(m2, m2, opts)
	rwork := vf(m2)
	work := cvf(lwork)
	t := cmf(*nb, *n, opts)
	c := cmf(m2, *n, opts)
	cf := cmf(m2, *n, opts)
	d := cmf(*n, m2, opts)
	df := cmf(*n, m2, opts)

	//     Put random stuff into A
	ldt = (*nb)
	golapack.Zlaset('F', &m2, n, &czero, &czero, a, &m2)
	golapack.Zlaset('F', nb, n, &czero, &czero, t, nb)
	for j = 1; j <= (*n); j++ {
		golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, &j, a.CVector(0, j-1))
	}
	if (*m) > 0 {
		for j = 1; j <= (*n); j++ {
			golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr((*m)-(*l)), a.CVector(minint((*n)+(*m), (*n)+1)-1, j-1))
		}
	}
	if (*l) > 0 {
		for j = 1; j <= (*n); j++ {
			golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr(minint(j, *l)), a.CVector(minint((*n)+(*m), (*n)+(*m)-(*l)+1)-1, j-1))
		}
	}

	//     Copy the matrix A to the array af.
	golapack.Zlacpy('F', &m2, n, a, &m2, af, &m2)

	//     Factor the matrix A in the array af.
	golapack.Ztpqrt(m, n, l, nb, af, &m2, af.Off(np1-1, 0), &m2, t, &ldt, work, &info)

	//     Generate the (M+N)-by-(M+N) matrix Q by applying H to I
	golapack.Zlaset('F', &m2, &m2, &czero, &one, q, &m2)
	golapack.Zgemqrt('R', 'N', &m2, &m2, &k, nb, af, &m2, t, &ldt, q, &m2, work, &info)

	//     Copy R
	golapack.Zlaset('F', &m2, n, &czero, &czero, r, &m2)
	golapack.Zlacpy('U', &m2, n, af, &m2, r, &m2)

	//     Compute |R - Q'*A| / |A| and store in RESULT(1)
	goblas.Zgemm(ConjTrans, NoTrans, &m2, n, &m2, toPtrc128(-one), q, &m2, a, &m2, &one, r, &m2)
	anorm = golapack.Zlange('1', &m2, n, a, &m2, rwork)
	resid = golapack.Zlange('1', &m2, n, r, &m2, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*anorm*float64(maxint(1, m2))))
	} else {
		result.Set(0, zero)
	}

	//     Compute |I - Q'*Q| and store in RESULT(2)
	golapack.Zlaset('F', &m2, &m2, &czero, &one, r, &m2)
	goblas.Zherk(Upper, ConjTrans, &m2, &m2, toPtrf64(real(-one)), q, &m2, toPtrf64(real(one)), r, &m2)
	resid = golapack.Zlansy('1', 'U', &m2, r, &m2, rwork)
	result.Set(1, resid/(eps*float64(maxint(1, m2))))

	//     Generate random m-by-n matrix C and a copy CF
	for j = 1; j <= (*n); j++ {
		golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, &m2, c.CVector(0, j-1))
	}
	cnorm = golapack.Zlange('1', &m2, n, c, &m2, rwork)
	golapack.Zlacpy('F', &m2, n, c, &m2, cf, &m2)

	//     Apply Q to C as Q*C
	golapack.Ztpmqrt('L', 'N', m, n, &k, l, nb, af.Off(np1-1, 0), &m2, t, &ldt, cf, &m2, cf.Off(np1-1, 0), &m2, work, &info)

	//     Compute |Q*C - Q*C| / |C|
	goblas.Zgemm(NoTrans, NoTrans, &m2, n, &m2, toPtrc128(-one), q, &m2, c, &m2, &one, cf, &m2)
	resid = golapack.Zlange('1', &m2, n, cf, &m2, rwork)
	if cnorm > zero {
		result.Set(2, resid/(eps*float64(maxint(1, m2))*cnorm))
	} else {
		result.Set(2, zero)
	}

	//     Copy C into CF again
	golapack.Zlacpy('F', &m2, n, c, &m2, cf, &m2)

	//     Apply Q to C as QT*C
	golapack.Ztpmqrt('L', 'C', m, n, &k, l, nb, af.Off(np1-1, 0), &m2, t, &ldt, cf, &m2, cf.Off(np1-1, 0), &m2, work, &info)

	//     Compute |QT*C - QT*C| / |C|
	goblas.Zgemm(ConjTrans, NoTrans, &m2, n, &m2, toPtrc128(-one), q, &m2, c, &m2, &one, cf, &m2)
	resid = golapack.Zlange('1', &m2, n, cf, &m2, rwork)
	if cnorm > zero {
		result.Set(3, resid/(eps*float64(maxint(1, m2))*cnorm))
	} else {
		result.Set(3, zero)
	}

	//     Generate random n-by-m matrix D and a copy DF
	for j = 1; j <= m2; j++ {
		golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, n, d.CVector(0, j-1))
	}
	dnorm = golapack.Zlange('1', n, &m2, d, n, rwork)
	golapack.Zlacpy('F', n, &m2, d, n, df, n)

	//     Apply Q to D as D*Q
	golapack.Ztpmqrt('R', 'N', n, m, n, l, nb, af.Off(np1-1, 0), &m2, t, &ldt, df, n, df.Off(0, np1-1), n, work, &info)

	//     Compute |D*Q - D*Q| / |D|
	goblas.Zgemm(NoTrans, NoTrans, n, &m2, &m2, toPtrc128(-one), d, n, q, &m2, &one, df, n)
	resid = golapack.Zlange('1', n, &m2, df, n, rwork)
	if cnorm > zero {
		result.Set(4, resid/(eps*float64(maxint(1, m2))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy D into DF again
	golapack.Zlacpy('F', n, &m2, d, n, df, n)

	//     Apply Q to D as D*QT
	golapack.Ztpmqrt('R', 'C', n, m, n, l, nb, af.Off(np1-1, 0), &m2, t, &ldt, df, n, df.Off(0, np1-1), n, work, &info)

	//     Compute |D*QT - D*QT| / |D|
	goblas.Zgemm(NoTrans, ConjTrans, n, &m2, &m2, toPtrc128(-one), d, n, q, &m2, &one, df, n)
	resid = golapack.Zlange('1', n, &m2, df, n, rwork)
	if cnorm > zero {
		result.Set(5, resid/(eps*float64(maxint(1, m2))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
