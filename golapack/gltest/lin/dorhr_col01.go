package lin

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dorhrcol01 tests DorhrCol using Dlatsqr, Dgemqrt and Dqrgtsqr.
// Therefore, Dlatsqr (part of DGEQR), Dgemqrt (part DGEMQR), Dqrgtsqr
// have to be tested before this test.
func dorhrCol01(m, n, mb1, nb1, nb2 int, result *mat.Vector) {
	var testzeros bool
	var anorm, cnorm, dnorm, eps, one, resid, zero float64
	var i, j, k, l, lwork, nb1Ub, nb2Ub, nrb int
	var err error

	srnamt := &gltest.Common.Srnamc.Srnamt
	iseed := make([]int, 4)

	workquery := vf(1)

	zero = 0.0
	one = 1.0

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	//     TEST MATRICES WITH HALF OF MATRIX BEING ZEROS
	testzeros = false

	eps = golapack.Dlamch(Epsilon)
	k = min(m, n)
	l = max(m, n, 1)

	//     Dynamically allocate local arrays
	// Allocate(A(m, n), Af(m, n), Q(&l, &l), R(m, &l), Rwork(&l), C(m, n), Cf(m, n), D(n, m), Df(n, m))
	a := mf(m, n, opts)
	af := mf(m, n, opts)
	q := mf(l, l, opts)
	r := mf(m, l, opts)
	rwork := vf(l)
	c := mf(m, n, opts)
	cf := mf(m, n, opts)
	d := mf(n, m, opts)
	df := mf(n, m, opts)

	//     Put random numbers into A and copy to AF
	for j = 1; j <= n; j++ {
		golapack.Dlarnv(2, &iseed, m, a.Off(1-1, j-1).Vector())
	}
	if testzeros {
		if m >= 4 {
			for j = 1; j <= n; j++ {
				golapack.Dlarnv(2, &iseed, m/2, a.Off(m/4-1, j-1).Vector())
			}
		}
	}
	golapack.Dlacpy(Full, m, n, a, af)

	//     Number of row blocks in Dlatsqr
	nrb = max(1, int(math.Ceil(float64(m-n)/float64(mb1-n))))

	// Allocate(T1(nb1, n*nrb))
	// Allocate(T2(nb2, n))
	// Allocate(Diag(n))
	t1 := mf(nb1, n*nrb, opts)
	t2 := mf(nb2, n, opts)
	diag := vf(n)

	//     Begin determine LWORK for the array WORK and allocate memory.
	//
	//     Dlatsqr requires NB1 to be bounded by N.
	nb1Ub = min(nb1, n)

	//     Dgemqrt requires NB2 to be bounded by N.
	nb2Ub = min(nb2, n)

	if err = golapack.Dlatsqr(m, n, mb1, nb1Ub, af, t1, workquery, -1); err != nil {
		panic(err)
	}
	lwork = int(workquery.Get(0))
	if err = golapack.Dorgtsqr(m, n, mb1, nb1, af, t1, workquery, -1); err != nil {
		panic(err)
	}
	lwork = max(lwork, int(workquery.Get(0)))

	//     In Dgemqrt, WORK is N*NB2_UB if SIDE = 'L',
	//                or  M*NB2_UB if SIDE = 'R'.
	lwork = max(lwork, nb2Ub*n, nb2Ub*m)

	// Allocate(Work(&lwork))
	work := vf(lwork)

	//     End allocate memory for WORK.
	//
	//
	//     Begin Householder reconstruction routines
	//
	//     Factor the matrix A in the array AF.
	*srnamt = "Dlatsqr"
	if err = golapack.Dlatsqr(m, n, mb1, nb1Ub, af, t1, work, lwork); err != nil {
		panic(err)
	}

	//     Copy the factor R into the array R.
	*srnamt = "Dlacpy"
	golapack.Dlacpy(Upper, n, n, af, r)

	//     Reconstruct the orthogonal matrix Q.
	*srnamt = "Dqrgtsqr"
	if err = golapack.Dorgtsqr(m, n, mb1, nb1, af, t1, work, lwork); err != nil {
		panic(err)
	}

	//     Perform the Householder reconstruction, the result is stored
	//     the arrays AF and T2.
	*srnamt = "DorhrCol"
	if err = golapack.DorhrCol(m, n, nb2, af, t2, diag); err != nil {
		panic(err)
	}

	//     Compute the factor R_hr corresponding to the Householder
	//     reconstructed Q_hr and place it in the upper triangle of AF to
	//     match the Q storage format in DGEQRT. R_hr = R_tsqr * S,
	//     this means changing the sign of I-th row of the matrix R_tsqr
	//     according to sign of of I-th diagonal element DIAG(I) of the
	//     matrix S.
	*srnamt = "Dlacpy"
	golapack.Dlacpy(Upper, n, n, r, af)

	for i = 1; i <= n; i++ {
		if diag.Get(i-1) == -one {
			af.Off(i-1, i-1).Vector().Scal(n+1-i, -one, m)
		}
	}

	//     End Householder reconstruction routines.
	//
	//
	//     Generate the m-by-m matrix Q
	golapack.Dlaset(Full, m, m, zero, one, q)

	*srnamt = "Dgemqrt"
	if err = golapack.Dgemqrt(Left, NoTrans, m, m, k, nb2Ub, af, t2, q, work); err != nil {
		panic(err)
	}

	//     Copy R
	golapack.Dlaset(Full, m, n, zero, zero, r)

	golapack.Dlacpy(Upper, m, n, af, r)

	//     TEST 1
	//     Compute |R - (Q**T)*A| / ( eps * m * |A| ) and store in RESULT(1)
	if err = r.Gemm(Trans, NoTrans, m, n, m, -one, q, a, one); err != nil {
		panic(err)
	}

	anorm = golapack.Dlange('1', m, n, a, rwork)
	resid = golapack.Dlange('1', m, n, r, rwork)
	if anorm > zero {
		result.Set(0, resid/(eps*float64(max(1, m))*anorm))
	} else {
		result.Set(0, zero)
	}

	//     TEST 2
	//     Compute |I - (Q**T)*Q| / ( eps * m ) and store in RESULT(2)
	golapack.Dlaset(Full, m, m, zero, one, r)
	if err = r.Syrk(Upper, Trans, m, m, -one, q, one); err != nil {
		panic(err)
	}
	resid = golapack.Dlansy('1', Upper, m, r, rwork)
	result.Set(1, resid/(eps*float64(max(1, m))))

	//     Generate random m-by-n matrix C
	for j = 1; j <= n; j++ {
		golapack.Dlarnv(2, &iseed, m, c.Off(1-1, j-1).Vector())
	}
	cnorm = golapack.Dlange('1', m, n, c, rwork)
	golapack.Dlacpy(Full, m, n, c, cf)

	//     Apply Q to C as Q*C = CF
	*srnamt = "Dgemqrt"
	if err = golapack.Dgemqrt(Left, NoTrans, m, n, k, nb2Ub, af, t2, cf, work); err != nil {
		panic(err)
	}

	//     TEST 3
	//     Compute |CF - Q*C| / ( eps *  m * |C| )
	if err = cf.Gemm(NoTrans, NoTrans, m, n, m, -one, q, c, one); err != nil {
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

	//     Apply Q to C as (Q**T)*C = CF
	*srnamt = "Dgemqrt"
	if err = golapack.Dgemqrt(Left, Trans, m, n, k, nb2Ub, af, t2, cf, work); err != nil {
		panic(err)
	}

	//     TEST 4
	//     Compute |CF - (Q**T)*C| / ( eps * m * |C|)
	if err = cf.Gemm(Trans, NoTrans, m, n, m, -one, q, c, one); err != nil {
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
		golapack.Dlarnv(2, &iseed, n, d.Off(1-1, j-1).Vector())
	}
	dnorm = golapack.Dlange('1', n, m, d, rwork)
	golapack.Dlacpy(Full, n, m, d, df)

	//     Apply Q to D as D*Q = DF
	*srnamt = "Dgemqrt"
	if err = golapack.Dgemqrt(Right, NoTrans, n, m, k, nb2Ub, af, t2, df, work); err != nil {
		panic(err)
	}

	//     TEST 5
	//     Compute |DF - D*Q| / ( eps * m * |D| )
	if err = df.Gemm(NoTrans, NoTrans, n, m, m, -one, d, q, one); err != nil {
		panic(err)
	}
	resid = golapack.Dlange('1', n, m, df, rwork)
	if dnorm > zero {
		result.Set(4, resid/(eps*float64(max(1, m))*dnorm))
	} else {
		result.Set(4, zero)
	}

	//     Copy D into DF again
	golapack.Dlacpy(Full, n, m, d, df)

	//     Apply Q to D as D*QT = DF
	*srnamt = "Dgemqrt"
	if err = golapack.Dgemqrt(Right, Trans, n, m, k, nb2Ub, af, t2, df, work); err != nil {
		panic(err)
	}

	//     TEST 6
	//     Compute |DF - D*(Q**T)| / ( eps * m * |D| )
	if err = df.Gemm(NoTrans, Trans, n, m, m, -one, d, q, one); err != nil {
		panic(err)
	}
	resid = golapack.Dlange('1', n, m, df, rwork)
	if dnorm > zero {
		result.Set(5, resid/(eps*float64(max(1, m))*dnorm))
	} else {
		result.Set(5, zero)
	}
}
