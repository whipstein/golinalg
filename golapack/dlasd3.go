package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlasd3 finds all the square roots of the roots of the secular
// equation, as defined by the values in D and Z.  It makes the
// appropriate calls to DLASD4 and then updates the singular
// vectors by matrix multiplication.
//
// This code makes very mild assumptions about floating point
// arithmetic. It will work on machines with a guard digit in
// add/subtract, or on those binary machines without guard digits
// which subtract like the Cray XMP, Cray YMP, Cray C 90, or Cray 2.
// It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
//
// Dlasd3 is called from DLASD1.
func Dlasd3(nl, nr, sqre, k int, d *mat.Vector, q *mat.Matrix, dsigma *mat.Vector, u, u2, vt, vt2 *mat.Matrix, idxc, ctot *[]int, z *mat.Vector) (info int, err error) {
	var negone, one, rho, temp, zero float64
	var ctemp, i, j, jc, ktemp, m, n, nlp1, nlp2, nrp1 int

	one = 1.0
	zero = 0.0
	negone = -1.0

	//     Test the input parameters.
	if nl < 1 {
		err = fmt.Errorf("nl < 1: nl=%v", nl)
	} else if nr < 1 {
		err = fmt.Errorf("nr < 1: nr=%v", nr)
	} else if (sqre != 1) && (sqre != 0) {
		err = fmt.Errorf("(sqre != 1) && (sqre != 0): sqre=%v", sqre)
	}

	n = nl + nr + 1
	m = n + sqre
	nlp1 = nl + 1
	nlp2 = nl + 2

	if (k < 1) || (k > n) {
		err = fmt.Errorf("(k < 1) || (k > n): k=%v, n=%v", k, n)
	} else if q.Rows < k {
		err = fmt.Errorf("q.Rows < k: q.Rows=%v, k=%v", q.Rows, k)
	} else if u.Rows < n {
		err = fmt.Errorf("u.Rows < n: u.Rows=%v, n=%v", u.Rows, n)
	} else if u2.Rows < n {
		err = fmt.Errorf("u2.Rows < n: u2.Rows=%v, n=%v", u2.Rows, n)
	} else if vt.Rows < m {
		err = fmt.Errorf("vt.Rows < m: vt.Rows=%v, m=%v", vt.Rows, m)
	} else if vt2.Rows < m {
		err = fmt.Errorf("vt2.Rows < m: vt2.Rows=%v, m=%v", vt2.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Dlasd3", err)
		return
	}

	//     Quick return if possible
	if k == 1 {
		d.Set(0, math.Abs(z.Get(0)))
		vt.Off(0, 0).Vector().Copy(m, vt2.Off(0, 0).Vector(), vt.Rows, vt.Rows)
		if z.Get(0) > zero {
			u.Off(0, 0).Vector().Copy(n, u2.Off(0, 0).Vector(), 1, 1)
		} else {
			for i = 1; i <= n; i++ {
				u.Set(i-1, 0, -u2.Get(i-1, 0))
			}
		}
		return
	}

	//     Modify values DSIGMA(i) to make sure all DSIGMA(i)-DSIGMA(j) can
	//     be computed with high relative accuracy (barring over/underflow).
	//     This is a problem on machines without a guard digit in
	//     add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2).
	//     The following code replaces DSIGMA(I) by 2*DSIGMA(I)-DSIGMA(I),
	//     which on any of these machines zeros out the bottommost
	//     bit of DSIGMA(I) if it is 1; this makes the subsequent
	//     subtractions DSIGMA(I)-DSIGMA(J) unproblematic when cancellation
	//     occurs. On binary machines with a guard digit (almost all
	//     machines) it does not change DSIGMA(I) at all. On hexadecimal
	//     and decimal machines with a guard digit, it slightly
	//     changes the bottommost bits of DSIGMA(I). It does not account
	//     for hexadecimal or decimal machines without guard digits
	//     (we know of none). We use a subroutine call to compute
	//     2*DSIGMA(I) to prevent optimizing compilers from eliminating
	//     this code.
	for i = 1; i <= k; i++ {
		dsigma.Set(i-1, Dlamc3(dsigma.GetPtr(i-1), dsigma.GetPtr(i-1))-dsigma.Get(i-1))
	}

	//     Keep a copy of Z.
	q.OffIdx(0).Vector().Copy(k, z, 1, 1)

	//     Normalize Z.
	rho = z.Nrm2(k, 1)
	if err = Dlascl('G', 0, 0, rho, one, k, 1, z.Matrix(k, opts)); err != nil {
		panic(err)
	}
	rho = rho * rho

	//     Find the new singular values.
	for j = 1; j <= k; j++ {
		*d.GetPtr(j - 1), info = Dlasd4(k, j, dsigma, z, u.Off(0, j-1).Vector(), rho, vt.Off(0, j-1).Vector())

		//        If the zero finder fails, report the convergence failure.
		if info != 0 {
			return
		}
	}

	//     Compute updated Z.
	for i = 1; i <= k; i++ {
		z.Set(i-1, u.Get(i-1, k-1)*vt.Get(i-1, k-1))
		for j = 1; j <= i-1; j++ {
			z.Set(i-1, z.Get(i-1)*(u.Get(i-1, j-1)*vt.Get(i-1, j-1)/(dsigma.Get(i-1)-dsigma.Get(j-1))/(dsigma.Get(i-1)+dsigma.Get(j-1))))
		}
		for j = i; j <= k-1; j++ {
			z.Set(i-1, z.Get(i-1)*(u.Get(i-1, j-1)*vt.Get(i-1, j-1)/(dsigma.Get(i-1)-dsigma.Get(j))/(dsigma.Get(i-1)+dsigma.Get(j))))
		}
		z.Set(i-1, math.Copysign(math.Sqrt(math.Abs(z.Get(i-1))), q.Get(i-1, 0)))
	}

	//     Compute left singular vectors of the modified diagonal matrix,
	//     and store related information for the right singular vectors.
	for i = 1; i <= k; i++ {
		vt.Set(0, i-1, z.Get(0)/u.Get(0, i-1)/vt.Get(0, i-1))
		u.Set(0, i-1, negone)
		for j = 2; j <= k; j++ {
			vt.Set(j-1, i-1, z.Get(j-1)/u.Get(j-1, i-1)/vt.Get(j-1, i-1))
			u.Set(j-1, i-1, dsigma.Get(j-1)*vt.Get(j-1, i-1))
		}
		temp = u.Off(0, i-1).Vector().Nrm2(k, 1)
		q.Set(0, i-1, u.Get(0, i-1)/temp)
		for j = 2; j <= k; j++ {
			jc = (*idxc)[j-1]
			q.Set(j-1, i-1, u.Get(jc-1, i-1)/temp)
		}
	}

	//     Update the left singular vector matrix.
	if k == 2 {
		if err = u.Gemm(NoTrans, NoTrans, n, k, k, one, u2, q, zero); err != nil {
			panic(err)
		}
		goto label100
	}
	if (*ctot)[0] > 0 {
		if err = u.Gemm(NoTrans, NoTrans, nl, k, (*ctot)[0], one, u2.Off(0, 1), q.Off(1, 0), zero); err != nil {
			panic(err)
		}
		if (*ctot)[2] > 0 {
			ktemp = 2 + (*ctot)[0] + (*ctot)[1]
			if err = u.Gemm(NoTrans, NoTrans, nl, k, (*ctot)[2], one, u2.Off(0, ktemp-1), q.Off(ktemp-1, 0), one); err != nil {
				panic(err)
			}
		}
	} else if (*ctot)[2] > 0 {
		ktemp = 2 + (*ctot)[0] + (*ctot)[1]
		if err = u.Gemm(NoTrans, NoTrans, nl, k, (*ctot)[2], one, u2.Off(0, ktemp-1), q.Off(ktemp-1, 0), zero); err != nil {
			panic(err)
		}
	} else {
		Dlacpy(Full, nl, k, u2, u)
	}
	u.Off(nlp1-1, 0).Vector().Copy(k, q.Off(0, 0).Vector(), q.Rows, u.Rows)
	ktemp = 2 + (*ctot)[0]
	ctemp = (*ctot)[1] + (*ctot)[2]
	if err = u.Off(nlp2-1, 0).Gemm(NoTrans, NoTrans, nr, k, ctemp, one, u2.Off(nlp2-1, ktemp-1), q.Off(ktemp-1, 0), zero); err != nil {
		panic(err)
	}

	//     Generate the right singular vectors.
label100:
	;
	for i = 1; i <= k; i++ {
		temp = vt.Off(0, i-1).Vector().Nrm2(k, 1)
		q.Set(i-1, 0, vt.Get(0, i-1)/temp)
		for j = 2; j <= k; j++ {
			jc = (*idxc)[j-1]
			q.Set(i-1, j-1, vt.Get(jc-1, i-1)/temp)
		}
	}

	//     Update the right singular vector matrix.
	if k == 2 {
		if err = vt.Gemm(NoTrans, NoTrans, k, m, k, one, q, vt2, zero); err != nil {
			panic(err)
		}
		return
	}
	ktemp = 1 + (*ctot)[0]
	if err = vt.Gemm(NoTrans, NoTrans, k, nlp1, ktemp, one, q.Off(0, 0), vt2.Off(0, 0), zero); err != nil {
		panic(err)
	}
	ktemp = 2 + (*ctot)[0] + (*ctot)[1]
	if ktemp <= vt2.Rows {
		if err = vt.Gemm(NoTrans, NoTrans, k, nlp1, (*ctot)[2], one, q.Off(0, ktemp-1), vt2.Off(ktemp-1, 0), one); err != nil {
			panic(err)
		}
	}

	ktemp = (*ctot)[0] + 1
	nrp1 = nr + sqre
	if ktemp > 1 {
		for i = 1; i <= k; i++ {
			q.Set(i-1, ktemp-1, q.Get(i-1, 0))
		}
		for i = nlp2; i <= m; i++ {
			vt2.Set(ktemp-1, i-1, vt2.Get(0, i-1))
		}
	}
	ctemp = 1 + (*ctot)[1] + (*ctot)[2]
	if err = vt.Off(0, nlp2-1).Gemm(NoTrans, NoTrans, k, nrp1, ctemp, one, q.Off(0, ktemp-1), vt2.Off(ktemp-1, nlp2-1), zero); err != nil {
		panic(err)
	}

	return
}
