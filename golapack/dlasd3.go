package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
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
// DLASD3 is called from DLASD1.
func Dlasd3(nl, nr, sqre, k *int, d *mat.Vector, q *mat.Matrix, ldq *int, dsigma *mat.Vector, u *mat.Matrix, ldu *int, u2 *mat.Matrix, ldu2 *int, vt *mat.Matrix, ldvt *int, vt2 *mat.Matrix, ldvt2 *int, idxc, ctot *[]int, z *mat.Vector, info *int) {
	var negone, one, rho, temp, zero float64
	var ctemp, i, j, jc, ktemp, m, n, nlp1, nlp2, nrp1 int

	one = 1.0
	zero = 0.0
	negone = -1.0

	//     Test the input parameters.
	(*info) = 0

	if (*nl) < 1 {
		(*info) = -1
	} else if (*nr) < 1 {
		(*info) = -2
	} else if ((*sqre) != 1) && ((*sqre) != 0) {
		(*info) = -3
	}

	n = (*nl) + (*nr) + 1
	m = n + (*sqre)
	nlp1 = (*nl) + 1
	nlp2 = (*nl) + 2

	if ((*k) < 1) || ((*k) > n) {
		(*info) = -4
	} else if (*ldq) < (*k) {
		(*info) = -7
	} else if (*ldu) < n {
		(*info) = -10
	} else if (*ldu2) < n {
		(*info) = -12
	} else if (*ldvt) < m {
		(*info) = -14
	} else if (*ldvt2) < m {
		(*info) = -16
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLASD3"), -(*info))
		return
	}

	//     Quick return if possible
	if (*k) == 1 {
		d.Set(0, math.Abs(z.Get(0)))
		goblas.Dcopy(&m, vt2.Vector(0, 0), ldvt2, vt.Vector(0, 0), ldvt)
		if z.Get(0) > zero {
			goblas.Dcopy(&n, u2.Vector(0, 0), toPtr(1), u.Vector(0, 0), toPtr(1))
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
	for i = 1; i <= (*k); i++ {
		dsigma.Set(i-1, Dlamc3(dsigma.GetPtr(i-1), dsigma.GetPtr(i-1))-dsigma.Get(i-1))
	}

	//     Keep a copy of Z.
	goblas.Dcopy(k, z, toPtr(1), q.VectorIdx(0), toPtr(1))

	//     Normalize Z.
	rho = goblas.Dnrm2(k, z, toPtr(1))
	Dlascl('G', toPtr(0), toPtr(0), &rho, &one, k, toPtr(1), z.Matrix(*k, opts), k, info)
	rho = rho * rho

	//     Find the new singular values.
	for j = 1; j <= (*k); j++ {
		Dlasd4(k, &j, dsigma, z, u.Vector(0, j-1), &rho, d.GetPtr(j-1), vt.Vector(0, j-1), info)

		//        If the zero finder fails, report the convergence failure.
		if (*info) != 0 {
			return
		}
	}

	//     Compute updated Z.
	for i = 1; i <= (*k); i++ {
		z.Set(i-1, u.Get(i-1, (*k)-1)*vt.Get(i-1, (*k)-1))
		for j = 1; j <= i-1; j++ {
			z.Set(i-1, z.Get(i-1)*(u.Get(i-1, j-1)*vt.Get(i-1, j-1)/(dsigma.Get(i-1)-dsigma.Get(j-1))/(dsigma.Get(i-1)+dsigma.Get(j-1))))
		}
		for j = i; j <= (*k)-1; j++ {
			z.Set(i-1, z.Get(i-1)*(u.Get(i-1, j-1)*vt.Get(i-1, j-1)/(dsigma.Get(i-1)-dsigma.Get(j+1-1))/(dsigma.Get(i-1)+dsigma.Get(j+1-1))))
		}
		z.Set(i-1, signf64(math.Sqrt(math.Abs(z.Get(i-1))), q.Get(i-1, 0)))
	}

	//     Compute left singular vectors of the modified diagonal matrix,
	//     and store related information for the right singular vectors.
	for i = 1; i <= (*k); i++ {
		vt.Set(0, i-1, z.Get(0)/u.Get(0, i-1)/vt.Get(0, i-1))
		u.Set(0, i-1, negone)
		for j = 2; j <= (*k); j++ {
			vt.Set(j-1, i-1, z.Get(j-1)/u.Get(j-1, i-1)/vt.Get(j-1, i-1))
			u.Set(j-1, i-1, dsigma.Get(j-1)*vt.Get(j-1, i-1))
		}
		temp = goblas.Dnrm2(k, u.Vector(0, i-1), toPtr(1))
		q.Set(0, i-1, u.Get(0, i-1)/temp)
		for j = 2; j <= (*k); j++ {
			jc = (*idxc)[j-1]
			q.Set(j-1, i-1, u.Get(jc-1, i-1)/temp)
		}
	}

	//     Update the left singular vector matrix.
	if (*k) == 2 {
		goblas.Dgemm(NoTrans, NoTrans, &n, k, k, &one, u2, ldu2, q, ldq, &zero, u, ldu)
		goto label100
	}
	if (*ctot)[0] > 0 {
		goblas.Dgemm(NoTrans, NoTrans, nl, k, &((*ctot)[0]), &one, u2.Off(0, 1), ldu2, q.Off(1, 0), ldq, &zero, u.Off(0, 0), ldu)
		if (*ctot)[2] > 0 {
			ktemp = 2 + (*ctot)[0] + (*ctot)[1]
			goblas.Dgemm(NoTrans, NoTrans, nl, k, &((*ctot)[2]), &one, u2.Off(0, ktemp-1), ldu2, q.Off(ktemp-1, 0), ldq, &one, u.Off(0, 0), ldu)
		}
	} else if (*ctot)[2] > 0 {
		ktemp = 2 + (*ctot)[0] + (*ctot)[1]
		goblas.Dgemm(NoTrans, NoTrans, nl, k, &((*ctot)[2]), &one, u2.Off(0, ktemp-1), ldu2, q.Off(ktemp-1, 0), ldq, &zero, u.Off(0, 0), ldu)
	} else {
		Dlacpy('F', nl, k, u2, ldu2, u, ldu)
	}
	goblas.Dcopy(k, q.Vector(0, 0), ldq, u.Vector(nlp1-1, 0), ldu)
	ktemp = 2 + (*ctot)[0]
	ctemp = (*ctot)[1] + (*ctot)[2]
	goblas.Dgemm(NoTrans, NoTrans, nr, k, &ctemp, &one, u2.Off(nlp2-1, ktemp-1), ldu2, q.Off(ktemp-1, 0), ldq, &zero, u.Off(nlp2-1, 0), ldu)

	//     Generate the right singular vectors.
label100:
	;
	for i = 1; i <= (*k); i++ {
		temp = goblas.Dnrm2(k, vt.Vector(0, i-1), toPtr(1))
		q.Set(i-1, 0, vt.Get(0, i-1)/temp)
		for j = 2; j <= (*k); j++ {
			jc = (*idxc)[j-1]
			q.Set(i-1, j-1, vt.Get(jc-1, i-1)/temp)
		}
	}

	//     Update the right singular vector matrix.
	if (*k) == 2 {
		goblas.Dgemm(NoTrans, NoTrans, k, &m, k, &one, q, ldq, vt2, ldvt2, &zero, vt, ldvt)
		return
	}
	ktemp = 1 + (*ctot)[0]
	goblas.Dgemm(NoTrans, NoTrans, k, &nlp1, &ktemp, &one, q.Off(0, 0), ldq, vt2.Off(0, 0), ldvt2, &zero, vt.Off(0, 0), ldvt)
	ktemp = 2 + (*ctot)[0] + (*ctot)[1]
	if ktemp <= (*ldvt2) {
		goblas.Dgemm(NoTrans, NoTrans, k, &nlp1, &((*ctot)[2]), &one, q.Off(0, ktemp-1), ldq, vt2.Off(ktemp-1, 0), ldvt2, &one, vt.Off(0, 0), ldvt)
	}

	ktemp = (*ctot)[0] + 1
	nrp1 = (*nr) + (*sqre)
	if ktemp > 1 {
		for i = 1; i <= (*k); i++ {
			q.Set(i-1, ktemp-1, q.Get(i-1, 0))
		}
		for i = nlp2; i <= m; i++ {
			vt2.Set(ktemp-1, i-1, vt2.Get(0, i-1))
		}
	}
	ctemp = 1 + (*ctot)[1] + (*ctot)[2]
	goblas.Dgemm(NoTrans, NoTrans, k, &nrp1, &ctemp, &one, q.Off(0, ktemp-1), ldq, vt2.Off(ktemp-1, nlp2-1), ldvt2, &zero, vt.Off(0, nlp2-1), ldvt)
}
