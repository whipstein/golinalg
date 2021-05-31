package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlasd2 merges the two sets of singular values together into a single
// sorted set.  Then it tries to deflate the size of the problem.
// There are two ways in which deflation can occur:  when two or more
// singular values are close together or if there is a tiny entry in the
// Z vector.  For each such occurrence the order of the related secular
// equation problem is reduced by one.
//
// DLASD2 is called from DLASD1.
func Dlasd2(nl, nr, sqre, k *int, d, z *mat.Vector, alpha, beta *float64, u *mat.Matrix, ldu *int, vt *mat.Matrix, ldvt *int, dsigma *mat.Vector, u2 *mat.Matrix, ldu2 *int, vt2 *mat.Matrix, ldvt2 *int, idxp, idx, idxc, idxq, coltyp *[]int, info *int) {
	var c, eight, eps, hlftol, one, s, tau, tol, two, z1, zero float64
	var ct, i, idxi, idxj, idxjp, j, jp, jprev, k2, m, n, nlp1, nlp2 int
	ctot := make([]int, 4)
	psm := make([]int, 4)

	zero = 0.0
	one = 1.0
	two = 2.0
	eight = 8.0

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

	if (*ldu) < n {
		(*info) = -10
	} else if (*ldvt) < m {
		(*info) = -12
	} else if (*ldu2) < n {
		(*info) = -15
	} else if (*ldvt2) < m {
		(*info) = -17
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLASD2"), -(*info))
		return
	}

	nlp1 = (*nl) + 1
	nlp2 = (*nl) + 2

	//     Generate the first part of the vector Z; and move the singular
	//     values in the first part of D one position backward.
	z1 = (*alpha) * vt.Get(nlp1-1, nlp1-1)
	z.Set(0, z1)
	for i = (*nl); i >= 1; i-- {
		z.Set(i+1-1, (*alpha)*vt.Get(i-1, nlp1-1))
		d.Set(i+1-1, d.Get(i-1))
		(*idxq)[i+1-1] = (*idxq)[i-1] + 1
	}

	//     Generate the second part of the vector Z.
	for i = nlp2; i <= m; i++ {
		z.Set(i-1, (*beta)*vt.Get(i-1, nlp2-1))
	}

	//     Initialize some reference arrays.
	for i = 2; i <= nlp1; i++ {
		(*coltyp)[i-1] = 1
	}
	for i = nlp2; i <= n; i++ {
		(*coltyp)[i-1] = 2
	}

	//     Sort the singular values into increasing order
	for i = nlp2; i <= n; i++ {
		(*idxq)[i-1] = (*idxq)[i-1] + nlp1
	}

	//     DSIGMA, IDXC, IDXC, and the first column of U2
	//     are used as storage space.
	for i = 2; i <= n; i++ {
		dsigma.Set(i-1, d.Get((*idxq)[i-1]-1))
		u2.Set(i-1, 0, z.Get((*idxq)[i-1]-1))
		(*idxc)[i-1] = (*coltyp)[(*idxq)[i-1]-1]
	}

	Dlamrg(nl, nr, dsigma.Off(1), toPtr(1), toPtr(1), toSlice(idx, 1))

	for i = 2; i <= n; i++ {
		idxi = 1 + (*idx)[i-1]
		d.Set(i-1, dsigma.Get(idxi-1))
		z.Set(i-1, u2.Get(idxi-1, 0))
		(*coltyp)[i-1] = (*idxc)[idxi-1]
	}

	//     Calculate the allowable deflation tolerance
	eps = Dlamch(Epsilon)
	tol = maxf64(math.Abs(*alpha), math.Abs(*beta))
	tol = eight * eps * maxf64(math.Abs(d.Get(n-1)), tol)

	//     There are 2 kinds of deflation -- first a value in the z-vector
	//     is small, second two (or more) singular values are very close
	//     together (their difference is small).
	//
	//     If the value in the z-vector is small, we simply permute the
	//     array so that the corresponding singular value is moved to the
	//     end.
	//
	//     If two values in the D-vector are close, we perform a two-sided
	//     rotation designed to make one of the corresponding z-vector
	//     entries zero, and then permute the array so that the deflated
	//     singular value is moved to the end.
	//
	//     If there are multiple singular values then the problem deflates.
	//     Here the number of equal singular values are found.  As each equal
	//     singular value is found, an elementary reflector is computed to
	//     rotate the corresponding singular subspace so that the
	//     corresponding components of Z are zero in this new basis.
	(*k) = 1
	k2 = n + 1
	for j = 2; j <= n; j++ {
		if math.Abs(z.Get(j-1)) <= tol {
			//           Deflate due to small z component.
			k2 = k2 - 1
			(*idxp)[k2-1] = j
			(*coltyp)[j-1] = 4
			if j == n {
				goto label120
			}
		} else {
			jprev = j
			goto label90
		}
	}
label90:
	;
	j = jprev
label100:
	;
	j = j + 1
	if j > n {
		goto label110
	}
	if math.Abs(z.Get(j-1)) <= tol {
		//        Deflate due to small z component.
		k2 = k2 - 1
		(*idxp)[k2-1] = j
		(*coltyp)[j-1] = 4
	} else {
		//        Check if singular values are close enough to allow deflation.
		if math.Abs(d.Get(j-1)-d.Get(jprev-1)) <= tol {
			//           Deflation is possible.
			s = z.Get(jprev - 1)
			c = z.Get(j - 1)

			//           Find sqrt(a**2+b**2) without overflow or
			//           destructive underflow.
			tau = Dlapy2(&c, &s)
			c = c / tau
			s = -s / tau
			z.Set(j-1, tau)
			z.Set(jprev-1, zero)

			//           Apply back the Givens rotation to the left and right
			//           singular vector matrices.
			idxjp = (*idxq)[(*idx)[jprev-1]+1-1]
			idxj = (*idxq)[(*idx)[j-1]+1-1]
			if idxjp <= nlp1 {
				idxjp = idxjp - 1
			}
			if idxj <= nlp1 {
				idxj = idxj - 1
			}
			goblas.Drot(&n, u.Vector(0, idxjp-1), toPtr(1), u.Vector(0, idxj-1), toPtr(1), &c, &s)
			goblas.Drot(&m, vt.Vector(idxjp-1, 0), ldvt, vt.Vector(idxj-1, 0), ldvt, &c, &s)
			if (*coltyp)[j-1] != (*coltyp)[jprev-1] {
				(*coltyp)[j-1] = 3
			}
			(*coltyp)[jprev-1] = 4
			k2 = k2 - 1
			(*idxp)[k2-1] = jprev
			jprev = j
		} else {
			(*k) = (*k) + 1
			u2.Set((*k)-1, 0, z.Get(jprev-1))
			dsigma.Set((*k)-1, d.Get(jprev-1))
			(*idxp)[(*k)-1] = jprev
			jprev = j
		}
	}
	goto label100
label110:
	;

	//     Record the last singular value.
	(*k) = (*k) + 1
	u2.Set((*k)-1, 0, z.Get(jprev-1))
	dsigma.Set((*k)-1, d.Get(jprev-1))
	(*idxp)[(*k)-1] = jprev

label120:
	;

	//     Count up the total number of the various types of columns, then
	//     form a permutation which positions the four column types into
	//     four groups of uniform structure (although one or more of these
	//     groups may be empty).
	for j = 1; j <= 4; j++ {
		ctot[j-1] = 0
	}
	for j = 2; j <= n; j++ {
		ct = (*coltyp)[j-1]
		ctot[ct-1] = ctot[ct-1] + 1
	}

	//     PSM(*) = Position in SubMatrix (of types 1 through 4)
	psm[0] = 2
	psm[1] = 2 + ctot[0]
	psm[2] = psm[1] + ctot[1]
	psm[3] = psm[2] + ctot[2]

	//     Fill out the IDXC array so that the permutation which it induces
	//     will place all type-1 columns first, all type-2 columns next,
	//     then all type-3's, and finally all type-4's, starting from the
	//     second column. This applies similarly to the rows of VT.
	for j = 2; j <= n; j++ {
		jp = (*idxp)[j-1]
		ct = (*coltyp)[jp-1]
		(*idxc)[psm[ct-1]-1] = j
		psm[ct-1] = psm[ct-1] + 1
	}

	//     Sort the singular values and corresponding singular vectors into
	//     DSIGMA, U2, and VT2 respectively.  The singular values/vectors
	//     which were not deflated go into the first K slots of DSIGMA, U2,
	//     and VT2 respectively, while those which were deflated go into the
	//     last N - K slots, except that the first column/row will be treated
	//     separately.
	for j = 2; j <= n; j++ {
		jp = (*idxp)[j-1]
		dsigma.Set(j-1, d.Get(jp-1))
		idxj = (*idxq)[(*idx)[(*idxp)[(*idxc)[j-1]-1]-1]+1-1]
		if idxj <= nlp1 {
			idxj = idxj - 1
		}
		goblas.Dcopy(&n, u.Vector(0, idxj-1), toPtr(1), u2.Vector(0, j-1), toPtr(1))
		goblas.Dcopy(&m, vt.Vector(idxj-1, 0), ldvt, vt2.Vector(j-1, 0), ldvt2)
	}

	//     Determine DSIGMA(1), DSIGMA(2) and Z(1)
	dsigma.Set(0, zero)
	hlftol = tol / two
	if math.Abs(dsigma.Get(1)) <= hlftol {
		dsigma.Set(1, hlftol)
	}
	if m > n {
		z.Set(0, Dlapy2(&z1, z.GetPtr(m-1)))
		if z.Get(0) <= tol {
			c = one
			s = zero
			z.Set(0, tol)
		} else {
			c = z1 / z.Get(0)
			s = z.Get(m-1) / z.Get(0)
		}
	} else {
		if math.Abs(z1) <= tol {
			z.Set(0, tol)
		} else {
			z.Set(0, z1)
		}
	}

	//     Move the rest of the updating row to Z.
	goblas.Dcopy(toPtr((*k)-1), u2.Vector(1, 0), toPtr(1), z.Off(1), toPtr(1))

	//     Determine the first column of U2, the first row of VT2 and the
	//     last row of VT.
	Dlaset('A', &n, toPtr(1), &zero, &zero, u2, ldu2)
	u2.Set(nlp1-1, 0, one)
	if m > n {
		for i = 1; i <= nlp1; i++ {
			vt.Set(m-1, i-1, -s*vt.Get(nlp1-1, i-1))
			vt2.Set(0, i-1, c*vt.Get(nlp1-1, i-1))
		}
		for i = nlp2; i <= m; i++ {
			vt2.Set(0, i-1, s*vt.Get(m-1, i-1))
			vt.Set(m-1, i-1, c*vt.Get(m-1, i-1))
		}
	} else {
		goblas.Dcopy(&m, vt.Vector(nlp1-1, 0), ldvt, vt2.Vector(0, 0), ldvt2)
	}
	if m > n {
		goblas.Dcopy(&m, vt.Vector(m-1, 0), ldvt, vt2.Vector(m-1, 0), ldvt2)
	}

	//     The deflated singular values and their corresponding vectors go
	//     into the back of D, U, and V respectively.
	if n > (*k) {
		goblas.Dcopy(toPtr(n-(*k)), dsigma.Off((*k)+1-1), toPtr(1), d.Off((*k)+1-1), toPtr(1))
		Dlacpy('A', &n, toPtr(n-(*k)), u2.Off(0, (*k)+1-1), ldu2, u.Off(0, (*k)+1-1), ldu)
		Dlacpy('A', toPtr(n-(*k)), &m, vt2.Off((*k)+1-1, 0), ldvt2, vt.Off((*k)+1-1, 0), ldvt)
	}

	//     Copy CTOT into COLTYP for referencing in DLASD3.
	for j = 1; j <= 4; j++ {
		(*coltyp)[j-1] = ctot[j-1]
	}
}
