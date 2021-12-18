package golapack

import (
	"fmt"
	"math"

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
// Dlasd2 is called from DLASD1.
func Dlasd2(nl, nr, sqre int, d, z *mat.Vector, alpha, beta float64, u, vt *mat.Matrix, dsigma *mat.Vector, u2, vt2 *mat.Matrix, idxp, idx, idxc, idxq, coltyp *[]int) (k int, err error) {
	var c, eight, eps, hlftol, one, s, tau, tol, two, z1, zero float64
	var ct, i, idxi, idxj, idxjp, j, jp, jprev, k2, m, n, nlp1, nlp2 int
	ctot := make([]int, 4)
	psm := make([]int, 4)

	zero = 0.0
	one = 1.0
	two = 2.0
	eight = 8.0

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

	if u.Rows < n {
		err = fmt.Errorf("u.Rows < n: u.Rows=%v, n=%v", u.Rows, n)
	} else if vt.Rows < m {
		err = fmt.Errorf("vt.Rows < m: vt.Rows=%v, m=%v", vt.Rows, m)
	} else if u2.Rows < n {
		err = fmt.Errorf("u2.Rows < n: u2.Rows=%v, n=%v", u2.Rows, n)
	} else if vt2.Rows < m {
		err = fmt.Errorf("vt2.Rows < m: vt2.Rows=%v, m=%v", vt2.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Dlasd2", err)
		return
	}

	nlp1 = nl + 1
	nlp2 = nl + 2

	//     Generate the first part of the vector Z; and move the singular
	//     values in the first part of D one position backward.
	z1 = alpha * vt.Get(nlp1-1, nlp1-1)
	z.Set(0, z1)
	for i = nl; i >= 1; i-- {
		z.Set(i, alpha*vt.Get(i-1, nlp1-1))
		d.Set(i, d.Get(i-1))
		(*idxq)[i] = (*idxq)[i-1] + 1
	}

	//     Generate the second part of the vector Z.
	for i = nlp2; i <= m; i++ {
		z.Set(i-1, beta*vt.Get(i-1, nlp2-1))
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

	Dlamrg(nl, nr, dsigma.Off(1), 1, 1, toSlice(idx, 1))

	for i = 2; i <= n; i++ {
		idxi = 1 + (*idx)[i-1]
		d.Set(i-1, dsigma.Get(idxi-1))
		z.Set(i-1, u2.Get(idxi-1, 0))
		(*coltyp)[i-1] = (*idxc)[idxi-1]
	}

	//     Calculate the allowable deflation tolerance
	eps = Dlamch(Epsilon)
	tol = math.Max(math.Abs(alpha), math.Abs(beta))
	tol = eight * eps * math.Max(math.Abs(d.Get(n-1)), tol)

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
	k = 1
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
			tau = Dlapy2(c, s)
			c = c / tau
			s = -s / tau
			z.Set(j-1, tau)
			z.Set(jprev-1, zero)

			//           Apply back the Givens rotation to the left and right
			//           singular vector matrices.
			idxjp = (*idxq)[(*idx)[jprev-1]]
			idxj = (*idxq)[(*idx)[j-1]]
			if idxjp <= nlp1 {
				idxjp = idxjp - 1
			}
			if idxj <= nlp1 {
				idxj = idxj - 1
			}
			u.Off(0, idxj-1).Vector().Rot(n, u.Off(0, idxjp-1).Vector(), 1, 1, c, s)
			vt.Off(idxj-1, 0).Vector().Rot(m, vt.Off(idxjp-1, 0).Vector(), vt.Rows, vt.Rows, c, s)
			if (*coltyp)[j-1] != (*coltyp)[jprev-1] {
				(*coltyp)[j-1] = 3
			}
			(*coltyp)[jprev-1] = 4
			k2 = k2 - 1
			(*idxp)[k2-1] = jprev
			jprev = j
		} else {
			k = k + 1
			u2.Set(k-1, 0, z.Get(jprev-1))
			dsigma.Set(k-1, d.Get(jprev-1))
			(*idxp)[k-1] = jprev
			jprev = j
		}
	}
	goto label100
label110:
	;

	//     Record the last singular value.
	k = k + 1
	u2.Set(k-1, 0, z.Get(jprev-1))
	dsigma.Set(k-1, d.Get(jprev-1))
	(*idxp)[k-1] = jprev

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
		idxj = (*idxq)[(*idx)[(*idxp)[(*idxc)[j-1]-1]-1]]
		if idxj <= nlp1 {
			idxj = idxj - 1
		}
		u2.Off(0, j-1).Vector().Copy(n, u.Off(0, idxj-1).Vector(), 1, 1)
		vt2.Off(j-1, 0).Vector().Copy(m, vt.Off(idxj-1, 0).Vector(), vt.Rows, vt2.Rows)
	}

	//     Determine DSIGMA(1), DSIGMA(2) and Z(1)
	dsigma.Set(0, zero)
	hlftol = tol / two
	if math.Abs(dsigma.Get(1)) <= hlftol {
		dsigma.Set(1, hlftol)
	}
	if m > n {
		z.Set(0, Dlapy2(z1, z.Get(m-1)))
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
	z.Off(1).Copy(k-1, u2.Off(1, 0).Vector(), 1, 1)

	//     Determine the first column of U2, the first row of VT2 and the
	//     last row of VT.
	Dlaset(Full, n, 1, zero, zero, u2)
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
		vt2.Off(0, 0).Vector().Copy(m, vt.Off(nlp1-1, 0).Vector(), vt.Rows, vt2.Rows)
	}
	if m > n {
		vt2.Off(m-1, 0).Vector().Copy(m, vt.Off(m-1, 0).Vector(), vt.Rows, vt2.Rows)
	}

	//     The deflated singular values and their corresponding vectors go
	//     into the back of D, U, and V respectively.
	if n > k {
		d.Off(k).Copy(n-k, dsigma.Off(k), 1, 1)
		Dlacpy(Full, n, n-k, u2.Off(0, k), u.Off(0, k))
		Dlacpy(Full, n-k, m, vt2.Off(k, 0), vt.Off(k, 0))
	}

	//     Copy CTOT into COLTYP for referencing in DLASD3.
	for j = 1; j <= 4; j++ {
		(*coltyp)[j-1] = ctot[j-1]
	}

	return
}
