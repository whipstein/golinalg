package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dlaed2 merges the two sets of eigenvalues together into a single
// sorted set.  Then it tries to deflate the size of the problem.
// There are two ways in which deflation can occur:  when two or more
// eigenvalues are close together or if there is a tiny entry in the
// Z vector.  For each such occurrence the order of the related secular
// equation problem is reduced by one.
func Dlaed2(k, n, n1 *int, d *mat.Vector, q *mat.Matrix, ldq *int, indxq *[]int, rho *float64, z, dlamda, w, q2 *mat.Vector, indx, indxc, indxp, coltyp *[]int, info *int) {
	var c, eight, eps, mone, one, s, t, tau, tol, two, zero float64
	var ct, i, imax, iq1, iq2, j, jmax, js, k2, n1p1, n2, nj, pj int
	ctot := make([]int, 4)
	psm := make([]int, 4)

	mone = -1.0
	zero = 0.0
	one = 1.0
	two = 2.0
	eight = 8.0

	//     Test the input parameters.
	(*info) = 0

	if (*n) < 0 {
		(*info) = -2
	} else if (*ldq) < maxint(1, *n) {
		(*info) = -6
	} else if minint(1, (*n)/2) > (*n1) || ((*n)/2) < (*n1) {
		(*info) = -3
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLAED2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	n2 = (*n) - (*n1)
	n1p1 = (*n1) + 1

	if (*rho) < zero {
		goblas.Dscal(&n2, &mone, z.Off(n1p1-1), toPtr(1))
	}

	//     Normalize z so that norm(z) = 1.  Since z is the concatenation of
	//     two normalized vectors, norm2(z) = math.Sqrt(2).
	t = one / math.Sqrt(two)
	goblas.Dscal(n, &t, z, toPtr(1))

	//     RHO = ABS( norm(z)**2 * RHO )
	(*rho) = math.Abs(two * (*rho))

	//     Sort the eigenvalues into increasing order
	for i = n1p1; i <= (*n); i++ {
		(*indxq)[i-1] = (*indxq)[i-1] + (*n1)
	}

	//     re-integrate the deflated parts from the last pass
	for i = 1; i <= (*n); i++ {
		dlamda.Set(i-1, d.Get((*indxq)[i-1]-1))
	}
	Dlamrg(n1, &n2, dlamda, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), indxc)
	for i = 1; i <= (*n); i++ {
		(*indx)[i-1] = (*indxq)[(*indxc)[i-1]-1]
	}

	//     Calculate the allowable deflation tolerance
	imax = goblas.Idamax(n, z, toPtr(1))
	jmax = goblas.Idamax(n, d, toPtr(1))
	eps = Dlamch(Epsilon)
	tol = eight * eps * maxf64(math.Abs(d.Get(jmax-1)), math.Abs(z.Get(imax-1)))

	//     If the rank-1 modifier is small enough, no more needs to be done
	//     except to reorganize Q so that its columns correspond with the
	//     elements in D.
	if (*rho)*math.Abs(z.Get(imax-1)) <= tol {
		(*k) = 0
		iq2 = 1
		for j = 1; j <= (*n); j++ {
			i = (*indx)[j-1]
			goblas.Dcopy(n, q.Vector(0, i-1), toPtr(1), q2.Off(iq2-1), toPtr(1))
			dlamda.Set(j-1, d.Get(i-1))
			iq2 = iq2 + (*n)
		}
		Dlacpy('A', n, n, q2.Matrix(*n, opts), n, q, ldq)
		goblas.Dcopy(n, dlamda, toPtr(1), d, toPtr(1))
		return
	}

	//     If there are multiple eigenvalues then the problem deflates.  Here
	//     the number of equal eigenvalues are found.  As each equal
	//     eigenvalue is found, an elementary reflector is computed to rotate
	//     the corresponding eigensubspace so that the corresponding
	//     components of Z are zero in this new basis.
	for i = 1; i <= (*n1); i++ {
		(*coltyp)[i-1] = 1
	}
	for i = n1p1; i <= (*n); i++ {
		(*coltyp)[i-1] = 3
	}

	(*k) = 0
	k2 = (*n) + 1
	for j = 1; j <= (*n); j++ {
		nj = (*indx)[j-1]
		if (*rho)*math.Abs(z.Get(nj-1)) <= tol {
			//           Deflate due to small z component.
			k2 = k2 - 1
			(*coltyp)[nj-1] = 4
			(*indxp)[k2-1] = nj
			if j == (*n) {
				goto label100
			}
		} else {
			pj = nj
			goto label80
		}
	}
label80:
	;
	j = j + 1
	nj = (*indx)[j-1]
	if j > (*n) {
		goto label100
	}
	if (*rho)*math.Abs(z.Get(nj-1)) <= tol {
		//        Deflate due to small z component.
		k2 = k2 - 1
		(*coltyp)[nj-1] = 4
		(*indxp)[k2-1] = nj
	} else {
		//        Check if eigenvalues are close enough to allow deflation.
		s = z.Get(pj - 1)
		c = z.Get(nj - 1)

		//        Find math.Sqrt(a**2+b**2) without overflow or
		//        destructive underflow.
		tau = Dlapy2(&c, &s)
		t = d.Get(nj-1) - d.Get(pj-1)
		c = c / tau
		s = -s / tau
		if math.Abs(t*c*s) <= tol {
			//           Deflation is possible.
			z.Set(nj-1, tau)
			z.Set(pj-1, zero)
			if (*coltyp)[nj-1] != (*coltyp)[pj-1] {
				(*coltyp)[nj-1] = 2
			}
			(*coltyp)[pj-1] = 4
			goblas.Drot(n, q.Vector(0, pj-1), toPtr(1), q.Vector(0, nj-1), toPtr(1), &c, &s)
			t = d.Get(pj-1)*math.Pow(c, 2) + d.Get(nj-1)*math.Pow(s, 2)
			d.Set(nj-1, d.Get(pj-1)*math.Pow(s, 2)+d.Get(nj-1)*math.Pow(c, 2))
			d.Set(pj-1, t)
			k2 = k2 - 1
			i = 1
		label90:
			;
			if k2+i <= (*n) {
				if d.Get(pj-1) < d.Get((*indxp)[k2+i-1]-1) {
					(*indxp)[k2+i-1-1] = (*indxp)[k2+i-1]
					(*indxp)[k2+i-1] = pj
					i = i + 1
					goto label90
				} else {
					(*indxp)[k2+i-1-1] = pj
				}
			} else {
				(*indxp)[k2+i-1-1] = pj
			}
			pj = nj
		} else {
			(*k) = (*k) + 1
			dlamda.Set((*k)-1, d.Get(pj-1))
			w.Set((*k)-1, z.Get(pj-1))
			(*indxp)[(*k)-1] = pj
			pj = nj
		}
	}
	goto label80
label100:
	;

	//     Record the last eigenvalue.
	(*k) = (*k) + 1
	dlamda.Set((*k)-1, d.Get(pj-1))
	w.Set((*k)-1, z.Get(pj-1))
	(*indxp)[(*k)-1] = pj

	//     Count up the total number of the various types of columns, then
	//     form a permutation which positions the four column types into
	//     four uniform groups (although one or more of these groups may be
	//     empty).
	for j = 1; j <= 4; j++ {
		ctot[j-1] = 0
	}
	for j = 1; j <= (*n); j++ {
		ct = (*coltyp)[j-1]
		ctot[ct-1] = ctot[ct-1] + 1
	}

	//     PSM(*) = Position in SubMatrix (of types 1 through 4)
	psm[0] = 1
	psm[1] = 1 + ctot[0]
	psm[2] = psm[1] + ctot[1]
	psm[3] = psm[2] + ctot[2]
	(*k) = (*n) - ctot[3]

	//     Fill out the INDXC array so that the permutation which it induces
	//     will place all type-1 columns first, all type-2 columns next,
	//     then all type-3's, and finally all type-4's.
	for j = 1; j <= (*n); j++ {
		js = (*indxp)[j-1]
		ct = (*coltyp)[js-1]
		(*indx)[psm[ct-1]-1] = js
		(*indxc)[psm[ct-1]-1] = j
		psm[ct-1] = psm[ct-1] + 1
	}

	//     Sort the eigenvalues and corresponding eigenvectors into DLAMDA
	//     and Q2 respectively.  The eigenvalues/vectors which were not
	//     deflated go into the first K slots of DLAMDA and Q2 respectively,
	//     while those which were deflated go into the last N - K slots.
	i = 1
	iq1 = 1
	iq2 = 1 + (ctot[0]+ctot[1])*(*n1)
	for j = 1; j <= ctot[0]; j++ {
		js = (*indx)[i-1]
		goblas.Dcopy(n1, q.Vector(0, js-1), toPtr(1), q2.Off(iq1-1), toPtr(1))
		z.Set(i-1, d.Get(js-1))
		i = i + 1
		iq1 = iq1 + (*n1)
	}

	for j = 1; j <= ctot[1]; j++ {
		js = (*indx)[i-1]
		goblas.Dcopy(n1, q.Vector(0, js-1), toPtr(1), q2.Off(iq1-1), toPtr(1))
		goblas.Dcopy(&n2, q.Vector((*n1)+1-1, js-1), toPtr(1), q2.Off(iq2-1), toPtr(1))
		z.Set(i-1, d.Get(js-1))
		i = i + 1
		iq1 = iq1 + (*n1)
		iq2 = iq2 + n2
	}

	for j = 1; j <= ctot[2]; j++ {
		js = (*indx)[i-1]
		goblas.Dcopy(&n2, q.Vector((*n1)+1-1, js-1), toPtr(1), q2.Off(iq2-1), toPtr(1))
		z.Set(i-1, d.Get(js-1))
		i = i + 1
		iq2 = iq2 + n2
	}

	iq1 = iq2
	for j = 1; j <= ctot[3]; j++ {
		js = (*indx)[i-1]
		goblas.Dcopy(n, q.Vector(0, js-1), toPtr(1), q2.Off(iq2-1), toPtr(1))
		iq2 = iq2 + (*n)
		z.Set(i-1, d.Get(js-1))
		i = i + 1
	}

	//     The deflated eigenvalues and their corresponding vectors go back
	//     into the last N - K slots of D and Q respectively.
	if (*k) < (*n) {
		Dlacpy('A', n, &(ctot[3]), q2.MatrixOff(iq1-1, *n, opts), n, q.Off(0, (*k)+1-1), ldq)
		goblas.Dcopy(toPtr((*n)-(*k)), z.Off((*k)+1-1), toPtr(1), d.Off((*k)+1-1), toPtr(1))
	}

	//     Copy CTOT into COLTYP for referencing in DLAED3.
	for j = 1; j <= 4; j++ {
		(*coltyp)[j-1] = ctot[j-1]
	}
}