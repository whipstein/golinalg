package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlaed8 merges the two sets of eigenvalues together into a single
// sorted set.  Then it tries to deflate the size of the problem.
// There are two ways in which deflation can occur:  when two or more
// eigenvalues are close together or if there is a tiny element in the
// Z vector.  For each such occurrence the order of the related secular
// equation problem is reduced by one.
func Zlaed8(k, n, qsiz *int, q *mat.CMatrix, ldq *int, d *mat.Vector, rho *float64, cutpnt *int, z, dlamda *mat.Vector, q2 *mat.CMatrix, ldq2 *int, w *mat.Vector, indxp, indx, indxq, perm *[]int, givptr *int, givcol *[]int, givnum *mat.Matrix, info *int) {
	var c, eight, eps, mone, one, s, t, tau, tol, two, zero float64
	var i, imax, j, jlam, jmax, jp, k2, n1, n1p1, n2 int

	mone = -1.0
	zero = 0.0
	one = 1.0
	two = 2.0
	eight = 8.0

	//     Test the input parameters.
	(*info) = 0

	if (*n) < 0 {
		(*info) = -2
	} else if (*qsiz) < (*n) {
		(*info) = -3
	} else if (*ldq) < max(1, *n) {
		(*info) = -5
	} else if (*cutpnt) < min(int(1), *n) || (*cutpnt) > (*n) {
		(*info) = -8
	} else if (*ldq2) < max(1, *n) {
		(*info) = -12
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZLAED8"), -(*info))
		return
	}

	//     Need to initialize GIVPTR to O here in case of quick exit
	//     to prevent an unspecified code behavior (usually sigfault)
	//     when IWORK array on entry to *stedc is not zeroed
	//     (or at least some IWORK entries which used in *laed7 for GIVPTR).
	(*givptr) = 0

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	n1 = (*cutpnt)
	n2 = (*n) - n1
	n1p1 = n1 + 1

	if (*rho) < zero {
		goblas.Dscal(n2, mone, z.Off(n1p1-1, 1))
	}

	//     Normalize z so that norm(z) = 1
	t = one / math.Sqrt(two)
	for j = 1; j <= (*n); j++ {
		(*indx)[j-1] = j
	}
	goblas.Dscal(*n, t, z.Off(0, 1))
	(*rho) = math.Abs(two * (*rho))

	//     Sort the eigenvalues into increasing order
	for i = (*cutpnt) + 1; i <= (*n); i++ {
		(*indxq)[i-1] = (*indxq)[i-1] + (*cutpnt)
	}
	for i = 1; i <= (*n); i++ {
		dlamda.Set(i-1, d.Get((*indxq)[i-1]-1))
		w.Set(i-1, z.Get((*indxq)[i-1]-1))
	}
	i = 1
	j = (*cutpnt) + 1
	Dlamrg(&n1, &n2, dlamda, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), indx)
	for i = 1; i <= (*n); i++ {
		d.Set(i-1, dlamda.Get((*indx)[i-1]-1))
		z.Set(i-1, w.Get((*indx)[i-1]-1))
	}

	//     Calculate the allowable deflation tolerance
	imax = goblas.Idamax(*n, z.Off(0, 1))
	jmax = goblas.Idamax(*n, d.Off(0, 1))
	eps = Dlamch(Epsilon)
	tol = eight * eps * d.GetMag(jmax-1)

	//     If the rank-1 modifier is small enough, no more needs to be done
	//     -- except to reorganize Q so that its columns correspond with the
	//     elements in D.
	if (*rho)*z.GetMag(imax-1) <= tol {
		(*k) = 0
		for j = 1; j <= (*n); j++ {
			(*perm)[j-1] = (*indxq)[(*indx)[j-1]-1]
			goblas.Zcopy(*qsiz, q.CVector(0, (*perm)[j-1]-1, 1), q2.CVector(0, j-1, 1))
		}
		Zlacpy('A', qsiz, n, q2, ldq2, q, ldq)
		return
	}

	//     If there are multiple eigenvalues then the problem deflates.  Here
	//     the number of equal eigenvalues are found.  As each equal
	//     eigenvalue is found, an elementary reflector is computed to rotate
	//     the corresponding eigensubspace so that the corresponding
	//     components of Z are zero in this new basis.
	(*k) = 0
	k2 = (*n) + 1
	for j = 1; j <= (*n); j++ {
		if (*rho)*z.GetMag(j-1) <= tol {
			//           Deflate due to small z component.
			k2 = k2 - 1
			(*indxp)[k2-1] = j
			if j == (*n) {
				goto label100
			}
		} else {
			jlam = j
			goto label70
		}
	}
label70:
	;
	j = j + 1
	if j > (*n) {
		goto label90
	}
	if (*rho)*z.GetMag(j-1) <= tol {
		//        Deflate due to small z component.
		k2 = k2 - 1
		(*indxp)[k2-1] = j
	} else {
		//        Check if eigenvalues are close enough to allow deflation.
		s = z.Get(jlam - 1)
		c = z.Get(j - 1)

		//        Find sqrt(a**2+b**2) without overflow or
		//        destructive underflow.
		tau = Dlapy2(&c, &s)
		t = d.Get(j-1) - d.Get(jlam-1)
		c = c / tau
		s = -s / tau
		if math.Abs(t*c*s) <= tol {
			//           Deflation is possible.
			z.Set(j-1, tau)
			z.Set(jlam-1, zero)

			//           Record the appropriate Givens rotation
			(*givptr) = (*givptr) + 1
			(*givcol)[0+((*givptr)-1)*2] = (*indxq)[(*indx)[jlam-1]-1]
			(*givcol)[1+((*givptr)-1)*2] = (*indxq)[(*indx)[j-1]-1]
			givnum.Set(0, (*givptr)-1, c)
			givnum.Set(1, (*givptr)-1, s)
			goblas.Zdrot(*qsiz, q.CVector(0, (*indxq)[(*indx)[jlam-1]-1]-1, 1), q.CVector(0, (*indxq)[(*indx)[j-1]-1]-1, 1), c, s)
			t = d.Get(jlam-1)*c*c + d.Get(j-1)*s*s
			d.Set(j-1, d.Get(jlam-1)*s*s+d.Get(j-1)*c*c)
			d.Set(jlam-1, t)
			k2 = k2 - 1
			i = 1
		label80:
			;
			if k2+i <= (*n) {
				if d.Get(jlam-1) < d.Get((*indxp)[k2+i-1]-1) {
					(*indxp)[k2+i-1-1] = (*indxp)[k2+i-1]
					(*indxp)[k2+i-1] = jlam
					i = i + 1
					goto label80
				} else {
					(*indxp)[k2+i-1-1] = jlam
				}
			} else {
				(*indxp)[k2+i-1-1] = jlam
			}
			jlam = j
		} else {
			(*k) = (*k) + 1
			w.Set((*k)-1, z.Get(jlam-1))
			dlamda.Set((*k)-1, d.Get(jlam-1))
			(*indxp)[(*k)-1] = jlam
			jlam = j
		}
	}
	goto label70
label90:
	;

	//     Record the last eigenvalue.
	(*k) = (*k) + 1
	w.Set((*k)-1, z.Get(jlam-1))
	dlamda.Set((*k)-1, d.Get(jlam-1))
	(*indxp)[(*k)-1] = jlam

label100:
	;

	//     Sort the eigenvalues and corresponding eigenvectors into DLAMDA
	//     and Q2 respectively.  The eigenvalues/vectors which were not
	//     deflated go into the first K slots of DLAMDA and Q2 respectively,
	//     while those which were deflated go into the last N - K slots.
	for j = 1; j <= (*n); j++ {
		jp = (*indxp)[j-1]
		dlamda.Set(j-1, d.Get(jp-1))
		(*perm)[j-1] = (*indxq)[(*indx)[jp-1]-1]
		goblas.Zcopy(*qsiz, q.CVector(0, (*perm)[j-1]-1, 1), q2.CVector(0, j-1, 1))
	}

	//     The deflated eigenvalues and their corresponding vectors go back
	//     into the last N - K slots of D and Q respectively.
	if (*k) < (*n) {
		goblas.Dcopy((*n)-(*k), dlamda.Off((*k), 1), d.Off((*k), 1))
		Zlacpy('A', qsiz, toPtr((*n)-(*k)), q2.Off(0, (*k)), ldq2, q.Off(0, (*k)), ldq)
	}
}
