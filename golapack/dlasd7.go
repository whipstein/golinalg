package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlasd7 merges the two sets of singular values together into a single
// sorted set. Then it tries to deflate the size of the problem. There
// are two ways in which deflation can occur:  when two or more singular
// values are close together or if there is a tiny entry in the Z
// vector. For each such occurrence the order of the related
// secular equation problem is reduced by one.
//
// DLASD7 is called from DLASD6.
func Dlasd7(icompq, nl, nr, sqre, k *int, d, z, zw, vf, vfw, vl, vlw *mat.Vector, alpha, beta *float64, dsigma *mat.Vector, idx, idxp, idxq, perm *[]int, givptr *int, givcol *[]int, ldgcol *int, givnum *mat.Matrix, ldgnum *int, c, s *float64, info *int) {
	var eight, eps, hlftol, one, tau, tol, two, z1, zero float64
	var i, idxi, idxj, idxjp, j, jp, jprev, k2, m, n, nlp1, nlp2 int

	zero = 0.0
	one = 1.0
	two = 2.0
	eight = 8.0

	//     Test the input parameters.
	(*info) = 0
	n = (*nl) + (*nr) + 1
	m = n + (*sqre)

	if ((*icompq) < 0) || ((*icompq) > 1) {
		(*info) = -1
	} else if (*nl) < 1 {
		(*info) = -2
	} else if (*nr) < 1 {
		(*info) = -3
	} else if ((*sqre) < 0) || ((*sqre) > 1) {
		(*info) = -4
	} else if (*ldgcol) < n {
		(*info) = -22
	} else if (*ldgnum) < n {
		(*info) = -24
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLASD7"), -(*info))
		return
	}

	nlp1 = (*nl) + 1
	nlp2 = (*nl) + 2
	if (*icompq) == 1 {
		(*givptr) = 0
	}

	//     Generate the first part of the vector Z and move the singular
	//     values in the first part of D one position backward.
	z1 = (*alpha) * vl.Get(nlp1-1)
	vl.Set(nlp1-1, zero)
	tau = vf.Get(nlp1 - 1)
	for i = (*nl); i >= 1; i-- {
		z.Set(i+1-1, (*alpha)*vl.Get(i-1))
		vl.Set(i-1, zero)
		vf.Set(i+1-1, vf.Get(i-1))
		d.Set(i+1-1, d.Get(i-1))
		(*idxq)[i+1-1] = (*idxq)[i-1] + 1
	}
	vf.Set(0, tau)

	//     Generate the second part of the vector Z.
	for i = nlp2; i <= m; i++ {
		z.Set(i-1, (*beta)*vf.Get(i-1))
		vf.Set(i-1, zero)
	}

	//     Sort the singular values into increasing order
	for i = nlp2; i <= n; i++ {
		(*idxq)[i-1] = (*idxq)[i-1] + nlp1
	}

	//     DSIGMA, IDXC, IDXC, and ZW are used as storage space.
	for i = 2; i <= n; i++ {
		dsigma.Set(i-1, d.Get((*idxq)[i-1]-1))
		zw.Set(i-1, z.Get((*idxq)[i-1]-1))
		vfw.Set(i-1, vf.Get((*idxq)[i-1]-1))
		vlw.Set(i-1, vl.Get((*idxq)[i-1]-1))
	}

	_idx1 := (*idx)[1:]
	Dlamrg(nl, nr, dsigma.Off(1), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), &_idx1)

	for i = 2; i <= n; i++ {
		idxi = 1 + (*idx)[i-1]
		d.Set(i-1, dsigma.Get(idxi-1))
		z.Set(i-1, zw.Get(idxi-1))
		vf.Set(i-1, vfw.Get(idxi-1))
		vl.Set(i-1, vlw.Get(idxi-1))
	}

	//     Calculate the allowable deflation tolerance
	eps = Dlamch(Epsilon)
	tol = maxf64(math.Abs(*alpha), math.Abs(*beta))
	tol = eight * eight * eps * maxf64(math.Abs(d.Get(n-1)), tol)

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
			if j == n {
				goto label100
			}
		} else {
			jprev = j
			goto label70
		}
	}
label70:
	;
	j = jprev
label80:
	;
	j = j + 1
	if j > n {
		goto label90
	}
	if math.Abs(z.Get(j-1)) <= tol {
		//        Deflate due to small z component.
		k2 = k2 - 1
		(*idxp)[k2-1] = j
	} else {
		//        Check if singular values are close enough to allow deflation.
		if math.Abs(d.Get(j-1)-d.Get(jprev-1)) <= tol {
			//           Deflation is possible.
			(*s) = z.Get(jprev - 1)
			(*c) = z.Get(j - 1)

			//           Find sqrt(a**2+b**2) without overflow or
			//           destructive underflow.
			tau = Dlapy2(c, s)
			z.Set(j-1, tau)
			z.Set(jprev-1, zero)
			(*c) = (*c) / tau
			(*s) = -(*s) / tau

			//           Record the appropriate Givens rotation
			if (*icompq) == 1 {
				(*givptr) = (*givptr) + 1
				idxjp = (*idxq)[(*idx)[jprev-1]+1-1]
				idxj = (*idxq)[(*idx)[j-1]+1-1]
				if idxjp <= nlp1 {
					idxjp = idxjp - 1
				}
				if idxj <= nlp1 {
					idxj = idxj - 1
				}
				(*givcol)[(*givptr)-1+1*(*ldgcol)] = idxjp
				(*givcol)[(*givptr)-1+0*(*ldgcol)] = idxj
				givnum.Set((*givptr)-1, 1, (*c))
				givnum.Set((*givptr)-1, 0, (*s))
			}
			goblas.Drot(1, vf.Off(jprev-1), 1, vf.Off(j-1), 1, *c, *s)
			goblas.Drot(1, vl.Off(jprev-1), 1, vl.Off(j-1), 1, *c, *s)
			k2 = k2 - 1
			(*idxp)[k2-1] = jprev
			jprev = j
		} else {
			(*k) = (*k) + 1
			zw.Set((*k)-1, z.Get(jprev-1))
			dsigma.Set((*k)-1, d.Get(jprev-1))
			(*idxp)[(*k)-1] = jprev
			jprev = j
		}
	}
	goto label80
label90:
	;

	//     Record the last singular value.
	(*k) = (*k) + 1
	zw.Set((*k)-1, z.Get(jprev-1))
	dsigma.Set((*k)-1, d.Get(jprev-1))
	(*idxp)[(*k)-1] = jprev

label100:
	;

	//     Sort the singular values into DSIGMA. The singular values which
	//     were not deflated go into the first K slots of DSIGMA, except
	//     that DSIGMA(1) is treated separately.
	for j = 2; j <= n; j++ {
		jp = (*idxp)[j-1]
		dsigma.Set(j-1, d.Get(jp-1))
		vfw.Set(j-1, vf.Get(jp-1))
		vlw.Set(j-1, vl.Get(jp-1))
	}
	if (*icompq) == 1 {
		for j = 2; j <= n; j++ {
			jp = (*idxp)[j-1]
			(*perm)[j-1] = (*idxq)[(*idx)[jp-1]+1-1]
			if (*perm)[j-1] <= nlp1 {
				(*perm)[j-1] = (*perm)[j-1] - 1
			}
		}
	}

	//     The deflated singular values go back into the last N - K slots of
	//     D.
	goblas.Dcopy(n-(*k), dsigma.Off((*k)+1-1), 1, d.Off((*k)+1-1), 1)

	//     Determine DSIGMA(1), DSIGMA(2), Z(1), VF(1), VL(1), VF(M), and
	//     VL(M).
	dsigma.Set(0, zero)
	hlftol = tol / two
	if math.Abs(dsigma.Get(1)) <= hlftol {
		dsigma.Set(1, hlftol)
	}
	if m > n {
		z.Set(0, Dlapy2(&z1, z.GetPtr(m-1)))
		if z.Get(0) <= tol {
			(*c) = one
			(*s) = zero
			z.Set(0, tol)
		} else {
			(*c) = z1 / z.Get(0)
			(*s) = -z.Get(m-1) / z.Get(0)
		}
		goblas.Drot(1, vf.Off(m-1), 1, vf, 1, *c, *s)
		goblas.Drot(1, vl.Off(m-1), 1, vl, 1, *c, *s)
	} else {
		if math.Abs(z1) <= tol {
			z.Set(0, tol)
		} else {
			z.Set(0, z1)
		}
	}

	//     Restore Z, VF, and VL.
	goblas.Dcopy((*k)-1, zw.Off(1), 1, z.Off(1), 1)
	goblas.Dcopy(n-1, vfw.Off(1), 1, vf.Off(1), 1)
	goblas.Dcopy(n-1, vlw.Off(1), 1, vl.Off(1), 1)
}
