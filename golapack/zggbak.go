package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zggbak forms the right or left eigenvectors of a complex generalized
// eigenvalue problem A*x = lambda*B*x, by backward transformation on
// the computed eigenvectors of the balanced pair of matrices output by
// ZGGBAL.
func Zggbak(job, side byte, n, ilo, ihi *int, lscale, rscale *mat.Vector, m *int, v *mat.CMatrix, ldv, info *int) {
	var leftv, rightv bool
	var i, k int

	//     Test the input parameters
	rightv = side == 'R'
	leftv = side == 'L'

	(*info) = 0
	if job != 'N' && job != 'P' && job != 'S' && job != 'B' {
		(*info) = -1
	} else if !rightv && !leftv {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*ilo) < 1 {
		(*info) = -4
	} else if (*n) == 0 && (*ihi) == 0 && (*ilo) != 1 {
		(*info) = -4
	} else if (*n) > 0 && ((*ihi) < (*ilo) || (*ihi) > maxint(1, *n)) {
		(*info) = -5
	} else if (*n) == 0 && (*ilo) == 1 && (*ihi) != 0 {
		(*info) = -5
	} else if (*m) < 0 {
		(*info) = -8
	} else if (*ldv) < maxint(1, *n) {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGGBAK"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}
	if (*m) == 0 {
		return
	}
	if job == 'N' {
		return
	}

	if (*ilo) == (*ihi) {
		goto label30
	}

	//     Backward balance
	if job == 'S' || job == 'B' {
		//        Backward transformation on right eigenvectors
		if rightv {
			for i = (*ilo); i <= (*ihi); i++ {
				goblas.Zdscal(m, rscale.GetPtr(i-1), v.CVector(i-1, 0), ldv)
			}
		}

		//        Backward transformation on left eigenvectors
		if leftv {
			for i = (*ilo); i <= (*ihi); i++ {
				goblas.Zdscal(m, lscale.GetPtr(i-1), v.CVector(i-1, 0), ldv)
			}
		}
	}

	//     Backward permutation
label30:
	;
	if job == 'P' || job == 'B' {
		//        Backward permutation on right eigenvectors
		if rightv {
			if (*ilo) == 1 {
				goto label50
			}
			for i = (*ilo) - 1; i >= 1; i-- {
				k = int(rscale.Get(i - 1))
				if k == i {
					goto label40
				}
				goblas.Zswap(m, v.CVector(i-1, 0), ldv, v.CVector(k-1, 0), ldv)
			label40:
			}

		label50:
			;
			if (*ihi) == (*n) {
				goto label70
			}
			for i = (*ihi) + 1; i <= (*n); i++ {
				k = int(rscale.Get(i - 1))
				if k == i {
					goto label60
				}
				goblas.Zswap(m, v.CVector(i-1, 0), ldv, v.CVector(k-1, 0), ldv)
			label60:
			}
		}

		//        Backward permutation on left eigenvectors
	label70:
		;
		if leftv {
			if (*ilo) == 1 {
				goto label90
			}
			for i = (*ilo) - 1; i >= 1; i-- {
				k = int(lscale.Get(i - 1))
				if k == i {
					goto label80
				}
				goblas.Zswap(m, v.CVector(i-1, 0), ldv, v.CVector(k-1, 0), ldv)
			label80:
			}

		label90:
			;
			if (*ihi) == (*n) {
				return
			}
			for i = (*ihi) + 1; i <= (*n); i++ {
				k = int(lscale.Get(i - 1))
				if k == i {
					goto label100
				}
				goblas.Zswap(m, v.CVector(i-1, 0), ldv, v.CVector(k-1, 0), ldv)
			label100:
			}
		}
	}
}