package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgebak forms the right or left eigenvectors of a real general matrix
// by backward transformation on the computed eigenvectors of the
// balanced matrix output by DGEBAL.
func Dgebak(job, side byte, n, ilo, ihi *int, scale *mat.Vector, m *int, v *mat.Matrix, ldv, info *int) {
	var leftv, rightv bool
	var one, s float64
	var i, ii, k int

	one = 1.0

	//     Decode and Test the input parameters
	rightv = side == 'R'
	leftv = side == 'L'

	(*info) = 0
	if job != 'N' && job != 'P' && job != 'S' && job != 'B' {
		(*info) = -1
	} else if !rightv && !leftv {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*ilo) < 1 || (*ilo) > maxint(1, *n) {
		(*info) = -4
	} else if (*ihi) < minint(*ilo, *n) || (*ihi) > (*n) {
		(*info) = -5
	} else if (*m) < 0 {
		(*info) = -7
	} else if (*ldv) < maxint(1, *n) {
		(*info) = -9
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGEBAK"), -(*info))
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

		if rightv {
			for i = (*ilo); i <= (*ihi); i++ {
				s = scale.Get(i - 1)
				goblas.Dscal(*m, s, v.Vector(i-1, 0), *ldv)
			}
		}

		if leftv {
			for i = (*ilo); i <= (*ihi); i++ {
				s = one / scale.Get(i-1)
				goblas.Dscal(*m, s, v.Vector(i-1, 0), *ldv)
			}
		}

	}

	//     Backward permutation
	//
	//     For  I = ILO-1 step -1 until 1,
	//              IHI+1 step 1 until N do --
label30:
	;
	if job == 'P' || job == 'B' {
		if rightv {
			for ii = 1; ii <= (*n); ii++ {
				i = ii
				if i >= (*ilo) && i <= (*ihi) {
					goto label40
				}
				if i < (*ilo) {
					i = (*ilo) - ii
				}
				k = int(scale.Get(i - 1))
				if k == i {
					goto label40
				}
				goblas.Dswap(*m, v.Vector(i-1, 0), *ldv, v.Vector(k-1, 0), *ldv)
			label40:
			}
		}

		if leftv {
			for ii = 1; ii <= (*n); ii++ {
				i = ii
				if i >= (*ilo) && i <= (*ihi) {
					goto label50
				}
				if i < (*ilo) {
					i = (*ilo) - ii
				}
				k = int(scale.Get(i - 1))
				if k == i {
					goto label50
				}
				goblas.Dswap(*m, v.Vector(i-1, 0), *ldv, v.Vector(k-1, 0), *ldv)
			label50:
			}
		}
	}
}
