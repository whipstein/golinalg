package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zunghr generates a complex unitary matrix Q which is defined as the
// product of IHI-ILO elementary reflectors of order N, as returned by
// ZGEHRD:
//
// Q = H(ilo) H(ilo+1) . . . H(ihi-1).
func Zunghr(n, ilo, ihi *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var one, zero complex128
	var i, iinfo, j, lwkopt, nb, nh int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Test the input arguments
	(*info) = 0
	nh = (*ihi) - (*ilo)
	lquery = ((*lwork) == -1)
	if (*n) < 0 {
		(*info) = -1
	} else if (*ilo) < 1 || (*ilo) > maxint(1, *n) {
		(*info) = -2
	} else if (*ihi) < minint(*ilo, *n) || (*ihi) > (*n) {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*lwork) < maxint(1, nh) && !lquery {
		(*info) = -8
	}

	if (*info) == 0 {
		nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNGQR"), []byte{' '}, &nh, &nh, &nh, toPtr(-1))
		lwkopt = maxint(1, nh) * nb
		work.SetRe(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNGHR"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		work.Set(0, 1)
		return
	}

	//     Shift the vectors which define the elementary reflectors one
	//     column to the right, and set the first ilo and the last n-ihi
	//     rows and columns to those of the unit matrix
	for j = (*ihi); j >= (*ilo)+1; j-- {
		for i = 1; i <= j-1; i++ {
			a.Set(i-1, j-1, zero)
		}
		for i = j + 1; i <= (*ihi); i++ {
			a.Set(i-1, j-1, a.Get(i-1, j-1-1))
		}
		for i = (*ihi) + 1; i <= (*n); i++ {
			a.Set(i-1, j-1, zero)
		}
	}
	for j = 1; j <= (*ilo); j++ {
		for i = 1; i <= (*n); i++ {
			a.Set(i-1, j-1, zero)
		}
		a.Set(j-1, j-1, one)
	}
	for j = (*ihi) + 1; j <= (*n); j++ {
		for i = 1; i <= (*n); i++ {
			a.Set(i-1, j-1, zero)
		}
		a.Set(j-1, j-1, one)
	}

	if nh > 0 {
		//        Generate Q(ilo+1:ihi,ilo+1:ihi)
		Zungqr(&nh, &nh, &nh, a.Off((*ilo)+1-1, (*ilo)+1-1), lda, tau.Off((*ilo)-1), work, lwork, &iinfo)
	}
	work.SetRe(0, float64(lwkopt))
}
