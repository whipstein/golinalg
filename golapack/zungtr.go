package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zungtr generates a complex unitary matrix Q which is defined as the
// product of n-1 elementary reflectors of order N, as returned by
// ZHETRD:
//
// if UPLO = 'U', Q = H(n-1) . . . H(2) H(1),
//
// if UPLO = 'L', Q = H(1) H(2) . . . H(n-1).
func Zungtr(uplo byte, n *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork, info *int) {
	var lquery, upper bool
	var one, zero complex128
	var i, iinfo, j, lwkopt, nb int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Test the input arguments
	(*info) = 0
	lquery = ((*lwork) == -1)
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	} else if (*lwork) < maxint(1, (*n)-1) && !lquery {
		(*info) = -7
	}

	if (*info) == 0 {
		if upper {
			nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNGQL"), []byte{' '}, toPtr((*n)-1), toPtr((*n)-1), toPtr((*n)-1), toPtr(-1))
		} else {
			nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNGQR"), []byte{' '}, toPtr((*n)-1), toPtr((*n)-1), toPtr((*n)-1), toPtr(-1))
		}
		lwkopt = maxint(1, (*n)-1) * nb
		work.SetRe(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNGTR"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		work.Set(0, 1)
		return
	}

	if upper {
		//        Q was determined by a call to ZHETRD with UPLO = 'U'
		//
		//        Shift the vectors which define the elementary reflectors one
		//        column to the left, and set the last row and column of Q to
		//        those of the unit matrix
		for j = 1; j <= (*n)-1; j++ {
			for i = 1; i <= j-1; i++ {
				a.Set(i-1, j-1, a.Get(i-1, j+1-1))
			}
			a.Set((*n)-1, j-1, zero)
		}
		for i = 1; i <= (*n)-1; i++ {
			a.Set(i-1, (*n)-1, zero)
		}
		a.Set((*n)-1, (*n)-1, one)

		//        Generate Q(1:n-1,1:n-1)
		Zungql(toPtr((*n)-1), toPtr((*n)-1), toPtr((*n)-1), a, lda, tau, work, lwork, &iinfo)

	} else {
		//        Q was determined by a call to ZHETRD with UPLO = 'L'.
		//
		//        Shift the vectors which define the elementary reflectors one
		//        column to the right, and set the first row and column of Q to
		//        those of the unit matrix
		for j = (*n); j >= 2; j-- {
			a.Set(0, j-1, zero)
			for i = j + 1; i <= (*n); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1-1))
			}
		}
		a.Set(0, 0, one)
		for i = 2; i <= (*n); i++ {
			a.Set(i-1, 0, zero)
		}
		if (*n) > 1 {
			//           Generate Q(2:n,2:n)
			Zungqr(toPtr((*n)-1), toPtr((*n)-1), toPtr((*n)-1), a.Off(1, 1), lda, tau, work, lwork, &iinfo)
		}
	}
	work.SetRe(0, float64(lwkopt))
}
