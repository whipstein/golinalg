package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhetrd2stage reduces a complex Hermitian matrix A to real symmetric
// tridiagonal form T by a unitary similarity transformation:
// Q1**H Q2**H* A * Q2 * Q1 = T.
func Zhetrd2stage(vect, uplo byte, n *int, a *mat.CMatrix, lda *int, d, e *mat.Vector, tau, hous2 *mat.CVector, lhous2 *int, work *mat.CVector, lwork, info *int) {
	var lquery, upper bool
	var abpos, ib, kd, ldab, lhmin, lwmin, lwrk, wpos int

	//     Test the input parameters
	(*info) = 0
	// wantq = vect == 'V'
	upper = uplo == 'U'
	lquery = ((*lwork) == -1) || ((*lhous2) == -1)

	//     Determine the block size, the workspace size and the hous size.
	kd = Ilaenv2stage(func() *int { y := 1; return &y }(), []byte("ZHETRD_2STAGE"), []byte{vect}, n, toPtr(-1), toPtr(-1), toPtr(-1))
	ib = Ilaenv2stage(func() *int { y := 2; return &y }(), []byte("ZHETRD_2STAGE"), []byte{vect}, n, &kd, toPtr(-1), toPtr(-1))
	lhmin = Ilaenv2stage(func() *int { y := 3; return &y }(), []byte("ZHETRD_2STAGE"), []byte{vect}, n, &kd, &ib, toPtr(-1))
	lwmin = Ilaenv2stage(func() *int { y := 4; return &y }(), []byte("ZHETRD_2STAGE"), []byte{vect}, n, &kd, &ib, toPtr(-1))
	//      WRITE(*,*),'ZHETRD_2STAGE N KD UPLO LHMIN LWMIN ',N, KD, UPLO,
	//     $            LHMIN, LWMIN
	//
	if vect != 'N' {
		(*info) = -1
	} else if !upper && uplo != 'L' {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*lda) < max(1, *n) {
		(*info) = -5
	} else if (*lhous2) < lhmin && !lquery {
		(*info) = -10
	} else if (*lwork) < lwmin && !lquery {
		(*info) = -12
	}

	if (*info) == 0 {
		hous2.SetRe(0, float64(lhmin))
		work.SetRe(0, float64(lwmin))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHETRD_2STAGE"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		work.Set(0, 1)
		return
	}

	//     Determine pointer position
	ldab = kd + 1
	lwrk = (*lwork) - ldab*(*n)
	abpos = 1
	wpos = abpos + ldab*(*n)
	Zhetrdhe2hb(uplo, n, &kd, a, lda, work.CMatrixOff(abpos-1, ldab, opts), &ldab, tau, work.Off(wpos-1), &lwrk, info)
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHETRD_HE2HB"), -(*info))
		return
	}
	Zhetrdhb2st('Y', vect, uplo, n, &kd, work.CMatrixOff(abpos-1, ldab, opts), &ldab, d, e, hous2, lhous2, work.Off(wpos-1), &lwrk, info)
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHETRD_HB2ST"), -(*info))
		return
	}

	hous2.SetRe(0, float64(lhmin))
	work.SetRe(0, float64(lwmin))
}
