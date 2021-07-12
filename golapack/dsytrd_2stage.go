package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsytrd2stage reduces a real symmetric matrix A to real symmetric
// tridiagonal form T by a orthogonal similarity transformation:
// Q1**T Q2**T* A * Q2 * Q1 = T.
func Dsytrd2stage(vect, uplo byte, n *int, a *mat.Matrix, lda *int, d, e, tau, hous2 *mat.Vector, lhous2 *int, work *mat.Vector, lwork, info *int) {
	var lquery, upper bool
	var abpos, ib, kd, ldab, lhmin, lwmin, lwrk, wpos int

	//     Test the input parameters
	(*info) = 0
	// wantq = vect == 'V'
	upper = uplo == 'U'
	lquery = ((*lwork) == -1) || ((*lhous2) == -1)

	//     Determine the block size, the workspace size and the hous size.
	kd = Ilaenv2stage(func() *int { y := 1; return &y }(), []byte("DSYTRD_2STAGE"), []byte{vect}, n, toPtr(-1), toPtr(-1), toPtr(-1))
	ib = Ilaenv2stage(func() *int { y := 2; return &y }(), []byte("DSYTRD_2STAGE"), []byte{vect}, n, &kd, toPtr(-1), toPtr(-1))
	lhmin = Ilaenv2stage(func() *int { y := 3; return &y }(), []byte("DSYTRD_2STAGE"), []byte{vect}, n, &kd, &ib, toPtr(-1))
	lwmin = Ilaenv2stage(func() *int { y := 4; return &y }(), []byte("DSYTRD_2STAGE"), []byte{vect}, n, &kd, &ib, toPtr(-1))
	//      WRITE(*,*),'DSYTRD_2STAGE N KD UPLO LHMIN LWMIN ',N, KD, UPLO,
	//     $            LHMIN, LWMIN

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
		hous2.Set(0, float64(lhmin))
		work.Set(0, float64(lwmin))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYTRD_2STAGE"), -(*info))
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
	DsytrdSy2sb(uplo, n, &kd, a, lda, work.MatrixOff(abpos-1, ldab, opts), &ldab, tau, work.Off(wpos-1), &lwrk, info)
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYTRD_SY2SB"), -(*info))
		return
	}
	DsytrdSb2st('Y', vect, uplo, n, &kd, work.MatrixOff(abpos-1, ldab, opts), &ldab, d, e, hous2, lhous2, work.Off(wpos-1), &lwrk, info)
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYTRD_SB2ST"), -(*info))
		return
	}

	hous2.Set(0, float64(lhmin))
	work.Set(0, float64(lwmin))
}

// 2
