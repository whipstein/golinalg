package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsytrd2stage reduces a real symmetric matrix A to real symmetric
// tridiagonal form T by a orthogonal similarity transformation:
// Q1**T Q2**T* A * Q2 * Q1 = T.
func Dsytrd2stage(vect byte, uplo mat.MatUplo, n int, a *mat.Matrix, d, e, tau, hous2 *mat.Vector, lhous2 int, work *mat.Vector, lwork int) (err error) {
	var lquery, upper bool
	var abpos, ib, kd, ldab, lhmin, lwmin, lwrk, wpos int

	//     Test the input parameters
	// wantq = vect == 'V'
	upper = uplo == Upper
	lquery = (lwork == -1) || (lhous2 == -1)

	//     Determine the block size, the workspace size and the hous size.
	kd = Ilaenv2stage(1, "Dsytrd2stage", []byte{vect}, n, -1, -1, -1)
	ib = Ilaenv2stage(2, "Dsytrd2stage", []byte{vect}, n, kd, -1, -1)
	lhmin = Ilaenv2stage(3, "Dsytrd2stage", []byte{vect}, n, kd, ib, -1)
	lwmin = Ilaenv2stage(4, "Dsytrd2stage", []byte{vect}, n, kd, ib, -1)
	//      WRITE(*,*),'Dsytrd2stage N KD UPLO LHMIN LWMIN ',N, KD, UPLO,
	//     $            LHMIN, LWMIN

	if vect != 'N' {
		err = fmt.Errorf("vect != 'N': vect='%c'", vect)
	} else if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if lhous2 < lhmin && !lquery {
		err = fmt.Errorf("lhous2 < lhmin && !lquery: lhous2=%v, lhmin=%v, lquery=%v", lhous2, lhmin, lquery)
	} else if lwork < lwmin && !lquery {
		err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
	}

	if err == nil {
		hous2.Set(0, float64(lhmin))
		work.Set(0, float64(lwmin))
	}

	if err != nil {
		gltest.Xerbla2("Dsytrd2stage", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		work.Set(0, 1)
		return
	}

	//     Determine pointer position
	ldab = kd + 1
	lwrk = lwork - ldab*n
	abpos = 1
	wpos = abpos + ldab*n
	if err = DsytrdSy2sb(uplo, n, kd, a, work.Off(abpos-1).Matrix(ldab, opts), tau, work.Off(wpos-1), lwrk); err != nil {
		gltest.Xerbla2("DsytrdSy2sb", err)
		return
	}
	if err = DsytrdSb2st('Y', vect, uplo, n, kd, work.Off(abpos-1).Matrix(ldab, opts), d, e, hous2, lhous2, work.Off(wpos-1), lwrk); err != nil {
		gltest.Xerbla2("DsytrdSb2st", err)
		return
	}

	hous2.Set(0, float64(lhmin))
	work.Set(0, float64(lwmin))

	return
}
